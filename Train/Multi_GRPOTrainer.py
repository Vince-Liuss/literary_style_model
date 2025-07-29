import os
import re
import math
import torch
import numpy as np
import logging
import wandb
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    pipeline,
    Pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from accelerate import Accelerator
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils import get_content_reward_messages_openchat, ScoreProcessor

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Arguments for GRPO training script for stylistic writing fine-tuning"""

    model_path: str = field(
        default="none",
        metadata={
            "help": "Path to the base pretrained model to fine-tune (e.g., 'meta-llama/Llama-2-7b-hf'). "
            "Should be a model already trained for text generation or instruction following."
        },
    )
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to reward model for evaluating generation quality. If None, uses reward functions "
            "defined in code. Reward model should output scores for style, coherence, completeness, etc."
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. "
            "Recommended for large models (>7B) or limited GPU memory. Reduces trainable parameters by ~99%."
        },
    )
    dataset_path: str = field(
        default="none",
        metadata={
            "help": "HuggingFace dataset path containing writing prompts for style training. "
            "Dataset should have 'prompt' field with creative writing prompts or story beginnings."
        },
    )
    max_length: int = field(
        default=1500,
        metadata={
            "help": "Maximum completion length in tokens (not including prompt). Controls how long "
            "generated stories/text can be. Typical range: 1000-2000 for creative writing."
        },
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={
            "help": "Learning rate for optimizer. GRPO is sensitive to LR. Recommended: 1e-6 to 5e-6 for "
            "large models (>7B), 5e-6 to 1e-5 for smaller models. Lower LR = more stable training."
        },
    )
    num_iterations: int = field(
        default=2,
        metadata={
            "help": "Number of GRPO training iterations/epochs. Each iteration processes the full dataset. "
            "Typical range: 1-5. Monitor reward curves - stop when rewards plateau or start declining."
        },
    )
    batch_size: int = field(
        default=1,
        metadata={
            "help": "Per-device mini-batch size for GRPO updates. Effective batch size = batch_size × "
            "grad_acc_steps × num_gpus. Start with 1-2 for large models, 4-8 for smaller models."
        },
    )
    grad_acc_steps: int = field(
        default=1,
        metadata={
            "help": "Gradient accumulation steps. Increases effective batch size without using more GPU memory. "
            "Effective batch size = batch_size × grad_acc_steps. Recommended total: 8-32 for GRPO."
        },
    )
    num_generations: int = field(
        default=8,
        metadata={
            "help": "Number of completions to generate per prompt for GRPO comparison. More generations = "
            "better policy learning but slower training. Typical range: 4-16. Higher for complex tasks."
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "KL divergence penalty coefficient. Controls trade-off between reward maximization and "
            "staying close to reference model. Higher = more conservative. Range: 0.01-0.5. "
            "Reduce if model becomes repetitive, increase if KL divergence gets too high (>1.0)."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={
            "help": "PPO-style clipping parameter for policy updates. Controls how much the policy can change "
            "per update. Lower = more conservative updates. Typical range: 0.1-0.3. Reduce if training unstable."
        },
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={
            "help": "Upper bound for dual clipping in GRPO. Should be slightly higher than epsilon. "
            "Helps prevent overly aggressive policy updates. Typical: epsilon + 0.05 to epsilon + 0.1."
        },
    )
    max_grad_norm: float = field(
        default=0.5,
        metadata={
            "help": "Maximum gradient norm for gradient clipping. Prevents exploding gradients and stabilizes "
            "training. Lower values = more stable but potentially slower learning. Typical range: 0.5-2.0."
        },
    )
    output_dir: str = field(
        default="./grpo_output",
        metadata={
            "help": "Directory for saving model checkpoints, logs, and training results. Creates subdirectories "
            "for different runs. Checkpoints saved based on save_steps parameter in GRPOConfig."
        },
    )
    log_wandb: bool = field(
        default=False,
        metadata={
            "help": "Whether to log training metrics to Weights & Biases. Logs reward scores, KL divergence, "
            "loss curves, generated text samples, and training hyperparameters for experiment tracking."
        },
    )
    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Loss formulation to use. Options: 'grpo' (sequence-level, not recommended due to length bias), "
            "'bnpo' (token-level normalization over local batch, default), 'dr_grpo' (global constant normalization). "
            "BNPO is recommended for most use cases as it handles variable-length sequences better."
        },
    )
    scale_rewards: bool = field(
        default=False,
        metadata={
            "help": "Whether to scale rewards by dividing by their standard deviation. If True, rewards are "
            "normalized to unit variance. If False, no scaling applied. Dr. GRPO paper recommends False "
            "to avoid question-level difficulty bias. Set to False for mathematical reasoning tasks."
        },
    )


def scale_similarity_score(similarity: float) -> float:
    """Scale similarity score from 0 to 1 using mean references."""
    CROSS_AUTHOR_MEAN = 0.048
    SAME_AUTHOR_MEAN = 0.701
    # Center around midpoint and scale
    midpoint = (CROSS_AUTHOR_MEAN + SAME_AUTHOR_MEAN) / 2
    scale_factor = 6.0 / (SAME_AUTHOR_MEAN - CROSS_AUTHOR_MEAN)

    # Apply sigmoid: S(x) = 1 / (1 + e^(-x))
    x = (similarity - midpoint) * scale_factor
    scaled = 1.0 / (1.0 + np.exp(-x))

    return np.clip(scaled, 0.05, 0.95)


def calculate_completeness_reward(
    prompts: List[str], completions: List[str], **kwargs
) -> torch.Tensor:
    """
    Calculates a nuanced reward that provides graded feedback for agentic training.
    """
    # Target for full reward. You can adjust this.
    TARGET_WORD_COUNT = 1400
    # Apply a 50% penalty to the score if the last sentence is incomplete.
    COMPLETENESS_PENALTY = 0.5
    # Word count below this threshold receives a 0 score to filter out noise.
    MINIMUM_THRESHOLD = 500

    # This improved regex correctly identifies various sentence endings, including those
    # with punctuation inside or outside of quotes (e.g., .", ".)
    COMPLETE_PATTERN = re.compile(
        r'\b\w+.*?((?:[.!?—]|\.{3})["\']?|["\'](?:[.!?—]|\.{3})|[!?]\.{3}["\']?)\s*$'
    )

    rewards = []
    for completion in completions:
        text = completion.strip() if completion else ""
        word_count = len(text.split())

        # 1. Hard filter for trivially short or empty completions
        if word_count < MINIMUM_THRESHOLD:
            rewards.append(0.0)
            continue

        # 2. Calculate the base score using a smooth square root curve
        length_score = math.sqrt(min(1.0, word_count / TARGET_WORD_COUNT))

        # 3. Determine if the completion is structurally sound
        lines = text.splitlines()
        last_line = lines[-1].strip() if lines else ""

        # The logic is corrected to allow single-word sentences (e.g., "Yes.")
        is_structurally_complete = len(last_line) >= 2 and COMPLETE_PATTERN.search(
            last_line
        )

        # 4. Apply penalty if structurally incomplete
        final_score = length_score
        if not is_structurally_complete:
            final_score *= COMPLETENESS_PENALTY

        rewards.append(final_score)

    return torch.tensor(rewards, dtype=torch.float32)


def setup_score_model(args: ScriptArguments) -> Tuple[Pipeline, AutoTokenizer]:
    model_name = "openchat/openchat-3.5-0106"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return (
        pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            batch_size=args.batch_size,
        ),
        tokenizer,
    )


def train_with_grpo(args: ScriptArguments) -> str:
    """Main GRPO training function using TRL's GRPO trainer"""

    style_reward_model = SentenceTransformer(args.reward_model_path)
    content_reward_pipe, content_tokenizer = setup_score_model(args)
    score_processor = ScoreProcessor(content_tokenizer)

    accelerator = Accelerator(
        log_with="wandb" if args.log_wandb else None,
    )

    if accelerator.is_main_process:
        run_name = (
            f"grpo-{args.model_path.split('/')[-1]}-{args.beta}-{args.learning_rate}"
        )
        accelerator.init_trackers(
            project_name="author-style-agent_multiGPU",
            config={**vars(args), "model": args.model_path},
            init_kwargs={"wandb": {"name": run_name, "tags": [args.loss_type]}},
        )

    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if args.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=256,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    with accelerator.main_process_first():
        logger.info("Loading dataset")
        dataset = load_dataset(args.dataset_path)

    ratio = (
        args.batch_size * args.grad_acc_steps * accelerator.num_processes
    ) / args.num_generations

    save_setps = len(dataset["train"]) // (ratio * 4)

    # Configure GRPO
    grpo_config = GRPOConfig(
        overwrite_output_dir=True,
        bf16=True,
        seed=42,
        data_seed=42,
        ds3_gather_for_generation=True,
        output_dir=args.output_dir,
        num_train_epochs=args.num_iterations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=args.max_length,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        reward_weights=[0.6, 0.3, 0.1],
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=True,
        sync_ref_model=False,
        ref_model_mixup_alpha=0.6,
        ref_model_sync_steps=128,
        remove_unused_columns=False,
        label_names=[],
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_setps,
        eval_strategy="no",
        save_only_model=True,
        run_name=(
            f"grpo-{args.model_path.split('/')[-1]}"
            if accelerator.is_main_process
            else None
        ),
        report_to="wandb" if accelerator.is_main_process and args.log_wandb else "none",
        save_total_limit=4,
        log_completions=True,
        num_completions_to_print=0,
        use_liger_kernel=True,
        generation_kwargs={
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "top_k": 50,
            "top_p": 0.95,
            "min_p": 0.02,
            "max_new_tokens": 1600,
            "min_new_tokens": 1400,
            "do_sample": True,
        },
    )

    def style_reward(
        prompts: List[str], completions: List[str], **kwargs
    ) -> List[float]:

        target_samples = kwargs["sample_text"]

        # Prepare text pairs
        completion_texts = []
        sample_texts = []

        for i, completion in enumerate(completions):
            sample_idx = i % len(target_samples)
            sample_text = target_samples[sample_idx]

            completion_texts.append(completion)
            sample_texts.append(sample_text)

        # Compute embeddings
        completion_embeddings = style_reward_model.encode(
            completion_texts, show_progress_bar=False, batch_size=args.batch_size
        )
        sample_embeddings = style_reward_model.encode(
            sample_texts, show_progress_bar=False, batch_size=args.batch_size
        )

        # Calculate cosine similarities using sentence transformers
        similarities = (
            cos_sim(completion_embeddings, sample_embeddings).diagonal().numpy()
        )

        # Scale similarities to 0-1 range
        style_rewards = [scale_similarity_score(sim) for sim in similarities]

        return style_rewards

    def content_reward(
        prompts: List[str], completions: List[str], **kwargs
    ) -> List[float]:
        # Access your dataset completion column - update column name as needed
        formatted_prompts = []
        for completion in completions:
            if not completion or not completion.strip():
                formatted_prompts.append("")
                continue

            # Get messages and apply chat template
            messages = get_content_reward_messages_openchat(completion)
            formatted_prompt = content_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted_prompt)

        # Batch process
        responses = content_reward_pipe(
            formatted_prompts,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            logits_processor=[score_processor],
            renormalize_logits=True,
        )

        # Extract scores
        rewards = []
        for i, response in enumerate(responses):
            # logger.info(f"Response: {response}")
            if not formatted_prompts[i]:
                rewards.append(0.0)
                continue

            assistant_response = response[0]["generated_text"]
            score = int(assistant_response) / 4.0
            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float32)

    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer")
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset["train"],
        reward_funcs=[
            style_reward,
            content_reward,
            calculate_completeness_reward,
        ],
    )

    # Start training
    logger.info("Starting GRPO training")
    grpo_trainer.train()

    # Finish W&B run
    if args.log_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Run GRPO training
    train_with_grpo(args)
