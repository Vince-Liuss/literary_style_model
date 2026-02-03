import os
import re
import math
import torch
import numpy as np
import logging
import time
import requests
import sys
import wandb
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from accelerate import PartialState
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils import get_content_reward_messages_openchat


# Disable Tokenizers Parallelism to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    # --- Model & Data Configuration ---
    model_path: str = field(default="none", metadata={"help": "Path to the base model"})
    dataset_path: str = field(default="none", metadata={"help": "Path to the dataset"})
    output_dir: str = field(
        default="./grpo_output",
        metadata={"help": "Directory to save checkpoints and logs"},
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    log_wandb: bool = field(default=False, metadata={"help": "Enable WandB logging"})
    author_name: str = field(
        default="Twain, Mark", metadata={"help": "Author name for dataset selection"}
    )

    # --- Training Hyperparameters ---
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate (Recommended: 1e-6 to 1e-5 for GRPO)"},
    )
    epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    batch_size: int = field(
        default=1, metadata={"help": "Per-device training batch size"}
    )
    grad_acc_steps: int = field(
        default=2, metadata={"help": "Gradient accumulation steps"}
    )
    max_length: int = field(
        default=1500, metadata={"help": "Max token length for generated completions"}
    )

    # --- GRPO Specific Settings ---
    num_generations: int = field(
        default=8, metadata={"help": "Number of generations per prompt (Group Size)"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient (Control drift from ref model)"},
    )
    loss_type: str = field(
        default="bnpo",
        metadata={"help": "Loss function type: 'bnpo' (default) or 'grpo'"},
    )
    scale_rewards: str = field(default="none", metadata={"help": "Normalize rewards"})
    max_grad_norm: float = field(
        default=0.1, metadata={"help": "Max gradient norm for clipping"}
    )


nlp = spacy.blank("en")
THE_END_PATTERN = re.compile(r"THE\s?END[.!]*$", re.IGNORECASE)


def calculate_completeness_reward(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    LOWER_TARGET = 1200
    UPPER_TARGET = 1500
    MINIMUM_THRESHOLD = 500
    COMPLETENESS_PENALTY = 0.0
    OVERLENGTH_PENALTY_START = 1600
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        doc = nlp.make_doc(completion)
        attr_array = doc.to_array([spacy.attrs.IS_PUNCT, spacy.attrs.IS_SPACE])
        word_count = (attr_array.sum(axis=1) == 0).sum()
        if word_count < MINIMUM_THRESHOLD:
            rewards.append(0.0)
            continue
        if word_count < LOWER_TARGET:
            length_score = word_count / LOWER_TARGET
        elif LOWER_TARGET <= word_count <= UPPER_TARGET:
            length_score = 1.0
        elif word_count <= OVERLENGTH_PENALTY_START:
            length_score = 1.0
        else:
            length_score = max(
                0.0,
                1.0 - math.sqrt((word_count - OVERLENGTH_PENALTY_START) / UPPER_TARGET),
            )
        is_complete = bool(THE_END_PATTERN.search(completion.strip()))
        if not is_complete:
            length_score *= COMPLETENESS_PENALTY
        rewards.append(float(length_score))
    return rewards


class RewardManager:
    """
    Manages reward models by loading them on the specific local GPU.
    This bypasses FSDP wrapping entirely to prevent 'SentenceTransformer' sharding errors.
    """

    def __init__(self, args, device):
        self.device = device
        self.args = args
        self.style_model = None

    def load_models(self):
        logger.info(f"Loading Reward Models on Isolated Device: {self.device}")

        # 1. Load Sentence Transformer (Style)
        self.style_model = SentenceTransformer(
            "VibrantVista/gte-large-en-v1.5-stylejudge",
            trust_remote_code=True,
            device=str(self.device),
        )
        self.style_model.eval()

    def get_content_reward(self, prompts: Union[List[Dict], Any]) -> int:
        """
        Requests a score (0-9) from the local reward scoring server.

        BEHAVIOR:
        - Silent: No print logs or error messages.
        - Infinite: Retries forever until a valid score 0-9 is obtained.
        - Safety: Includes a 1s sleep to prevent CPU overheating during outages.
        """

        score_regex = r"^[0-9]$"

        payload = {
            "model": "openchat/openchat-3.5-0106",
            "messages": prompts,
            "max_tokens": 3,
            "temperature": 0.0,
            "stop": ["<|end_of_turn|>"],
            "structured_outputs": {"regex": score_regex},
        }

        api_url = "http://localhost:8000/v1/chat/completions"

        while True:
            try:
                response = requests.post(api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"].strip()
                        if len(content) == 1 and content.isdigit():
                            return int(content)

                # If 500/503 error, wait and retry
                elif response.status_code >= 500:
                    time.sleep(1)
                    continue
                elif response.status_code == 400:
                    logger.warning("Reward server returned 400 Bad Request.")
                    sys.exit(1)

            except Exception:
                pass

            # Mandatory wait to prevent CPU spamming while the server is down
            time.sleep(1)


def train_with_grpo(args: ScriptArguments):
    # 1. Setup Distributed State
    distributed_state = PartialState()
    device = distributed_state.device

    reward_manager = RewardManager(args, device)
    reward_manager.load_models()

    clean_name = args.author_name.replace(", ", "_").replace("'", "")
    folder_name = f"{clean_name}-{args.beta}"
    args.output_dir = os.path.join(args.output_dir, folder_name)
    if args.log_wandb:
        if distributed_state.is_main_process:
            wandb.init(
                project="author-style-agent",
                name=f"{clean_name}-{args.beta}",
                config=vars(args),
            )

    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    tokenizer.padding_side = "left"

    if args.use_lora:
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

    # 4. Define Reward Functions
    def style_reward_func(prompts, completions, **kwargs):
        target_samples = kwargs.get("sample_text", [""] * len(completions))
        factor = len(completions) // len(target_samples) if len(target_samples) else 1
        target_samples = [t for t in target_samples for _ in range(factor)]

        c_emb = reward_manager.style_model.encode(
            completions,
            batch_size=8,
            show_progress_bar=False,
        )
        t_emb = reward_manager.style_model.encode(
            target_samples,
            batch_size=8,
            show_progress_bar=False,
        )

        sims = cos_sim(c_emb, t_emb).diagonal().cpu().numpy()
        return [float(s) for s in sims]

    def content_reward_func(
        prompts: List[str], completions: List[str], **kwargs
    ) -> torch.Tensor:

        is_expanded = len(prompts) == len(completions)
        generations_per_prompt = (
            len(completions) // len(prompts) if not is_expanded else 1
        )
        formatted_prompts = []
        for i, completion in enumerate(completions):
            if not completion or not str(completion).strip():
                formatted_prompts.append("")
                continue

            if is_expanded:
                current_prompt = prompts[i]
            else:
                current_prompt = prompts[i // generations_per_prompt]
            messages = get_content_reward_messages_openchat(current_prompt, completion)
            formatted_prompts.append(messages)

        rewards = []
        for prompt in formatted_prompts:
            score = reward_manager.get_content_reward(prompt)
            score = float(score) / 10.0
            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float32)

    # 5. Dataset Setup
    def format_for_grpo(example):
        messages = [{"role": "user", "content": example["prompt"]}]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "sample_text": example["sample_text"],
        }

    with distributed_state.main_process_first():
        dataset = load_from_disk(args.dataset_path)
        train_dataset = dataset[args.author_name]
        train_dataset = train_dataset.map(format_for_grpo, batched=False)

    # 6. Trainer Setup (Full Settings Restored)
    grpo_config = GRPOConfig(
        model_init_kwargs={"dtype": torch.bfloat16},
        overwrite_output_dir=True,
        bf16=True,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate,
        ds3_gather_for_generation=True,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.2},
        warmup_ratio=0.03,
        num_generations=args.num_generations,
        max_completion_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        loss_type=args.loss_type,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        scale_rewards=args.scale_rewards,
        logging_steps=1,
        reward_weights=[0.6, 0.3, 0.1],
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        report_to="wandb" if args.log_wandb else "none",
        log_completions=True,
        save_only_model=True,
        mask_truncated_completions=False,
        use_liger_kernel=False,
        importance_sampling_level="sequence",
        use_bias_correction_kl=False,
        top_entropy_quantile=1.0,
        num_completions_to_print=0,
        generation_kwargs={
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "top_p": 0.95,
            "min_p": 0.05,
            "max_tokens": args.max_length,
        },
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15 if args.beta != 0.0 else 0.3,
        vllm_max_model_length=4096,
        vllm_tensor_parallel_size=4,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        ddp_timeout=3600,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[
            style_reward_func,
            content_reward_func,
            calculate_completeness_reward,
        ],
    )

    logger.info("Starting GRPO training")
    trainer.train()

    # 7. Robust Saving
    logger.info("Saving Model & Tokenizer...")
    trainer.save_model(args.output_dir)

    if distributed_state.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Training Complete. Model saved to {args.output_dir}")

    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    train_with_grpo(args)
