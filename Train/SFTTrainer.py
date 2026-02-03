from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import (
    SFTTrainer,
    SFTConfig,
)
from accelerate import PartialState

from datasets import load_from_disk
from dataclasses import dataclass, field
import wandb
import torch
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the storyteller model training script.
    """

    # Data & Model arguments
    dataset: str = field(
        default="dataset.json",
        metadata={"help": "Path to local dataset file (json/csv/parquet)"},
    )
    model: str = field(default="none", metadata={"help": "Model name or path"})
    output_dir: str = field(
        default="./storyteller_model", metadata={"help": "Directory to save the model"}
    )
    # Training arguments
    epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=2, metadata={"help": "Batch size for training"})

    accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )

    learning_rate: float = field(
        default=2e-5, metadata={"help": "Learning rate for training"}
    )

    max_length: int = field(
        default=4096, metadata={"help": "Maximum sequence length for training"}
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA for parameter-efficient finetuning"},
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Enable Weights & Biases logging"}
    )
    # WandB configuration
    wandb_project: str = field(
        default="storyteller-sft", metadata={"help": "WandB project name"}
    )
    # Advanced options
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


def train_storyteller_model(args):
    # Distributed settings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    distributed_state = PartialState()
    is_main_process = distributed_state.is_main_process

    model_short = args.model.split("/")[-1]

    # Setup run name and output path
    if args.use_wandb and is_main_process:
        wandb_run_name = f"sft-{model_short}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
        )
        if is_main_process:
            print(f"WandB initialized: {args.wandb_project}/{wandb_run_name}")
        report_to = "wandb"
    else:
        report_to = "none"

    # Load tokenizer
    if is_main_process:
        print(f"Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def convert_to_messages(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return {"text": text}

    # --- Load Dataset ---
    if is_main_process:
        print(f"Loading dataset from {args.dataset}")

    dataset = load_from_disk(args.dataset).shuffle(seed=42)

    dataset = dataset.map(
        convert_to_messages,
        remove_columns=[col for col in dataset.column_names if col != "text"],
        num_proc=8,
    )

    # Load base model
    if is_main_process:
        print(f"Loading base model {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        use_cache=False,
    )

    # Apply LoRA if requested
    if args.use_lora:
        if is_main_process:
            print("Applying LoRA for efficient fine-tuning")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "w1",
                "w2",
                "w3",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
            init_lora_weights="gaussian",
        )

        model = get_peft_model(model, lora_config)

        if is_main_process:
            model.print_trainable_parameters()

    # Configure training arguments
    training_args = SFTConfig(
        bf16=True,
        shuffle_dataset=True,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        max_length=args.max_length,
        dataset_num_proc=8,
        packing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        save_strategy="no",
        save_total_limit=1,
        warmup_ratio=0.05,
        logging_steps=1,
        report_to=report_to,
        dataloader_num_workers=8,
        dataset_text_field="text",
        remove_unused_columns=False,
        pad_to_multiple_of=8,
        ddp_find_unused_parameters=False,
    )
    # Initialize the SFT Trainer
    if is_main_process:
        print("Initializing SFT Trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train the model
    if is_main_process:
        print(f"Starting SFT training for {args.epochs} epochs")
    trainer.train()

    # Save the model
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.state.fsdp_plugin:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(args.output_dir)

    if is_main_process:
        print(f"Saving tokenizer to {args.output_dir}")
        tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb and is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    train_storyteller_model(args)
