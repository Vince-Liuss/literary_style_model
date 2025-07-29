from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import (
    SFTTrainer,
    SFTConfig,
    setup_chat_format,
)

from datasets import load_dataset
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
        default="none", metadata={"help": "Dataset name or path"}
    )
    model: str = field(
        default="none", metadata={"help": "Model name or path"}
    )
    output_dir: str = field(
        default="./storyteller_model", metadata={"help": "Directory to save the model"}
    )
    # Training arguments
    epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=2, metadata={"help": "Batch size for training"})
    accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4, metadata={"help": "Learning rate for training"}
    )
    max_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for training"}
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
    huggingface_repo: str = field(
        default="none", metadata={"help": "Hugging Face repository name"}
    )
    # Advanced options
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing to save memory"}
    )


def train_storyteller_model(
    dataset_name="VibrantVista/SFT_dataset",
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="./storyteller_model",
    epochs=3,
    batch_size=2,
    accumulation_steps=4,
    learning_rate=2e-4,
    max_length=2048,
    use_lora=False,
    use_wandb=False,
    wandb_project="storyteller-sft",
    huggingface_repo="none",
):
    """
    Train a storyteller model using Supervised Fine-Tuning (SFT) on our dataset

    Args:
        dataset_name: HuggingFace dataset name or path to local dataset
        model_name: Base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size for training
        accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate for optimization
        max_length: Maximum sequence length for training
        use_lora: Whether to use LoRA for parameter-efficient finetuning
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)

    Returns:
        tuple: (trained model, tokenizer)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize wandb if requested
    if use_wandb:
        model_short = model_name.split("/")[-1]
        wandb_run_name = f"sft-{model_short}"

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model": model_name,
                "dataset": dataset_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "use_lora": use_lora,
                "gradient_accumulation_steps": accumulation_steps,
            },
        )
        print(f"WandB initialized: {wandb_project}/{wandb_run_name}")
        report_to = "wandb"
    else:
        report_to = "none"

    # create the output directory for the model
    output_dir = os.path.join(output_dir, wandb_run_name)

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Load tokenizer
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Ensure padding token exists
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is not None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         tokenizer.pad_token = tokenizer.bos_token

    # Load base model
    print(f"Loading base model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # setup support for flash attention
    # tokenizer.padding_side = "left"
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA for efficient fine-tuning")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            # Updated target modules for Llama 3.2 architecture
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
            # Added for better performance with Llama 3.2
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Configure training arguments
    training_args = SFTConfig(
        bf16=True,
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        max_length=max_length,
        max_seq_length=max_length,
        optim="adamw_torch_fused",
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        save_strategy="no",
        save_total_limit=1,
        warmup_ratio=0.1,
        logging_steps=1,
        report_to=report_to,
        dataset_num_proc=8,
        dataloader_num_workers=8,
        packing=False,
        pad_to_multiple_of=8,
        push_to_hub=True,
        hub_token="your_huggingface_token_here",  # Replace with your Hugging Face token
        hub_model_id=huggingface_repo,
        hub_private_repo=True,
    )

    # Initialize the SFT Trainer
    print("Initializing SFT Trainer")
    # Use formatting_func in SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train the model
    print(f"Starting SFT training for {epochs} epochs")
    trainer.train()

    # Save the best model
    print(f"Saving the model and push to hub")
    trainer.push_to_hub(commit_message="End of training")

    # Close wandb run if it was used
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Parse script arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    train_storyteller_model(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_lora=args.use_lora,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        huggingface_repo=args.huggingface_repo,
    )
