import os
import gc
import torch
import wandb
from accelerate import PartialState  # Replaces torch.distributed and manual env parsing
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_from_disk
from transformers import EarlyStoppingCallback

# --- CONFIGURATION HUB ---
#change model here to finetune different models
MODEL_CONFIGS = {
    "BAAI/bge-m3": {
        "enabled": True,
        "batch_size":8,
        "grad_accum": 2,
        "lr": 2e-5,
        "chkpt": False,
        "trust_remote": True,
    },
    "AdamLucek/ModernBERT-embed-base-legal-MRL": {
        "enabled": True,
        "batch_size": 16,
        "grad_accum": 1,
        "lr": 2e-5,
        "chkpt": False,
        "trust_remote": True,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "enabled": True,
        "batch_size": 16,
        "grad_accum": 1,
        "lr": 2e-5,
        "chkpt": False,
        "trust_remote": True,
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "enabled": True,
        "batch_size": 8,
        "grad_accum": 2,
        "lr": 2e-5,
        "chkpt": False,
        "trust_remote": True,
    },
}

DATASET_PATH = "dataset_paths"
BASE_OUTPUT_DIR = "your_output_directory"
PROJECT_NAME = "Style-Similarity-Finetuning"


def get_integrity_formatter(model_name):
    """
    Ensures text is formatted for the model's native symmetric manifold.
    Bypasses retrieval-bias for stylistic judging.
    """
    # 1. NOMIC-V1.5(Instruction-Tuned)
    # Nomic expects a prefix for tasks, but for symmetric similarity,
    if any(m in model_name for m in ["nomic-embed-text-v1.5"]):

        def format_func(example):
            return {
                "sentence1": f"classification: {example['sentence1']}",
                "sentence2": f"classification: {example['sentence2']}",
                "score": example["score"],
            }

    # 2. BGE-M3, GTE-MODERNBERT, JINA-V2, ModernBERT-Base (Raw)
    else:

        def format_func(example):
            return {
                "sentence1": example["sentence1"],
                "sentence2": example["sentence2"],
                "score": example["score"],
            }

    return format_func


def load_data_local(dataset_name):
    """Loads and splits the dataset."""
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_from_disk(dataset_name)
    columns_to_keep = ["sentence1", "sentence2", "score"]
    train_dataset = dataset["train"].select_columns(columns_to_keep)
    eval_dataset = dataset["validation"].select_columns(columns_to_keep)
    test_dataset = dataset["test"].select_columns(columns_to_keep)
    return train_dataset, eval_dataset, test_dataset


def configure_training_arguments(
    output_dir, train_ds_len, model_conf, run_name, epochs
):
    """Configures optimized training arguments dynamically."""
    batch_size = model_conf["batch_size"]
    grad_accum = model_conf["grad_accum"]

    # 1. Get World Size (Total GPUs) via PartialState
    state = PartialState()
    world_size = state.num_processes

    # 2. Calculate Effective Batch Size
    total_effective_batch = batch_size * grad_accum * world_size

    # 3. Calculate True Steps Per Epoch
    steps_per_epoch = max(1, train_ds_len // total_effective_batch)

    # 4. Set Eval/Save frequency
    eval_steps = max(1, steps_per_epoch // 5)
    save_steps = eval_steps

    if state.is_main_process:
        print(f"⚙️  Config for {run_name}:")
        print(f"   - GPUs: {world_size}")
        print(f"   - Per-GPU Batch: {batch_size}")
        print(f"   - Grad Accum: {grad_accum}")
        print(f"   - Total Effective Batch: {total_effective_batch}")
        print(f"   - Total Train Examples: {train_ds_len}")
        print(f"   - Steps Per Epoch: {steps_per_epoch}")
        print(f"   - Eval Steps: {eval_steps}")
        print(f"   - Save Steps: {save_steps}")

    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=False,
        bf16=True,  # Ampere+ supported
        gradient_checkpointing=model_conf["chkpt"],
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True} if model_conf["chkpt"] else None
        ),
        group_by_length=True,
        optim="adamw_torch_fused",
        learning_rate=model_conf["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_style_spearman_cosine",
        greater_is_better=True,
        report_to="wandb",
        run_name=run_name,
        logging_steps=1,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=True,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_drop_last=True,
    )


def train_ddp():
    """
    Main training logic adapted for DDP execution using Accelerate PartialState.
    """
    # Initialize Distributed State (Handles Process Group, Device Placement, etc.)
    state = PartialState()

    # Enable high precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if state.is_main_process:
        print(f"🔧 Master process started. (Process ID: {os.getpid()})")
        print(f"   - Device: {state.device}")
        print(f"   - Num Processes: {state.num_processes}")

    # Load data once to be reused across models
    train_ds, eval_ds, test_ds = load_data_local(DATASET_PATH)

    models_to_train = [k for k, v in MODEL_CONFIGS.items() if v["enabled"]]
    if not models_to_train:
        print("No models enabled in config.")
        return

    # Iterating through models sequentially
    for model_name in models_to_train:
        model_conf = MODEL_CONFIGS[model_name]
        model_short = model_name.split("/")[-1]
        run_name = f"ft-{model_short}"

        if state.is_main_process:
            print(f"\n{'='*40}")
            print(f"🚀 Starting training for: {model_name}")
            print(f"{'='*40}\n")

        formatter = get_integrity_formatter(model_name)

        if state.is_main_process:
            print(f"Formatting data for {model_name}...")

        # Re-map dataset for specific model formatting
        w_train = train_ds.map(formatter, desc=f"Formatting Train {run_name}")
        w_eval = eval_ds.map(formatter, desc=f"Formatting Eval {run_name}")
        w_test = test_ds.map(formatter, desc=f"Formatting Test {run_name}")

        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{run_name}-ckpt")
        final_path = os.path.join(BASE_OUTPUT_DIR, f"{run_name}-final")

        if state.is_main_process:
            wandb.init(
                project=PROJECT_NAME,
                name=run_name,
                reinit=True,
                config=model_conf,
                group=f"ddp-{model_short}",
            )

        trainer = None
        model = None

        # A. Load Model
        # Use state.device to place model on correct local GPU
        model = SentenceTransformer(
            model_name,
            trust_remote_code=model_conf["trust_remote"],
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=state.device,
        )

        # B. Setup Loss
        train_loss = losses.CosineSimilarityLoss(model=model)

        # C. Evaluator
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=w_eval["sentence1"],
            sentences2=w_eval["sentence2"],
            scores=w_eval["score"],
            name="style",
        )

        # D. Args
        args = configure_training_arguments(
            output_dir, len(w_train), model_conf, run_name, epochs=5
        )

        # E. Trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=w_train,
            eval_dataset=w_eval,
            loss=train_loss,
            evaluator=evaluator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3, early_stopping_threshold=0.003
                )
            ],
        )

        # F. Train
        trainer.train()

        # G. Final Eval & Save
        # Only main process does this logic
        if state.is_main_process:
            print(f"Evaluating {run_name} on Test Set...")
            test_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=w_test["sentence1"],
                sentences2=w_test["sentence2"],
                scores=w_test["score"],
                name="test",
            )
            test_metrics = test_evaluator(model)
            spearman = test_metrics.get("test_spearman_cosine", 0.0)
            wandb.log({"final_test_spearman": spearman})
            print(f"✅ {run_name} Final Test Spearman: {spearman:.4f}")

            print(f"Saving model to {final_path}")
            trainer.save_model(final_path)

        # --- BARRIER SYNC ---
        # Waits for everyone (including Rank 0's eval) before proceeding
        state.wait_for_everyone()

        # Cleanup for next iteration
        if state.is_main_process:
            wandb.finish()

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    train_ddp()


if __name__ == "__main__":
    main()
