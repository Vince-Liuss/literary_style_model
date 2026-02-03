import os
import gc
import torch
import wandb
from dataclasses import dataclass, field
from transformers import HfArgumentParser, EarlyStoppingCallback
from datasets import load_from_disk

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import MSELoss
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderCorrelationEvaluator,
)

# --- CONFIGURATION HUB ---
DATASET_PATH = "VibrantVista/style-judge-dataset"
BASE_OUTPUT_DIR = "your_output_directory"
PROJECT_NAME = "Style-Similarity-Finetuning"


@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "The model checkpoint to train"})
    batch_size: int = field(default=16, metadata={"help": "Per device batch size"})
    grad_accum: int = field(default=1, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate"})
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code"}
    )


def get_integrity_formatter(example):
    return {
        "sentence1": example["sentence1"],
        "sentence2": example["sentence2"],
        "label": float(example["score"]),
    }


def load_data_local(dataset_path: str):
    print(f"Loading dataset '{dataset_path}'...")
    ds = load_from_disk(dataset_path)
    keep = ["sentence1", "sentence2", "score"]
    train_ds = ds["train"].select_columns(keep)
    eval_ds = ds["validation"].select_columns(keep)
    test_ds = ds["test"].select_columns(keep)
    return train_ds, eval_ds, test_ds


def configure_training_arguments(
    output_dir: str,
    train_len: int,
    script_args: ScriptArguments,
    run_name: str,
    epochs: int,
):
    batch_size = script_args.batch_size
    grad_accum = script_args.grad_accum
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    total_effective_batch = batch_size * grad_accum * world_size
    steps_per_epoch = max(1, train_len // total_effective_batch)
    eval_steps = max(1, steps_per_epoch // 5)
    save_steps = eval_steps

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"⚙️  Config for {run_name}:")
        print(f"   - GPUs: {world_size}")
        print(f"   - Per-GPU Batch: {batch_size}")
        print(f"   - Grad Accum: {grad_accum}")
        print(f"   - Total Effective Batch: {total_effective_batch}")
        print(f"   - Total Train Examples: {train_len}")
        print(f"   - Steps Per Epoch: {steps_per_epoch}")
        print(f"   - Eval Steps: {eval_steps}")
        print(f"   - Save Steps: {save_steps}")

    # IMPORTANT:
    metric_key = "style_spearman"

    return CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=False,
        bf16=True,
        group_by_length=True,
        optim="adamw_torch_fused",
        learning_rate=script_args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to="wandb",
        run_name=run_name,
        logging_steps=1,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_drop_last=True,
    )


def train_ddp():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if is_main_process:
        print(f"📌 Training CrossEncoder (regression): {script_args.model_name}")

    train_ds, eval_ds, test_ds = load_data_local(DATASET_PATH)

    model_short = script_args.model_name.split("/")[-1]
    run_name = f"ft-{model_short}-crossencoder-reg"

    formatter = get_integrity_formatter()

    if is_main_process:
        print(f"Formatting data for {script_args.model_name}...")

    w_train = train_ds.map(
        formatter,
        remove_columns=train_ds.column_names,
        desc=f"Formatting Train {run_name}",
    )
    w_eval = eval_ds.map(
        formatter,
        remove_columns=eval_ds.column_names,
        desc=f"Formatting Eval {run_name}",
    )
    w_test = test_ds.map(
        formatter,
        remove_columns=test_ds.column_names,
        desc=f"Formatting Test {run_name}",
    )

    if set(w_train.column_names) != {"sentence1", "sentence2", "label"}:
        raise ValueError(f"Unexpected train columns: {w_train.column_names}")

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"{run_name}-ckpt")
    final_path = os.path.join(BASE_OUTPUT_DIR, f"{run_name}-final")

    if is_main_process:
        wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            reinit=True,
            config=script_args.__dict__,
            group=f"ddp-{model_short}",
        )

    model = CrossEncoder(
        script_args.model_name,
        num_labels=1,
        max_length=8192,
        trust_remote_code=script_args.trust_remote_code,
        activation_fn=torch.nn.Sigmoid(),
    )

    loss = MSELoss(model=model, activation_fn=torch.nn.Sigmoid())

    dev_evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=list(zip(w_eval["sentence1"], w_eval["sentence2"])),
        scores=[float(x) for x in w_eval["label"]],
        name="style",
    )

    args = configure_training_arguments(
        output_dir, len(w_train), script_args, run_name, epochs=5
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=w_train,
        eval_dataset=w_eval,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.003
            )
        ],
    )

    trainer.train()

    if is_main_process:
        print(f"Evaluating {run_name} on Test Set...")
        test_evaluator = CrossEncoderCorrelationEvaluator(
            sentence_pairs=list(zip(w_test["sentence1"], w_test["sentence2"])),
            scores=[float(x) for x in w_test["label"]],
            name="test",
        )
        test_metrics = test_evaluator(model)
        primary_key = test_evaluator.primary_metric  # "test_spearman"
        wandb.log({f"final_{primary_key}": float(test_metrics.get(primary_key, 0.0))})
        print(
            f"✅ {run_name} Final Test {primary_key}: {test_metrics.get(primary_key, 0.0):.6f}"
        )

        print(f"Saving model to {final_path}")
        trainer.save_model(final_path)
        wandb.finish()

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    train_ddp()


if __name__ == "__main__":
    main()
