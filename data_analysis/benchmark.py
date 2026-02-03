import os
import logging
import textwrap
import wandb
import torch
import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.multiprocessing as mp
import time
import json
import argparse
from queue import Empty
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --- CONFIGURATION ---
# Correct PyTorch Precision Settings
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
WANDB_PROJECT = "Style-Judge-Benchmark-MultiSize"
OUTPUT_DIR = "../data/benchmark_results"
DEFAULT_RAW_RESULTS_PATH = os.path.join(OUTPUT_DIR, "benchmark_raw_results.csv")
BENCHMARK_PAIRS_PATH = "../data/benchmark_fixed_pairs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def setup_publication_style():
    # Keep your seaborn theme (do not change)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

    # More consistent sizing/spacing + vector-safe text
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "axes.linewidth": 2.5,
            "grid.linewidth": 1.5,
            "xtick.major.width": 2.5,
            "ytick.major.width": 2.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            # Layout improvements (compact, stable)
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
            "figure.constrained_layout.hspace": 0.02,
            "figure.constrained_layout.wspace": 0.02,
        }
    )


def get_benchmark_formatter(model_path):
    m_path_lower = model_path.lower()
    if "nomic-embed-text-v1.5" in m_path_lower:
        return lambda x: f"classification: {x}"
    return lambda x: x


def determine_batch_size(model_context_type, split_name):
    try:
        size_val = int("".join(filter(str.isdigit, str(split_name))))
    except:
        size_val = 2000
    if "Short" in model_context_type:
        return 1024
    return 32 if size_val <= 1000 else 16


def gpu_worker(gpu_id, task_queue, lock, results_path):
    """
    Worker process that consumes tasks from the queue and runs them on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    logger.info(f"🔧 Worker started on GPU {gpu_id}")

    try:
        if not os.path.exists(BENCHMARK_PAIRS_PATH):
            logger.error(f"❌ Worker {gpu_id}: Pairs path not found.")
            return
        benchmark_ds = load_from_disk(BENCHMARK_PAIRS_PATH)
    except Exception as e:
        logger.error(f"❌ Worker {gpu_id} failed to load dataset: {e}")
        return

    while True:
        try:
            # Use simple get() with timeout to avoid race condition
            task = task_queue.get(timeout=5)
        except Empty:
            logger.info(f"🏁 Worker on GPU {gpu_id} finished all tasks.")
            break

        family, status, model_path, context_type, task_id = task

        try:
            logger.info(f"⚡ GPU {gpu_id} processing: {task_id}")

            # Reverted: The fix_mistral_regex flag causes crashes on BGE-M3 (Metaspace tokenizer)
            model = SentenceTransformer(
                model_path,
                trust_remote_code=True,
                device=device,
            )

            formatter = get_benchmark_formatter(model_path)
            model_results = []

            for split_name in benchmark_ds.keys():
                ds_split = benchmark_ds[split_name]
                safe_batch = determine_batch_size(context_type, split_name)

                s1 = [formatter(x) for x in ds_split["sentence1"]]
                s2 = [formatter(x) for x in ds_split["sentence2"]]

                emb1 = model.encode(
                    s1,
                    batch_size=safe_batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device,
                )
                emb2 = model.encode(
                    s2,
                    batch_size=safe_batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device,
                )

                scores = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()

                for i, score in enumerate(scores):
                    model_results.append(
                        {
                            "Model Family": family,
                            "Status": status,
                            "Split": split_name,
                            "Pair Type": ds_split[i]["label"],
                            "Predicted Score": float(score),
                            "Ground Truth": ds_split[i]["score"],
                            "Subject": ds_split[i]["subject"],
                            "Author1": (
                                ds_split[i]["author1"]
                                if "author1" in ds_split.column_names
                                else "N/A"
                            ),
                            "Author2": (
                                ds_split[i]["author2"]
                                if "author2" in ds_split.column_names
                                else "N/A"
                            ),
                        }
                    )
                del emb1, emb2

            if model_results:
                df_new = pd.DataFrame(model_results)
                with lock:
                    is_first = not os.path.exists(results_path)
                    df_new.to_csv(results_path, mode="a", header=is_first, index=False)
                logger.info(f"✅ GPU {gpu_id} saved {len(df_new)} rows for {task_id}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} Failed on {task_id}: {e}")


def load_model_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_and_log(fig, filename_base, output_dir, wandb_key):
    safe_name = filename_base.replace("/", "-").replace(" ", "_")
    png_path = os.path.join(output_dir, f"{safe_name}.png")
    pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")

    # Tight, consistent export (do not change theme)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    wandb.log({f"plots/{wandb_key}_{safe_name}": wandb.Image(png_path)})
    wandb.save(pdf_path, base_path=output_dir, policy="now")


def _wrap_ticklabels(ax, axis="y", width=28, fontsize=11):
    if axis == "y":
        labels = [
            textwrap.fill(t.get_text(), width=width) for t in ax.get_yticklabels()
        ]
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        labels = [
            textwrap.fill(t.get_text(), width=width) for t in ax.get_xticklabels()
        ]
        ax.set_xticklabels(labels, fontsize=fontsize)


def _place_legend_top(fig, handles, labels, ncol=2):
    # Compact, consistent legend for facets (keeps palette/theme)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def format_grid(g, x_label, y_label, y_limits):
    g.set_axis_labels(x_label, y_label)
    g.set_titles(col_template="{col_name}")

    if y_label == "Predicted score":
        g.set(ylim=y_limits)
    else:
        g.set(xlim=y_limits)

    # Pull legend out of the axes (more compact + avoids overlap)
    legend = getattr(g, "_legend", None)
    if legend is not None:
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.texts]
        legend.remove()
        _place_legend_top(g.fig, handles, labels, ncol=min(2, len(labels)))

    for ax in g.axes.flat:
        if y_label == "Predicted score":
            ax.axhline(0.5, ls="--", color="0.3", alpha=0.8, linewidth=1.0)
            ax.grid(True, axis="y", alpha=0.35)
        else:
            ax.axvline(0.5, ls="--", color="0.3", alpha=0.8, linewidth=1.0)
            ax.grid(True, axis="x", alpha=0.35)

        sns.despine(ax=ax, top=True, right=True)

    # Slightly reduce extra whitespace (still respects theme)
    g.fig.subplots_adjust(
        top=0.90, bottom=0.08, left=0.08, right=0.98, wspace=0.08, hspace=0.18
    )


def _facet_wrap_for_models(n_models: int) -> int:
    # More compact than fixed col_wrap=2, without changing style
    if n_models <= 2:
        return 2
    if n_models <= 4:
        return 2
    return 3


def run_inference(model_config_list, results_path):
    if not os.path.exists(BENCHMARK_PAIRS_PATH):
        logger.error(
            f"❌ Benchmark pairs not found at {BENCHMARK_PAIRS_PATH}. Run 'build_benchmark_dataset.py' first."
        )
        return

    # Check already processed in the current results file
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        # Handle empty CSV case
        if not df_existing.empty:
            processed_tasks = set(
                df_existing["Model Family"] + "_" + df_existing["Status"]
            )
            logger.info(
                f"🔄 Resuming... Found {len(processed_tasks)} processed tasks in {results_path}."
            )
        else:
            processed_tasks = set()
    else:
        processed_tasks = set()

    task_queue = mp.Queue()
    tasks_count = 0

    logger.info("📋 Preparing Task Queue from JSON config...")

    for entry in model_config_list:
        family = entry["family"]
        base_path = entry["base_path"]
        ft_path = entry.get("ft_path")  # ft_path is optional
        context_type = entry["context"]

        # 1. Base Model
        task_id_base = f"{family}_Base"
        if task_id_base not in processed_tasks:
            task_queue.put((family, "Base", base_path, context_type, task_id_base))
            tasks_count += 1

        # 2. Fine-Tuned Model (if exists)
        if ft_path:
            task_id_ft = f"{family}_Fine-Tuned"
            if os.path.exists(ft_path):
                if task_id_ft not in processed_tasks:
                    task_queue.put(
                        (family, "Fine-Tuned", ft_path, context_type, task_id_ft)
                    )
                    tasks_count += 1
            else:
                logger.warning(f"⚠️ Fine-tuned path not found: {ft_path}")

    if tasks_count == 0:
        logger.info("🎉 All tasks in config already completed.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("❌ No GPUs available for inference.")
        return

    logger.info(f"🔥 Launching {num_gpus} workers for {tasks_count} pending tasks...")

    lock = mp.Lock()
    processes = []

    for i in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(i, task_queue, lock, results_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("✅ All workers completed.")


def generate_reports(results_path):
    if not os.path.exists(results_path):
        logger.warning(f"No results found at {results_path} to report.")
        return

    df = pd.read_csv(results_path)
    if df.empty:
        logger.warning("Results file is empty.")
        return

    logger.info(f"📊 Generating Reports for {len(df)} rows from {results_path}...")

    STATUS_ORDER = ["Base", "Fine-Tuned"]

    # --- Table 1: Overall Stats ---
    rows_overall = []
    groups = df.groupby(["Model Family", "Status", "Split", "Pair Type"])
    for (fam, stat, split, ptype), group in groups:
        sc = group["Predicted Score"]
        rows_overall.append(
            {
                "Model": fam,
                "Status": stat,
                "Split": split,
                "Analysis Type": ptype,
                "Comparisons": len(group),
                "Mean Sim.": round(sc.mean(), 4),
                "Std. Dev.": round(sc.std(), 4),
                "Median": round(sc.median(), 4),
                "Min": round(sc.min(), 4),
                "Max": round(sc.max(), 4),
                "Q25": round(sc.quantile(0.25), 4),
                "Q75": round(sc.quantile(0.75), 4),
            }
        )
    df_over = pd.DataFrame(rows_overall)
    df_over["_sort"] = df_over["Status"].apply(lambda x: 1 if x == "Fine-Tuned" else 0)
    df_over = df_over.sort_values(
        ["Model", "Split", "_sort", "Analysis Type"],
        ascending=[True, True, True, False],
    ).drop(columns=["_sort"])

    table_path = os.path.join(OUTPUT_DIR, "paper_table_overall.csv")
    df_over.to_csv(table_path, index=False)
    wandb.log({"tables/Overall_Stats": wandb.Table(dataframe=df_over)})
    wandb.save(table_path, base_path=OUTPUT_DIR, policy="now")

    # --- Table 2: Subject Breakdown ---
    rows_subj = []
    g_subj = df.groupby(["Model Family", "Status", "Split", "Subject", "Pair Type"])
    for (fam, stat, split, subj, ptype), group in g_subj:
        sc = group["Predicted Score"]
        if len(sc) < 2:
            continue
        rows_subj.append(
            {
                "Model": fam,
                "Status": stat,
                "Split": split,
                "Subject": subj,
                "Analysis Type": ptype,
                "Comparisons": len(group),
                "Mean Sim.": round(sc.mean(), 4),
                "Std. Dev.": round(sc.std(), 4),
                "Median": round(sc.median(), 4),
                "Q25": round(sc.quantile(0.25), 4),
                "Q75": round(sc.quantile(0.75), 4),
            }
        )
    df_sub = pd.DataFrame(rows_subj)

    sub_table_path = os.path.join(OUTPUT_DIR, "paper_table_subjects.csv")
    df_sub.to_csv(sub_table_path, index=False)
    wandb.log({"tables/Subject_Stats": wandb.Table(dataframe=df_sub)})
    wandb.save(sub_table_path, base_path=OUTPUT_DIR, policy="now")

    # Keep your plotting theme; just unify a few compact rcParams locally
    sns.set_theme(
        context="paper", style="whitegrid", font="DejaVu Sans", font_scale=1.0
    )
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    HUE_ORDER = ["Same Author", "Cross Author"]
    palette = dict(zip(HUE_ORDER, sns.color_palette("colorblind", n_colors=2)))
    ymin, ymax = -0.02, 1.02

    # --- Plots ---
    # 1) Splits: vertical boxplots, compact facets + consistent legend
    splits = sorted(df["Split"].unique())
    n_models = df["Model Family"].nunique()
    col_wrap = _facet_wrap_for_models(n_models)

    for split in splits:
        d_split = df[df["Split"] == split]
        if d_split.empty:
            continue

        g = sns.catplot(
            data=d_split,
            x="Status",
            y="Predicted Score",
            hue="Pair Type",
            col="Model Family",
            kind="box",
            col_wrap=col_wrap,
            height=2.55,  # slightly tighter than 2.8
            aspect=1.25,  # slightly tighter than 1.35
            palette=palette,
            hue_order=HUE_ORDER,
            order=STATUS_ORDER,
            showfliers=False,
            linewidth=0.8,
            whis=(5, 95),
        )

        format_grid(g, "", "Predicted score", (ymin, ymax))
        save_and_log(g.fig, f"box_{split}", OUTPUT_DIR, "box")

    # 2) Per-family: horizontal boxplots with stable facet sizing + wrapped labels
    for fam in sorted(df["Model Family"].unique()):
        d_fam = df[df["Model Family"] == fam]
        if d_fam.empty:
            continue

        n_subjects = d_fam["Subject"].nunique()

        # Compact, stable sizing
        facet_height = min(6.5, max(3.0, 2.5 + 0.28 * n_subjects))  # inches
        aspect_ratio = 1.75  # width = height * aspect

        # Keep theme/palette, but locally tame the oversized/bold global rcParams
        with (
            sns.plotting_context("paper", font_scale=1.0),
            mpl.rc_context(
                {
                    "axes.titlesize": 16,
                    "axes.labelsize": 14,
                    "xtick.labelsize": 11,
                    "ytick.labelsize": 11,
                    "font.weight": "bold",  # keep your style
                    "axes.labelweight": "bold",
                    "axes.titleweight": "bold",
                }
            ),
        ):
            g2 = sns.catplot(
                data=d_fam,
                y="Subject",
                x="Predicted Score",
                hue="Pair Type",
                col="Status",
                kind="box",
                palette=palette,
                hue_order=HUE_ORDER,
                col_order=STATUS_ORDER,
                height=facet_height,
                aspect=aspect_ratio,
                sharey=True,
                showfliers=False,
                linewidth=0.9,
                whis=(5, 95),
                orient="h",
                legend=True,  # create legend so we can move it cleanly
                legend_out=False,
            )

            # Titles + axis labels
            g2.set_axis_labels("Predicted score", "")
            g2.set_titles(col_template="{col_name}", y=1.02)

            # X-limits (keep your zoom logic, but make it consistent across facets)
            data_min = float(d_fam["Predicted Score"].min())
            g2.set(xlim=(max(0.0, data_min - 0.08), 1.02))

            # Grid + reference line + label wrapping + reduce y-label heaviness
            for ax in g2.axes.flat:
                ax.xaxis.grid(True, alpha=0.30)
                ax.axvline(0.5, ls="--", color="0.3", alpha=0.8, linewidth=1.0)
                sns.despine(ax=ax, top=True, right=True)

                # Wrap only left facet labels
                if ax.get_subplotspec().is_first_col():
                    labels = [
                        textwrap.fill(t.get_text(), width=26)
                        for t in ax.get_yticklabels()
                    ]
                    ax.set_yticklabels(labels)
                    for t in ax.get_yticklabels():
                        t.set_fontweight("normal")  # key: stops the “shouting” look

            # Move legend to bottom (prevents title collision, removes big top whitespace)
            sns.move_legend(
                g2,
                "lower center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=2,
                frameon=False,
                title=None,
            )

            # Tighten layout for long y labels (left margin) + legend (bottom)
            g2.fig.subplots_adjust(
                left=0.34,  # more room for wrapped subjects
                right=0.98,
                top=0.92,
                bottom=0.20,
                wspace=0.10,
            )

            # Save
            safe_fam = fam.replace("/", "-").replace(" ", "_")
            png_path = os.path.join(OUTPUT_DIR, f"subj_{safe_fam}.png")
            pdf_path = os.path.join(OUTPUT_DIR, f"subj_{safe_fam}.pdf")

            g2.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
            g2.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            plt.close(g2.fig)

            wandb.log({f"plots/subj_{safe_fam}": wandb.Image(png_path)})
            wandb.save(pdf_path, base_path=OUTPUT_DIR, policy="now")


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Run Benchmark Inference with JSON Config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config JSON"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_RAW_RESULTS_PATH,
        help="Path to save raw results",
    )
    args = parser.parse_args()

    wandb.init(project=WANDB_PROJECT, name="Benchmark-Exec-Final")
    setup_publication_style()

    # Load config
    model_config_list = load_model_config(args.config)

    run_inference(model_config_list, args.output_csv)
    generate_reports(args.output_csv)
    wandb.finish()


if __name__ == "__main__":
    main()
