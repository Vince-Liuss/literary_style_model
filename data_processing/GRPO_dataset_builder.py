import logging
import random
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional
from textwrap import dedent

from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict

from prompts import PromptManager


# --- 1. Configuration ---
RAW_DATA_PATH = Path("../data/target_books")
OUTPUT_DIR = Path("../data/GRPO_dataset")

TARGET_CHUNK_SIZES = ["1500"]
MIN_CHUNK_LENGTH = 900
RANDOM_SEED = 42

BOOK_CONFIGS = [
    {
        "author": "Twain, Mark",
        "title": "Adventures of Huckleberry Finn",
        "genre": "Adventure stories",
    },
    {
        "author": "Dickens, Charles",
        "title": "A Tale of Two Cities",
        "genre": "Historical fiction",
    },
    {
        "author": "Austen, Jane",
        "title": "Pride and Prejudice",
        "genre": "Man-woman relationships -- Fiction",
    },
    {
        "author": "Hardy, Thomas",
        "title": "Tess of the d'Urbervilles: A Pure Woman",
        "genre": "Young women -- Fiction",
    },
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_canonical_book_info(
    raw_author: str, raw_title: str, configs: List[Dict]
) -> Optional[Dict[str, Any]]:
    for config in configs:
        # Loose matching or exact matching depending on raw data cleanliness
        if raw_title == config["title"] and raw_author == config["author"]:
            return config
    return None


def fetch_all_chunks(
    data_path: Path, book_configs: List[Dict], min_length: int
) -> Dict[str, List[Dict]]:
    """
    Fetches chunks from multiple split sizes if available.
    """
    logger.info(f"Scanning dataset at: {data_path}")
    ds = load_from_disk(str(data_path))

    available_keys = []
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        for k in TARGET_CHUNK_SIZES + ["train"]:
            if k in keys:
                available_keys.append(k)

    if not available_keys:
        logger.warning(f"No valid keys found in {keys}. Checking root...")
        iterable_ds = ds
        if isinstance(ds, Dataset):
            pass

    all_chunks = {cfg["author"]: [] for cfg in book_configs}

    # Iterate through found splits
    for key in available_keys:
        iterable_ds = ds[key]
        logger.info(f"  - Processing split: {key}")

        for item in tqdm(iterable_ds, desc=f"Fetching {key}", leave=False):
            if int(item.get("target_length", 0)) < min_length:
                continue
            config = get_canonical_book_info(
                item.get("author"), item.get("title"), book_configs
            )
            if config:
                clean_item = item.copy()
                clean_item.update(config)
                all_chunks[config["author"]].append(clean_item)

    return all_chunks


def build_split_datasets(author_chunks: Dict[str, List[Dict]]) -> Dict[str, List[dict]]:
    """
    Creates prompt-style pairs using PromptManager.
    Splits unique prompts 90/10 first, then cycles them to reach target size.

    Logic:
    - Train Size = 4 * Unique Train Prompts
    - Test Size = 10 * Unique Test Prompts
    """
    split_datasets = {}

    logger.info(f"Processing 4 Authors: {list(author_chunks.keys())}")

    for author, chunks in author_chunks.items():
        if not chunks:
            logger.warning(f"No chunks found for {author}. Skipping.")
            continue

        book_title = chunks[0]["title"]

        # --- 1. Get Prompts via Manager ---
        author_prompts_map = PromptManager.get_target_author_prompt(author, book_title)
        prompts = author_prompts_map.get(author, [])

        if not prompts:
            logger.warning(f"No plot prompts found for {author} in PromptManager.")
            continue

        # --- 2. 90/10 Split on Unique Prompts ---
        random.seed(RANDOM_SEED)
        unique_prompts = sorted(
            list(set(prompts))
        )  # Sort to ensure deterministic shuffle
        random.shuffle(unique_prompts)

        split_idx = int(len(unique_prompts) * 0.8)
        train_unique = unique_prompts[:split_idx]
        test_unique = unique_prompts[split_idx:]

        # --- 3. Dynamic Sizing Logic ---
        # Requirement: Train = 4x unique count, Test = 10x unique count
        n_train_target = len(train_unique) * 45
        n_test_target = len(test_unique) * 20

        logger.info(f"  - {author}: {len(unique_prompts)} unique plots.")
        logger.info(
            f"    -> Split: {len(train_unique)} unique train (x45={n_train_target}) | {len(test_unique)} unique test (x20={n_test_target})"
        )

        # --- 4. Prepare Target Chunks ---
        random.shuffle(chunks)

        def generate_rows(prompt_source_list, limit_count, desc_tag):
            if not prompt_source_list or limit_count <= 0:
                return []

            rows = []
            # Create infinite cyclers for both prompts and chunks
            prompt_cycler = itertools.cycle(prompt_source_list)
            chunk_cycler = itertools.cycle(chunks)

            pbar = tqdm(
                total=limit_count, desc=f"  - Generating {desc_tag}", leave=False
            )

            while len(rows) < limit_count:
                p = next(prompt_cycler)
                t_chunk = next(chunk_cycler)

                rows.append(
                    {
                        "prompt": p,
                        "sample_text": t_chunk["chunk_text"],
                        "genre": t_chunk["genre"],
                        "author": author,
                        "title": book_title,
                    }
                )
                pbar.update(1)

            pbar.close()
            return rows

        # --- 5. Generate Balanced Datasets ---
        author_train_rows = generate_rows(train_unique, n_train_target, "Train")
        author_test_rows = generate_rows(test_unique, n_test_target, "Test")

        # Final shuffle to mix the order of prompts
        random.shuffle(author_train_rows)
        random.shuffle(author_test_rows)

        # Assign to dict
        split_datasets[author] = author_train_rows
        split_datasets[f"{author}_test"] = author_test_rows

    return split_datasets


def main():
    all_chunks = fetch_all_chunks(RAW_DATA_PATH, BOOK_CONFIGS, MIN_CHUNK_LENGTH)

    split_map = build_split_datasets(all_chunks)

    if not split_map:
        logger.error("No training data generated.")
        return

    logger.info("-" * 40)
    logger.info("Building Final DatasetDict (Key = Author Name):")

    dataset_dict_args = {}
    total_samples = 0

    for split_key, rows in split_map.items():
        if not rows:
            continue

        logger.info(f"  - {split_key:<30} | {len(rows)} samples")
        total_samples += len(rows)

        ds = Dataset.from_list(rows)
        # Final shuffle for safety
        ds = ds.shuffle(seed=RANDOM_SEED)
        dataset_dict_args[split_key] = ds

    logger.info(f"Total Combined Samples: {total_samples}")
    logger.info("-" * 40)

    dd = DatasetDict(dataset_dict_args)

    save_path = OUTPUT_DIR.resolve()
    dd.save_to_disk(str(save_path))
    logger.info(f"Generation Complete. Saved to {save_path}")


if __name__ == "__main__":
    main()
