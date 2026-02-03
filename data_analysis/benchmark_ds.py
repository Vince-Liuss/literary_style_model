import os
import logging
import random
import math
import itertools
from collections import defaultdict, Counter
from datasets import load_from_disk, Dataset, DatasetDict

# --- CONFIGURATION ---
RAW_DATASET_PATH = "../data/benchmark_test_dataset"
OUTPUT_PAIRS_PATH = "../data/benchmark_fixed_pairs"
TRAIN_DATASET_PATH = "VibrantVista/style-judge-dataset"

TARGET_SUBJECTS = {
    "Adventure stories",
    "Historical fiction",
    "Young women -- Fiction",
    "Man-woman relationships -- Fiction",
}

SAMPLES_PER_SPLIT = 3000
SEED = 42
MAX_CHUNKS_FOR_PAIRING = 50  # Max chunks to use per title (stability limit)
MIN_CHUNK_LENGTH = 50  # Filter noise

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_excluded_authors(train_path):
    """
    Loads training data to identify authors that must be excluded.
    """
    if not train_path or not os.path.exists(train_path):
        return set()

    logger.info(f"🔍 Scanning training data at {train_path} for authors...")
    try:
        ds = load_from_disk(train_path)
        splits_to_check = ["train"] if "train" in ds else ds.keys()

        authors = set()
        for split in splits_to_check:
            # Check standard column names
            cols = ds[split].column_names
            if "author" in cols:
                authors.update(ds[split]["author"])
            if "author1" in cols:
                authors.update(ds[split]["author1"])
            if "author2" in cols:
                authors.update(ds[split]["author2"])

        logger.info(f"🚫 Found {len(authors)} authors in training set to exclude.")
        return authors
    except Exception as e:
        logger.warning(f"⚠️ Could not load training authors: {e}")
        return set()


def select_pairs_with_minimal_reuse(candidates, target_count, usage_counter):
    """
    Selects 'target_count' pairs from 'candidates' prioritizing chunks that haven't been used much.
    Uses a multi-pass approach relaxing the usage constraint incrementally.
    """
    selected = []
    remaining = candidates

    # Progressive Relaxation:
    # Pass 0: Pick pairs where chunks have Usage <= 0
    # Pass 1: Pick pairs where chunks have Usage <= 1
    # ... up to Usage <= 19
    for max_allowed_usage in range(20):
        if len(selected) >= target_count:
            break

        next_pass = []
        for p in remaining:
            if len(selected) >= target_count:
                break

            s1 = p["sentence1"]
            s2 = p["sentence2"]

            # Check if both chunks are within the current usage tolerance
            if (
                usage_counter[s1] <= max_allowed_usage
                and usage_counter[s2] <= max_allowed_usage
            ):
                selected.append(p)
                usage_counter[s1] += 1
                usage_counter[s2] += 1
            else:
                next_pass.append(p)

        remaining = next_pass

    return selected


def build_pairs_from_chunks(
    dataset_split, split_name, excluded_authors, sample_limit=3000
):
    logger.info(f"   🔨 Processing split: {split_name}...")

    # 1. Group Data: Subject -> Author -> Title -> List of Unique Chunks
    # Using basic dicts to avoid lambda/pickle issues
    content_map = {}

    skipped_authors = 0
    skipped_short = 0

    for row in dataset_split:
        auth = row.get("author")
        # Strict exclusion check
        if excluded_authors and auth in excluded_authors:
            skipped_authors += 1
            continue

        txt = str(row.get("chunk_text", "")).strip()
        if len(txt) < MIN_CHUNK_LENGTH:
            skipped_short += 1
            continue

        # Subject Check
        row_subs = row.get("subjects", [])
        if not isinstance(row_subs, list):
            row_subs = [row_subs]

        valid_s = None
        for s in row_subs:
            if s in TARGET_SUBJECTS:
                valid_s = s
                break

        if valid_s:
            if valid_s not in content_map:
                content_map[valid_s] = {}
            if auth not in content_map[valid_s]:
                content_map[valid_s][auth] = {}
            if row["title"] not in content_map[valid_s][auth]:
                content_map[valid_s][auth][row["title"]] = set()

            content_map[valid_s][auth][row["title"]].add(txt)

    if skipped_authors > 0:
        logger.info(
            f"      📉 Skipped {skipped_authors} rows (Train Overlap), {skipped_short} rows (Too Short)."
        )

    # 1.5. REPORT AUTHOR COUNTS
    subj_list = sorted(list(content_map.keys()))
    if not subj_list:
        logger.warning(f"      ⚠️ No valid subjects found in {split_name}.")
        return None

    logger.info(f"      📊 Author Counts per Subject ({split_name}):")
    for s in subj_list:
        num_auths = len(content_map[s])
        logger.info(f"         - {s}: {num_auths} authors")

    # 2. Generate Pairs
    candidates = {}
    total_candidates = 0

    # Initialize deterministic RNG
    rng = random.Random(SEED)

    for s in subj_list:
        auths = sorted(list(content_map[s].keys()))

        if len(auths) < 2:
            logger.warning(f"      ⚠️ Skipping Subject '{s}': Only 1 author available.")
            continue

        pos_pairs = []
        neg_pairs = []

        # --- POSITIVES (Same Author, Diff Titles) ---
        for a in auths:
            titles = sorted(list(content_map[s][a].keys()))

            # STRICT: Needs 2+ titles
            if len(titles) < 2:
                continue

            # Iterate pairs of titles
            for t1, t2 in itertools.combinations(titles, 2):
                # Sort chunks for stability
                c1 = sorted(list(content_map[s][a][t1]))
                c2 = sorted(list(content_map[s][a][t2]))

                # Cap inputs
                if len(c1) > MAX_CHUNKS_FOR_PAIRING:
                    # Deterministic sample via slice because lists are sorted
                    # To add variety, we can jump: e.g., take every 2nd element
                    # But simple slicing is safest for stability.
                    c1 = c1[:MAX_CHUNKS_FOR_PAIRING]
                if len(c2) > MAX_CHUNKS_FOR_PAIRING:
                    c2 = c2[:MAX_CHUNKS_FOR_PAIRING]

                for p in itertools.product(c1, c2):
                    pos_pairs.append(
                        {
                            "sentence1": p[0],
                            "sentence2": p[1],
                            "label": "Same Author",
                            "score": 1.0,
                            "subject": s,
                            "author1": a,
                            "author2": a,
                        }
                    )

        # --- NEGATIVES (Cross Author) ---
        for i in range(len(auths)):
            a1 = auths[i]
            a2 = auths[(i + 1) % len(auths)]  # Round robin pair

            if a1 == a2:
                continue

            # Aggregate all chunks for A1 and A2 across all their titles
            # Sort to ensure order is fixed before slicing
            c1_all = []
            for t in sorted(content_map[s][a1].keys()):
                c1_all.extend(sorted(list(content_map[s][a1][t])))

            c2_all = []
            for t in sorted(content_map[s][a2].keys()):
                c2_all.extend(sorted(list(content_map[s][a2][t])))

            # Hard limit chunks per author pair to keep negatives balanced with positives
            # (We sort, then shuffle deterministically, then slice)
            rng.shuffle(c1_all)
            rng.shuffle(c2_all)

            c1 = c1_all[:MAX_CHUNKS_FOR_PAIRING]
            c2 = c2_all[:MAX_CHUNKS_FOR_PAIRING]

            for p in itertools.product(c1, c2):
                neg_pairs.append(
                    {
                        "sentence1": p[0],
                        "sentence2": p[1],
                        "label": "Cross Author",
                        "score": 0.0,
                        "subject": s,
                        "author1": a1,
                        "author2": a2,
                    }
                )

        # Shuffle Candidates for this subject
        rng.shuffle(pos_pairs)
        rng.shuffle(neg_pairs)

        candidates[s] = {"pos": pos_pairs, "neg": neg_pairs}
        total_candidates += len(pos_pairs) + len(neg_pairs)

    # 3. Dynamic Quota (Standard Logic)
    active_subjects = sorted(list(candidates.keys()))
    if not active_subjects:
        logger.warning(f"      ⚠️ No valid pairs generated for {split_name}.")
        return None

    targets = {s: 0 for s in active_subjects}
    remaining_global_quota = min(sample_limit, total_candidates)
    active_set = set(active_subjects)

    while remaining_global_quota > 0 and active_set:
        quota_per_subj = math.ceil(remaining_global_quota / len(active_set))
        subjects_to_drop = set()

        for s in sorted(list(active_set)):
            available = len(candidates[s]["pos"]) + len(candidates[s]["neg"])
            current = targets[s]
            take = min(quota_per_subj, available - current, remaining_global_quota)

            targets[s] += take
            remaining_global_quota -= take

            if targets[s] >= available:
                subjects_to_drop.add(s)

            if remaining_global_quota == 0:
                break
        active_set -= subjects_to_drop

    # 4. Final Selection with Reuse Minimization
    all_pairs = []

    # Global usage counter for this split
    usage_counter = Counter()

    for s in active_subjects:
        pos_list = candidates[s]["pos"]
        neg_list = candidates[s]["neg"]

        target = targets[s]
        if target == 0:
            continue

        n_pos = min(len(pos_list), (target + 1) // 2)
        n_neg = target - n_pos

        # Adjust for imbalance in available data
        if n_neg > len(neg_list):
            n_neg = len(neg_list)
            n_pos = min(len(pos_list), target - n_neg)
        elif n_pos > len(pos_list):
            n_pos = len(pos_list)
            n_neg = min(len(neg_list), target - n_pos)

        # Smart Selection instead of Slicing
        selected_pos = select_pairs_with_minimal_reuse(pos_list, n_pos, usage_counter)
        selected_neg = select_pairs_with_minimal_reuse(neg_list, n_neg, usage_counter)

        all_pairs.extend(selected_pos)
        all_pairs.extend(selected_neg)

    logger.info(
        f"      ✅ Generated {len(all_pairs)} pairs for {split_name} (Target: {sample_limit})"
    )
    return Dataset.from_list(all_pairs)


def print_balance_report(dataset, split_name):
    df = dataset.to_pandas()
    total = len(df)

    logger.info(f"\n📊 --- BALANCE REPORT: {split_name} (N={total}) ---")

    subj_counts = df["subject"].value_counts()
    logger.info(f"   Subjects:")
    for subj, count in subj_counts.items():
        pct = (count / total) * 100
        logger.info(f"      - {subj}: {count} ({pct:.1f}%)")

    label_counts = df["label"].value_counts()
    logger.info("   Pairs:")
    for label, count in label_counts.items():
        pct = (count / total) * 100
        logger.info(f"      - {label}: {count} ({pct:.1f}%)")

    subj_diff = subj_counts.max() - subj_counts.min()
    label_diff = label_counts.max() - label_counts.min()

    if subj_diff <= len(subj_counts) and label_diff <= 1:
        logger.info("   ✅ STATUS: PERFECTLY BALANCED")
    else:
        logger.warning(
            f"   ⚠️ STATUS: IMBALANCED (Subj Diff: {subj_diff}, Label Diff: {label_diff})"
        )
    logger.info("----------------------------------------------------\n")


def main():
    logger.info("🚀 Starting Benchmark Dataset Builder (Stable/Deterministic Mode)")

    excluded_authors = get_excluded_authors(TRAIN_DATASET_PATH)

    if not os.path.exists(RAW_DATASET_PATH):
        logger.error(f"❌ Raw dataset not found at {RAW_DATASET_PATH}")
        return

    raw_ds = load_from_disk(RAW_DATASET_PATH)
    fixed_splits = {}

    # 1. IDENTIFY SPLITS
    # We explicitly look for numeric keys (500, 1000, etc.) to match the expected dataset structure.
    all_keys = list(raw_ds.keys())
    split_keys = [k for k in all_keys if k.isdigit()]

    # Sort numerically (500 < 1000 < 1500)
    split_keys = sorted(split_keys, key=lambda x: int(x))

    if not split_keys:
        logger.warning(
            f"⚠️ No numeric splits found in {all_keys}. Falling back to default sort."
        )
        split_keys = sorted(all_keys)
    else:
        logger.info(f"📋 Found chunk-size splits: {split_keys}")

    for split in split_keys:
        # Pass the specific split's data directly to the builder
        ds_pair = build_pairs_from_chunks(
            raw_ds[split], split, excluded_authors, SAMPLES_PER_SPLIT
        )
        if ds_pair and len(ds_pair) > 0:
            fixed_splits[split] = ds_pair
            print_balance_report(ds_pair, split)

    if fixed_splits:
        final_ds = DatasetDict(fixed_splits)
        final_ds.save_to_disk(OUTPUT_PAIRS_PATH)
        logger.info(f"💾 Benchmark Dataset saved to: {OUTPUT_PAIRS_PATH}")
    else:
        logger.error("❌ No pairs were generated.")


if __name__ == "__main__":
    main()
