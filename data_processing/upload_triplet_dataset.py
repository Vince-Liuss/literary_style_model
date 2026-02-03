import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import logging
import itertools

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# I/O & Pre-Checks
# ---------------------------------------------------------------------------


def load_raw_data(file_path: str) -> list:
    """Load raw JSONL chunk data."""
    logger.info(f"Loading raw dataset from: {file_path}")
    data: list[dict] = []

    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return []

    with jsonlines.open(file_path) as reader:
        for line in tqdm(reader, desc="Loading data"):
            raw_label = line.get("chunk_label")
            if raw_label is None:
                raw_label = line.get("label", f"chunk_{len(data)}")

            mask_ratio = float(line.get("mask_ratio", 0.0))

            entry = {
                "chunk_text": (line.get("chunk_text") or ""),
                "refilled_text": (line.get("refilled_text") or ""),
                "author": line.get("author"),
                "gutenberg_id": line.get("gutenberg_id"),
                "title": line.get("title"),
                "subjects": line.get("subjects", []),
                "mask_ratio": mask_ratio,
                "base_label": str(raw_label),
            }
            data.append(entry)

    logger.info(f"Loaded {len(data)} raw chunks")
    return data


def detect_incomplete_chunks(data: list[dict]) -> set[str]:
    """Scans for 'stuck' chunks that don't have the full ratio set (0.1-0.9)."""
    logger.info("--- SCANNING FOR INCOMPLETE CHUNKS ---")
    ratios_by_label = defaultdict(set)
    for entry in data:
        if entry["mask_ratio"] > 0.0:
            r = round(entry["mask_ratio"], 1)
            ratios_by_label[entry["base_label"]].add(r)

    expected_ratios = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    blacklist = set()

    for label, found_ratios in ratios_by_label.items():
        if not expected_ratios.issubset(found_ratios):
            blacklist.add(label)

    if blacklist:
        logger.warning(f"🚫 AUTO-DETECTED {len(blacklist)} INCOMPLETE CHUNKS.")
    else:
        logger.info("✅ No incomplete chunks detected.")

    return blacklist


def analyze_lazy_copies(data: list[dict]) -> None:
    """Analyzes the dataset for input-output identity."""
    total_refilled = 0
    exact_copies = 0
    for entry in data:
        if entry["mask_ratio"] > 0.0:
            total_refilled += 1
            if entry["chunk_text"].strip() == entry["refilled_text"].strip():
                exact_copies += 1

    if total_refilled > 0:
        logger.info(f"Lazy Copy Rate: {(exact_copies / total_refilled) * 100:.4f}%")


# ---------------------------------------------------------------------------
# Core Builder Class
# ---------------------------------------------------------------------------


class RobustStyleDatasetBuilder:
    """Style-similarity dataset builder with Zone-Based Style Inertia."""

    CORE_SUBJECTS = [
        "Adventure stories",
        "Historical fiction",
        "Young women -- Fiction",
        "Man-woman relationships -- Fiction",
    ]

    def __init__(
        self,
        raw_data: list[dict],
        cleaned_metadata_path: str,
        auto_blacklist: set[str] = None,
    ):
        self.blacklist = auto_blacklist or set()
        self.meta_by_gid = self._load_cleaned_metadata(cleaned_metadata_path)
        self.df = self.process_raw_data(raw_data)

        if self.df.empty:
            logger.warning("DataFrame is empty after processing!")
            return

        self.df = self.df.reset_index(drop=True)
        self.df["chunk_id"] = np.arange(len(self.df), dtype=int)

        # Pre-compute valid ratio pairs for every target bin
        self.same_author_map, self.cross_author_map = self._precompute_ratio_maps()

        logger.info(
            f"Dataset: {len(self.df)} chunks, "
            f"{self.df['author_id'].nunique()} authors, "
            f"subjects: {self.df['subject'].value_counts().to_dict()}"
        )

    def _load_cleaned_metadata(self, path: str) -> dict:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Cleaned metadata file not found: {path}")

        with path_obj.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        books = meta.get("books", meta)
        meta_by_gid: dict[str, dict] = {}
        for b in books:
            gid = b.get("gutenberg_id")
            if gid is None:
                continue
            gid_str = str(gid)
            subjects = b.get("subjects", []) or []
            subjects = [s for s in subjects if s in self.CORE_SUBJECTS]
            if not subjects:
                continue

            meta_by_gid[gid_str] = {
                "author": b.get("author", "Unknown"),
                "subjects": subjects,
                "title": b.get("title", ""),
            }
        return meta_by_gid

    def process_raw_data(self, data: list[dict]) -> pd.DataFrame:
        processed: list[dict] = []
        seen_original_content = set()

        for entry in tqdm(data, desc="Processing raw data"):
            base_label = entry.get("base_label")
            if base_label in self.blacklist:
                continue

            gid = str(entry.get("gutenberg_id"))
            meta = self.meta_by_gid.get(gid)
            if not meta:
                continue

            subject = meta["subjects"][0]
            author_id = meta["author"]
            mask_ratio = entry.get("mask_ratio", 0.0)

            chunk_text = (entry.get("chunk_text") or "").replace("*", "").strip()

            if chunk_text:
                orig_key = (gid, chunk_text)
                if orig_key not in seen_original_content:
                    seen_original_content.add(orig_key)
                    processed.append(
                        {
                            "text": chunk_text,
                            "author_id": author_id,
                            "book_id": gid,
                            "subject": subject,
                            "mask_ratio": 0.0,
                            "is_refilled": False,
                            "label": base_label,
                        }
                    )

            if mask_ratio > 0.0:
                refilled_text = (
                    (entry.get("refilled_text") or "").replace("*", "").strip()
                )
                if refilled_text:
                    processed.append(
                        {
                            "text": refilled_text,
                            "author_id": author_id,
                            "book_id": gid,
                            "subject": subject,
                            "mask_ratio": mask_ratio,
                            "is_refilled": True,
                            "label": f"{base_label}_{mask_ratio}",
                        }
                    )

        return pd.DataFrame(processed)

    # ---------------------------------------------------------------------------
    # Math & Scoring Logic
    # ---------------------------------------------------------------------------

    @staticmethod
    def calculate_integrity(r1: float, r2: float) -> float:
        q1 = 1.0 - (r1**2)
        q2 = 1.0 - (r2**2)
        return np.sqrt(q1 * q2)

    @staticmethod
    def get_zoned_score(integrity: float, same_author: bool = True) -> float:
        min_i = 0.19
        safe_integrity = max(integrity, min_i)
        norm_integrity = (safe_integrity - min_i) / (1.0 - min_i)

        if same_author:
            return 0.5 + (0.5 * norm_integrity)
        else:
            return 0.5 * (1.0 - norm_integrity)

    @staticmethod
    def bin_midrange_score(s: float) -> float:
        s_bin = round(float(s) * 10.0) / 10.0
        return max(0.1, min(0.9, s_bin))

    def _precompute_ratio_maps(self) -> tuple[dict, dict]:
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        same_map = defaultdict(list)
        cross_map = defaultdict(list)

        for r1, r2 in itertools.combinations_with_replacement(ratios, 2):
            integrity = self.calculate_integrity(r1, r2)
            s_same = self.bin_midrange_score(self.get_zoned_score(integrity, True))
            same_map[s_same].append((r1, r2))
            if r1 > 0.0 and r2 > 0.0:
                s_cross = self.bin_midrange_score(
                    self.get_zoned_score(integrity, False)
                )
                cross_map[s_cross].append((r1, r2))

        return dict(same_map), dict(cross_map)

    # ---------------------------------------------------------------------------
    # Splits Logic
    # ---------------------------------------------------------------------------

    def create_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Creating splits (Prioritizing Multi-Book Authors for Val/Test)...")
        df = self.df
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        # Group authors
        subj_per_author = df.groupby("author_id")["subject"].nunique()
        multi_subject_authors = set(subj_per_author[subj_per_author > 1].index)

        train_authors, val_authors, test_authors = (
            set(multi_subject_authors),
            set(),
            set(),
        )

        df_single = df[~df["author_id"].isin(multi_subject_authors)]

        for subj, subj_df in df_single.groupby("subject"):
            auth_stats = subj_df.groupby("author_id").agg(
                {"chunk_id": "count", "book_id": "nunique"}
            )
            auth_stats.columns = ["chunk_count", "book_count"]

            total_chunks = auth_stats["chunk_count"].sum()
            targets = [0.76 * total_chunks, 0.12 * total_chunks, 0.12 * total_chunks]
            currents = [0, 0, 0]

            authors_list = []
            for aid, row in auth_stats.iterrows():
                authors_list.append(
                    {
                        "aid": aid,
                        "chunks": row["chunk_count"],
                        "books": row["book_count"],
                    }
                )

            # Sort: Fewest books first (Train), Most books last (Val/Test)
            random.shuffle(authors_list)
            authors_list.sort(key=lambda x: x["books"])

            for auth in authors_list:
                c = auth["chunks"]
                costs = []
                for i in range(3):
                    new_curr = currents.copy()
                    new_curr[i] += c
                    cost = sum(
                        ((new_curr[j] - targets[j]) / max(targets[j], 1)) ** 2
                        for j in range(3)
                    )
                    costs.append((i, cost))

                best_idx = min(costs, key=lambda x: x[1])[0]
                if best_idx == 0:
                    train_authors.add(auth["aid"])
                elif best_idx == 1:
                    val_authors.add(auth["aid"])
                else:
                    test_authors.add(auth["aid"])
                currents[best_idx] += c

        df_train = df[df["author_id"].isin(train_authors)].copy()
        df_val = df[df["author_id"].isin(val_authors)].copy()
        df_test = df[df["author_id"].isin(test_authors)].copy()

        return df_train, df_val, df_test

    # ---------------------------------------------------------------------------
    # Pair Builders (TARGET DRIVEN)
    # ---------------------------------------------------------------------------

    def build_extreme_pairs_for_split(
        self, df_pool: pd.DataFrame, name: str, target_per_subj: int
    ) -> tuple[pd.DataFrame, set[int], set[tuple[int, int]]]:
        """Builds 1.0 and 0.0 pairs trying to meet target_per_subj."""
        logger.info(
            f"{name}: building extreme (0.0/1.0) pairs (Target {target_per_subj}/subj)..."
        )
        df_orig = df_pool[df_pool["is_refilled"] == False].copy()
        if df_orig.empty:
            return pd.DataFrame(), set(), set()

        pairs, used_ids, seen_pairs = [], set(), set()
        # Local usage tracker to enforce freshness first, then relax
        orig_usage = Counter()
        rng = np.random.default_rng(SEED)

        for subj in tqdm(df_orig["subject"].unique(), desc=f"{name} extremes"):
            df_subj = df_orig[df_orig["subject"] == subj]

            # --- 1.0 Pairs (Same Author) ---
            generated_1 = 0
            # Dynamic relaxation loop
            cap = 5
            max_cap = 500

            # Identify valid author groups first
            valid_groups = []
            for _, df_a in df_subj.groupby("author_id"):
                books = df_a["book_id"].unique()
                if len(books) >= 2:
                    valid_groups.append(df_a)

            while generated_1 < target_per_subj and cap <= max_cap:
                progress_made = False
                rng.shuffle(valid_groups)

                for df_a in valid_groups:
                    if generated_1 >= target_per_subj:
                        break
                    by_book = {
                        b: g.reset_index(drop=True) for b, g in df_a.groupby("book_id")
                    }
                    books = list(by_book.keys())

                    # Try pair
                    for i in range(len(books)):
                        for j in range(i + 1, len(books)):
                            b1, b2 = books[i], books[j]
                            d1 = by_book[b1]
                            d2 = by_book[b2]

                            # Find chunks under current cap
                            cands1 = d1[
                                d1["chunk_id"].map(lambda x: orig_usage[x] < cap)
                            ]
                            cands2 = d2[
                                d2["chunk_id"].map(lambda x: orig_usage[x] < cap)
                            ]

                            if cands1.empty or cands2.empty:
                                continue

                            # Pick one pair
                            r1 = cands1.sample(
                                1, random_state=rng.integers(0, 1e9)
                            ).iloc[0]
                            r2 = cands2.sample(
                                1, random_state=rng.integers(0, 1e9)
                            ).iloc[0]
                            c1, c2 = int(r1["chunk_id"]), int(r2["chunk_id"])
                            key = tuple(sorted((c1, c2)))

                            if key in seen_pairs:
                                continue
                            seen_pairs.add(key)
                            used_ids.update([c1, c2])
                            orig_usage[c1] += 1
                            orig_usage[c2] += 1

                            pairs.append(
                                {
                                    "sentence1": r1["text"],
                                    "sentence2": r2["text"],
                                    "score": 1.0,
                                    "subject": subj,
                                    "author1": r1["author_id"],
                                    "author2": r2["author_id"],
                                    "book1": r1["book_id"],
                                    "book2": r2["book_id"],
                                    "id1": r1["label"],
                                    "id2": r2["label"],
                                    "pair_type": "extreme_1.0",
                                }
                            )
                            generated_1 += 1
                            progress_made = True
                            break  # Move to next author
                        if progress_made:
                            break

                if not progress_made:
                    cap *= 2  # Relax cap if stuck
                elif generated_1 < target_per_subj:
                    pass  # Keep going with same cap

            # --- 0.0 Pairs (Diff Author) ---
            generated_0 = 0
            cap = 5
            authors = df_subj["author_id"].unique()

            while generated_0 < target_per_subj and cap <= max_cap and len(authors) > 1:
                progress_made = False
                # Pair random authors
                auth_list = list(authors)
                rng.shuffle(auth_list)

                # Pair adjacent authors in shuffled list
                for i in range(0, len(auth_list) - 1, 2):
                    if generated_0 >= target_per_subj:
                        break
                    a1, a2 = auth_list[i], auth_list[i + 1]

                    d1 = df_subj[df_subj["author_id"] == a1]
                    d2 = df_subj[df_subj["author_id"] == a2]

                    cands1 = d1[d1["chunk_id"].map(lambda x: orig_usage[x] < cap)]
                    cands2 = d2[d2["chunk_id"].map(lambda x: orig_usage[x] < cap)]

                    if cands1.empty or cands2.empty:
                        continue

                    r1 = cands1.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
                    r2 = cands2.sample(1, random_state=rng.integers(0, 1e9)).iloc[0]
                    c1, c2 = int(r1["chunk_id"]), int(r2["chunk_id"])

                    key = tuple(sorted((c1, c2)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    used_ids.update([c1, c2])
                    orig_usage[c1] += 1
                    orig_usage[c2] += 1

                    pairs.append(
                        {
                            "sentence1": r1["text"],
                            "sentence2": r2["text"],
                            "score": 0.0,
                            "subject": subj,
                            "author1": r1["author_id"],
                            "author2": r2["author_id"],
                            "book1": r1["book_id"],
                            "book2": r2["book_id"],
                            "id1": r1["label"],
                            "id2": r2["label"],
                            "pair_type": "extreme_0.0",
                        }
                    )
                    generated_0 += 1
                    progress_made = True

                if not progress_made:
                    cap *= 2

        return pd.DataFrame(pairs), used_ids, seen_pairs

    def build_midrange_pairs_for_split(
        self,
        df_pool: pd.DataFrame,
        name: str,
        extreme_original_ids: set[int],
        seen_pairs: set[tuple[int, int]],
        target_per_subj_bin: int,
        usage_counter: Counter = None,
    ) -> pd.DataFrame:
        logger.info(
            f"{name}: building SAME-AUTHOR midrange (Target {target_per_subj_bin})..."
        )
        pairs = []
        rng = np.random.default_rng(SEED)
        mid_usage = usage_counter if usage_counter is not None else Counter()

        for subj, subj_df in tqdm(
            df_pool.groupby("subject"), desc=f"{name} same-author"
        ):
            priority_bins = [0.6, 0.7, 0.8, 0.9]  # No 0.5

            # Pre-group data by author/book/ratio
            subj_data = {}
            for aid, grp in subj_df.groupby("author_id"):
                books = grp["book_id"].unique()
                if len(books) < 2:
                    continue

                by_book_ratio = defaultdict(lambda: defaultdict(list))
                for r in grp.to_dict("records"):
                    ratio = round(float(r["mask_ratio"]), 1)
                    if ratio == 0.0 and r["chunk_id"] in extreme_original_ids:
                        continue
                    by_book_ratio[r["book_id"]][ratio].append(r)
                subj_data[aid] = (list(books), by_book_ratio)

            for target_bin in priority_bins:
                required_ratio_pairs = self.same_author_map.get(target_bin, [])
                if not required_ratio_pairs:
                    continue

                generated_count = 0
                current_cap = 3  # Start low for freshness
                max_cap = 500  # Hard limit

                valid_authors = list(subj_data.keys())

                while generated_count < target_per_subj_bin and current_cap < max_cap:
                    progress_made = False
                    rng.shuffle(valid_authors)
                    random.shuffle(required_ratio_pairs)

                    for author_id in valid_authors:
                        if generated_count >= target_per_subj_bin:
                            break
                        books, by_book_ratio = subj_data[author_id]

                        # Try to find ONE pair for this author
                        found_pair = False

                        # Randomize book order
                        rng.shuffle(books)

                        for r1_req, r2_req in required_ratio_pairs:
                            if found_pair:
                                break
                            r1_req, r2_req = round(r1_req, 1), round(r2_req, 1)

                            for b1 in books:
                                if found_pair:
                                    break
                                cands1 = [
                                    c
                                    for c in by_book_ratio[b1].get(r1_req, [])
                                    if mid_usage[int(c["chunk_id"])] < current_cap
                                ]
                                if not cands1:
                                    continue

                                for b2 in books:
                                    if b1 == b2:
                                        continue
                                    cands2 = [
                                        c
                                        for c in by_book_ratio[b2].get(r2_req, [])
                                        if mid_usage[int(c["chunk_id"])] < current_cap
                                    ]
                                    if not cands2:
                                        continue

                                    # Found candidates
                                    rec1 = rng.choice(cands1)
                                    rec2 = rng.choice(cands2)
                                    c1, c2 = int(rec1["chunk_id"]), int(
                                        rec2["chunk_id"]
                                    )

                                    key = tuple(sorted((c1, c2)))
                                    if key in seen_pairs:
                                        continue
                                    seen_pairs.add(key)

                                    pairs.append(
                                        {
                                            "sentence1": rec1["text"],
                                            "sentence2": rec2["text"],
                                            "score": target_bin,
                                            "subject": subj,
                                            "author1": rec1["author_id"],
                                            "author2": rec2["author_id"],
                                            "book1": rec1["book_id"],
                                            "book2": rec2["book_id"],
                                            "id1": rec1["label"],
                                            "id2": rec2["label"],
                                            "pair_type": f"mid_{target_bin}",
                                        }
                                    )
                                    mid_usage[c1] += 1
                                    mid_usage[c2] += 1
                                    generated_count += 1
                                    progress_made = True
                                    found_pair = True
                                    break

                    # If we did a full pass of authors and are stuck, relax cap
                    if not progress_made:
                        current_cap *= 2

        return pd.DataFrame(pairs)

    def build_cross_author_pairs(
        self,
        df_pool: pd.DataFrame,
        name: str,
        seen_pairs: set[tuple[int, int]],
        target_per_subj_bin: int,
        usage_counter: Counter = None,
    ) -> pd.DataFrame:
        logger.info(
            f"{name}: building CROSS-AUTHOR pairs (Target {target_per_subj_bin})..."
        )
        pairs = []
        ca_usage = usage_counter if usage_counter is not None else Counter()
        rng = np.random.default_rng(SEED)

        for subj, subj_df in tqdm(
            df_pool.groupby("subject"), desc=f"{name} cross-author"
        ):
            authors = list(subj_df["author_id"].unique())
            if len(authors) < 2:
                continue

            # Pre-index
            auth_data = defaultdict(lambda: defaultdict(list))
            for r in subj_df[subj_df["is_refilled"]].to_dict("records"):
                ratio = round(float(r["mask_ratio"]), 1)
                auth_data[r["author_id"]][ratio].append(r)

            priority_bins = [0.4, 0.3, 0.2, 0.1, 0.0]

            for target_bin in priority_bins:
                required_ratio_pairs = self.cross_author_map.get(target_bin, [])
                if not required_ratio_pairs:
                    continue

                generated_count = 0
                current_cap = 3
                max_cap = 500

                while generated_count < target_per_subj_bin and current_cap < max_cap:
                    progress_made = False
                    rng.shuffle(authors)
                    random.shuffle(required_ratio_pairs)

                    # Iterate adjacent authors
                    for i in range(len(authors)):
                        if generated_count >= target_per_subj_bin:
                            break
                        # Wrap around
                        a1 = authors[i]
                        a2 = authors[(i + 1) % len(authors)]

                        found_pair = False

                        for r1_req, r2_req in required_ratio_pairs:
                            if found_pair:
                                break
                            r1_req, r2_req = round(r1_req, 1), round(r2_req, 1)

                            cands1 = [
                                c
                                for c in auth_data[a1].get(r1_req, [])
                                if ca_usage[int(c["chunk_id"])] < current_cap
                            ]
                            cands2 = [
                                c
                                for c in auth_data[a2].get(r2_req, [])
                                if ca_usage[int(c["chunk_id"])] < current_cap
                            ]

                            if not cands1 or not cands2:
                                continue

                            rec1 = rng.choice(cands1)
                            rec2 = rng.choice(cands2)
                            c1, c2 = int(rec1["chunk_id"]), int(rec2["chunk_id"])

                            key = tuple(sorted((c1, c2)))
                            if key in seen_pairs:
                                continue
                            seen_pairs.add(key)

                            pairs.append(
                                {
                                    "sentence1": rec1["text"],
                                    "sentence2": rec2["text"],
                                    "score": target_bin,
                                    "subject": subj,
                                    "author1": rec1["author_id"],
                                    "author2": rec2["author_id"],
                                    "book1": rec1["book_id"],
                                    "book2": rec2["book_id"],
                                    "id1": rec1["label"],
                                    "id2": rec2["label"],
                                    "pair_type": f"cross_{target_bin}",
                                }
                            )
                            ca_usage[c1] += 1
                            ca_usage[c2] += 1
                            generated_count += 1
                            progress_made = True
                            found_pair = True

                    if not progress_made:
                        current_cap *= 2

        return pd.DataFrame(pairs)

    def build_pairs_for_split(
        self, df_pool: pd.DataFrame, name: str, total_target: int = None
    ) -> pd.DataFrame:
        if "chunk_id" not in df_pool.columns:
            df_pool = df_pool.reset_index(drop=True)
            df_pool["chunk_id"] = np.arange(len(df_pool))

        # --- TARGET CALCULATION ---
        # 10 bins (0.0, 0.1, 0.2, 0.3, 0.4 | 0.6, 0.7, 0.8, 0.9, 1.0) - No 0.5
        n_subj = df_pool["subject"].nunique()
        target_per_subj_bin = (total_target // 10) // max(n_subj, 1)

        logger.info(
            f"{name} Target: {total_target} total | {target_per_subj_bin} per subject-bin"
        )

        shared_usage = Counter()

        extremes, used_ids, seen = self.build_extreme_pairs_for_split(
            df_pool, name, target_per_subj_bin
        )

        same_mids = self.build_midrange_pairs_for_split(
            df_pool,
            name,
            used_ids,
            seen,
            target_per_subj_bin,
            usage_counter=shared_usage,
        )

        cross_mids = self.build_cross_author_pairs(
            df_pool, name, seen, target_per_subj_bin, usage_counter=shared_usage
        )

        all_pairs = pd.concat([extremes, same_mids, cross_mids], ignore_index=True)
        all_pairs["score"] = all_pairs["score"].apply(lambda x: round(x, 1))

        logger.info(f"{name} Raw Candidates: {len(all_pairs)}")

        # --- STRICT BALANCING ---
        if total_target:
            target_per_bucket = total_target // (n_subj * 10)
            balanced_dfs = []
            for (subj, sc), group in all_pairs.groupby(["subject", "score"]):
                if len(group) >= target_per_bucket:
                    balanced_dfs.append(
                        group.sample(n=target_per_bucket, random_state=SEED)
                    )
                else:
                    balanced_dfs.append(
                        group
                    )  # Should ideally not happen now with dynamic caps

            all_pairs = pd.concat(balanced_dfs, ignore_index=True)
            all_pairs = all_pairs.sample(frac=1.0, random_state=SEED).reset_index(
                drop=True
            )
            logger.info(f"{name} Final Count: {len(all_pairs)} (Target {total_target})")

        return all_pairs

    def verify_integrity(self, train, val, test):
        logger.info("Running integrity checks...")
        splits = {"train": train, "val": val, "test": test}

        auths = {
            n: set(df["author1"]) | set(df["author2"]) if not df.empty else set()
            for n, df in splits.items()
        }
        if auths["train"] & auths["val"]:
            logger.error("Train/Val Author Leakage!")
        if auths["train"] & auths["test"]:
            logger.error("Train/Test Author Leakage!")

        for name, df in splits.items():
            if df.empty:
                continue
            dist = df["score"].value_counts().sort_index().to_dict()
            logger.info(f"{name} Score Dist: {dist}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


def main() -> None:
    input_file_path = "../data/Dataset_Replacements_Final.jsonl"
    cleaned_metadata_path = "../data/cleaned_metadata.json"
    output_disk_path = "../data/style_judge_datasetV2"

    raw_data = load_raw_data(input_file_path)
    if not raw_data:
        return

    auto_blacklist = detect_incomplete_chunks(raw_data)
    analyze_lazy_copies(raw_data)

    builder = RobustStyleDatasetBuilder(
        raw_data, cleaned_metadata_path, auto_blacklist=auto_blacklist
    )
    if builder.df.empty:
        return

    df_train_pool, df_val_pool, df_test_pool = builder.create_splits()

    # Targets: 130000 (Train) / 13000 (Val/Test) for 10 bins
    train_df = builder.build_pairs_for_split(
        df_train_pool, "Train", total_target=130000
    )
    val_df = builder.build_pairs_for_split(df_val_pool, "Val", total_target=13000)
    test_df = builder.build_pairs_for_split(df_test_pool, "Test", total_target=13000)

    builder.verify_integrity(train_df, val_df, test_df)

    output_columns = [
        "sentence1",
        "sentence2",
        "score",
        "subject",
        "author1",
        "author2",
        "book1",
        "book2",
        "id1",
        "id2",
    ]

    logger.info("Converting to HuggingFace Dataset...")
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[output_columns], preserve_index=False
            ),
            "validation": Dataset.from_pandas(
                val_df[output_columns], preserve_index=False
            ),
            "test": Dataset.from_pandas(test_df[output_columns], preserve_index=False),
        }
    )

    dataset_dict.save_to_disk(output_disk_path)
    logger.info("✅ Save Successful!")


if __name__ == "__main__":
    main()
