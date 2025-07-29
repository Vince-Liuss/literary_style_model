import re
import random
import jsonlines
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import logging
import pandas as pd

# --- 1. Configure Logging ---
# Set up basic logging for better feedback during script execution.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 2. Data Loading and Cleaning ---
def get_dataset(file_path):
    """
    Load and clean dataset from the source JSONL file.
    Performs initial text cleaning and robustly handles mask_ratio conversion.
    Aggregates raw data by author.
    """
    logger.info(f"Loading raw dataset from: {file_path}")
    dataset_from_file = []
    try:
        with jsonlines.open(file_path) as reader:
            for line in tqdm(reader, desc="Loading data"):
                filled_text = line.get("filled_text", "")

                # Regex to remove specific markdown/control characters and MASK tokens.
                # Handles: **, \**, \\, [ and [MASK], [MASK_SENTENCE], [MASK_PARAGRAPH]
                filled_text = re.sub(r"\*\*|\\\*\*|\\{2}|\[", "", filled_text)
                filled_text = re.sub(
                    r"\[MASK(?:_SENTENCE|_PARAGRAPH)?\]", "", filled_text
                )
                line["filled_text"] = (
                    filled_text.strip()
                )  # Remove leading/trailing whitespace

                # Robustly convert mask_ratio to float, defaulting to 0.0 if invalid.
                try:
                    line["mask_ratio"] = float(line.get("mask_ratio", 0.0))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid mask_ratio found for chunk_id: {line.get('chunk_id', 'N/A')}. Setting to 0.0."
                    )
                    line["mask_ratio"] = 0.0

                dataset_from_file.append(line)
    except FileNotFoundError:
        logger.error(
            f"Error: Source file not found at {file_path}. Please check the path."
        )
        return defaultdict(list)
    except Exception as e:
        logger.error(f"An error occurred while loading or parsing the dataset: {e}")
        return defaultdict(list)

    # Group all loaded data by author for easier processing later.
    all_authors_data = defaultdict(list)
    for item in dataset_from_file:
        author = item.get("author", "unknown_author")
        all_authors_data[author].append(item)
    logger.info(f"Successfully loaded data for {len(all_authors_data)} authors.")
    return all_authors_data


# --- 3. Data Processing and Organization ---
def process_source_data(all_authors_data_raw):
    """
    Processes raw author data into two main structures:
    1. A flat list of all 'filled_text' chunks with metadata.
    2. A dictionary of 'original_text' chunks, grouped by author, ensuring uniqueness
       based on (author, title, chunk_id) to avoid duplicate original texts.
    """
    all_filled_chunks = []
    originals_by_author = defaultdict(list)

    logger.info("Processing and organizing raw source data into structured chunks...")
    for author, source_data in all_authors_data_raw.items():
        seen_original_keys_for_author = (
            set()
        )  # To track unique original chunks per author
        for example in tqdm(source_data, desc=f"Processing data for {author}"):
            # Append the processed 'filled' version to a flat list.
            # Include 'chunk_id' for later precise filtering.
            all_filled_chunks.append(
                {
                    "text": example["filled_text"],
                    "mask_ratio": example["mask_ratio"],
                    "author": author,
                    "title": example.get("title", "no_title"),
                    "chunk_id": example.get("chunk_id", "no_chunk_id"),
                }
            )

            # Add the 'original' version to a dictionary, grouped by author.
            # Ensure each original chunk is added only once based on its unique identifier.
            original_key = f"{author}_{example.get('title', 'no_title')}_{example.get('chunk_id', 'no_chunk_id')}"
            if original_key not in seen_original_keys_for_author:
                seen_original_keys_for_author.add(original_key)
                originals_by_author[author].append(
                    {
                        "text": example["original_text"],
                        "author": author,
                        "title": example.get("title", "no_title"),
                        "chunk_id": example.get("chunk_id", "no_chunk_id"),
                    }
                )
    logger.info(
        f"Finished processing: {len(all_filled_chunks)} filled chunks and {sum(len(v) for v in originals_by_author.values())} unique original chunks."
    )
    return all_filled_chunks, originals_by_author


# --- 4. Pair Generation Functions ---
def create_scored_pairs_for_split(
    filled_chunks_subset,
    originals_by_author_subset,
    include_refilled_pairs=True,
    include_original_original_pairs=True,
):
    """
    Builds a dataset of scored pairs for a given subset of filled and original chunks.
    Generates positive (masked), negative (cross-author), and optionally refilled-refilled pairs.
    Perfect positive (original-original) pairs are generated only if include_original_original_pairs is True.
    All positive pairs ensure sentences come from different titles by the same author.
    """
    logger.info("Generating scored pairs for the current dataset split...")
    scored_pairs = []
    authors_list_subset = list(originals_by_author_subset.keys())

    # Group filled chunks by author and title for efficient sampling for Type 4 pairs.
    filled_chunks_by_author_title = defaultdict(lambda: defaultdict(list))
    for f_chunk in filled_chunks_subset:
        filled_chunks_by_author_title[f_chunk["author"]][f_chunk["title"]].append(
            f_chunk
        )

    # --- Type 1: Positive Pairs (Same Author, Different Title, with Mask Ratio Score) ---
    # Pair a filled chunk with an original chunk from the same author but a different title.
    for filled_chunk in tqdm(
        filled_chunks_subset, desc="Generating Pos (Same Author, Diff Title)"
    ):
        current_author = filled_chunk["author"]
        current_title = filled_chunk["title"]

        # Find original chunks by the same author from a different title.
        same_author_different_title_candidates = [
            orig
            for orig in originals_by_author_subset.get(current_author, [])
            if orig["title"] != current_title
        ]

        if same_author_different_title_candidates:
            original_chunk_candidate = random.choice(
                same_author_different_title_candidates
            )
            # Score reflects content overlap: 1.0 means no mask, 0.0 means full mask.
            score = 1.0 - filled_chunk["mask_ratio"]
            scored_pairs.append(
                {
                    "sentence1": original_chunk_candidate["text"],
                    "sentence2": filled_chunk["text"],
                    "score": score,
                    "type": "positive_masked_different_title",
                    "authors_match": True,
                    "titles_match": False,
                }
            )

    # --- Type 2: Negative Pairs (Cross Author) ---
    # Pair a filled chunk with an original chunk from a completely different author.
    for filled_chunk in tqdm(
        filled_chunks_subset, desc="Generating Neg (Cross Author)"
    ):
        current_author = filled_chunk["author"]
        other_authors = [
            a for a in authors_list_subset if a != current_author
        ]  # Authors not matching current_author

        if other_authors:
            random_other_author = random.choice(other_authors)

            # Ensure the randomly chosen other author actually has original chunks available.
            if originals_by_author_subset.get(random_other_author):
                original_chunk_from_other_author = random.choice(
                    originals_by_author_subset[random_other_author]
                )
                scored_pairs.append(
                    {
                        "sentence1": original_chunk_from_other_author["text"],
                        "sentence2": filled_chunk["text"],
                        "score": 0.0,  # Score 0.0 for cross-author pairs.
                        "type": "negative_cross_author",
                        "authors_match": False,
                        "titles_match": False,
                    }
                )

    # --- Type 3: Perfect Positive Pairs (Score=1.0, Same Author, Different Title) ---
    # Pair two different original chunks from the same author but from different titles.
    if include_original_original_pairs:
        logger.info(
            "Generating Perfect Positive Pairs (score=1.0, Same Author, Different Title)..."
        )
        for author, author_originals in tqdm(
            originals_by_author_subset.items(), desc="Building Original-Original Pairs"
        ):
            originals_by_title = defaultdict(list)
            for chunk in author_originals:
                originals_by_title[chunk["title"]].append(chunk)

            # An author must have original chunks from at least two different titles to form these pairs.
            if len(originals_by_title) < 2:
                continue

            titles_of_author = list(originals_by_title.keys())
            for title1 in titles_of_author:
                for chunk1 in originals_by_title[title1]:
                    # Select a different title from the same author.
                    other_titles = [t for t in titles_of_author if t != title1]
                    if (
                        not other_titles
                    ):  # Safeguard, should not happen if len(originals_by_title) >= 2
                        continue

                    title2 = random.choice(other_titles)
                    chunk2_candidates = originals_by_title[title2]

                    if chunk2_candidates:
                        chunk2 = random.choice(chunk2_candidates)
                        scored_pairs.append(
                            {
                                "sentence1": chunk1["text"],
                                "sentence2": chunk2["text"],
                                "score": 1.0,  # Score 1.0 for perfect matches.
                                "type": "perfect_positive_different_title",
                                "authors_match": True,
                                "titles_match": False,
                            }
                        )

    # --- Type 4: Refilled-Refilled Pairs (Same Author, Different Title, Scored by Mask Ratio Difference) ---
    # Pair two refilled chunks from the same author but from different titles.
    # Score reflects the absolute difference in their mask ratios (0.0 to 1.0).
    if include_refilled_pairs:
        logger.info(
            "Generating Refilled-Refilled Pairs (Same Author, Different Title, Mask Ratio Score)..."
        )
        for author, titles_data in tqdm(
            filled_chunks_by_author_title.items(),
            desc="Building Refilled-Refilled Pairs",
        ):
            titles_of_author = list(titles_data.keys())

            # Need at least two different titles from this author to form a pair.
            if len(titles_of_author) < 2:
                continue

            for title1, filled_chunks1 in titles_data.items():
                other_titles = [t for t in titles_of_author if t != title1]
                if not other_titles:  # Safeguard
                    continue

                for chunk1 in filled_chunks1:
                    title2 = random.choice(other_titles)
                    filled_chunks2_candidates = titles_data[title2]

                    if filled_chunks2_candidates:
                        chunk2 = random.choice(filled_chunks2_candidates)

                        # Scoring: 1.0 - absolute difference in mask ratios.
                        # If mask ratios are very similar, score is high (close to 1.0).
                        # If mask ratios are very different, score is low (close to 0.0).
                        score = 1.0 - abs(chunk1["mask_ratio"] - chunk2["mask_ratio"])

                        scored_pairs.append(
                            {
                                "sentence1": chunk1["text"],
                                "sentence2": chunk2["text"],
                                "score": score,
                                "type": "refilled_refilled_different_title",
                                "authors_match": True,
                                "titles_match": False,
                            }
                        )

    logger.info(f"Total scored pairs generated for this split: {len(scored_pairs)}")
    return scored_pairs


def create_test_original_pairs(
    originals_pool, num_positive_partners_per_chunk=3, num_negative_partners_per_chunk=3
):
    """
    Generates test pairs exclusively from a dedicated pool of original chunks.
    Includes both perfect positive (same author, different title, score=1.0)
    and negative (cross-author, score=0.0) pairs.
    This function ensures no data leakage from training/validation into the test set.

    Args:
        originals_pool (list): A list of original chunk dictionaries reserved for testing.
        num_positive_partners_per_chunk (int): How many positive (same-author, diff-title) partners
                                                to try and find for each chunk.
        num_negative_partners_per_chunk (int): How many negative (cross-author) partners
                                                to try and find for each chunk.
    """
    logger.info(
        "Generating Test Original-Original Pairs (Positive & Negative) from dedicated pool..."
    )
    test_pairs = []

    # Group the test pool originals by author and then by title for pairing.
    originals_by_title_by_author = defaultdict(lambda: defaultdict(list))
    for chunk in originals_pool:
        originals_by_title_by_author[chunk["author"]][chunk["title"]].append(chunk)

    authors_in_test_pool = list(originals_by_title_by_author.keys())

    # --- Test Type A: Perfect Positive Pairs (Same Author, Different Title, Score=1.0) ---
    logger.info("Building Positive Test Pairs (Same Author, Different Title)...")
    for author in tqdm(authors_in_test_pool, desc="Building Positive Test Pairs"):
        titles_data = originals_by_title_by_author[author]
        titles_of_author = list(titles_data.keys())

        # An author must have at least two different titles to form positive pairs.
        if len(titles_of_author) < 2:
            continue

        for title1 in titles_of_author:
            for chunk1 in titles_data[title1]:
                other_titles = [t for t in titles_of_author if t != title1]
                if not other_titles:
                    continue

                # Sample multiple partners from different titles
                num_partners_to_sample = min(
                    num_positive_partners_per_chunk, len(other_titles)
                )
                sampled_titles = random.sample(other_titles, num_partners_to_sample)

                for title2 in sampled_titles:
                    chunk2_candidates = originals_by_title_by_author[author][title2]
                    if chunk2_candidates:
                        chunk2 = random.choice(chunk2_candidates)
                        test_pairs.append(
                            {
                                "sentence1": chunk1["text"],
                                "sentence2": chunk2["text"],
                                "score": 1.0,
                                "type": "perfect_positive_different_title_test",
                                "authors_match": True,
                                "titles_match": False,
                            }
                        )

    # --- Test Type B: Negative Pairs (Cross Author, Score=0.0) ---
    # This requires at least two different authors in the test pool.
    if len(authors_in_test_pool) >= 2:
        logger.info(
            "Building Negative Test Pairs (Cross Author, Original-Original, Score=0.0)..."
        )
        # To avoid creating an overwhelming number of negative pairs, we'll iterate through
        # a subset of chunks and pair them with a subset of other authors' chunks.

        # Collect all individual chunks from the test pool for easier random sampling.
        all_test_pool_chunks = []
        for author_data in originals_by_title_by_author.values():
            for title_chunks in author_data.values():
                all_test_pool_chunks.extend(title_chunks)

        # Iterate through each chunk in the test pool to find negative partners.
        for chunk1 in tqdm(all_test_pool_chunks, desc="Generating Negative Test Pairs"):
            current_author = chunk1["author"]

            # Find other authors in the test pool that are different from the current chunk's author.
            other_authors_for_neg_pair = [
                a for a in authors_in_test_pool if a != current_author
            ]
            if not other_authors_for_neg_pair:
                continue  # Cannot form a cross-author pair if only one author.

            # Sample a few distinct other authors to create negative pairs with.
            num_authors_to_sample = min(
                num_negative_partners_per_chunk, len(other_authors_for_neg_pair)
            )
            sampled_other_authors = random.sample(
                other_authors_for_neg_pair, num_authors_to_sample
            )

            for random_other_author in sampled_other_authors:
                chunk2_candidates_from_other_author_by_title = (
                    originals_by_title_by_author[random_other_author]
                )
                if not chunk2_candidates_from_other_author_by_title:
                    continue  # The other author might not have any chunks remaining after filtering.

                # Flatten the list of chunks for the sampled other author to pick a random one.
                all_chunks_from_other_author = []
                for (
                    title_chunks
                ) in chunk2_candidates_from_other_author_by_title.values():
                    all_chunks_from_other_author.extend(title_chunks)

                if all_chunks_from_other_author:  # Ensure there are chunks to pick from
                    chunk2 = random.choice(all_chunks_from_other_author)
                    test_pairs.append(
                        {
                            "sentence1": chunk1["text"],
                            "sentence2": chunk2["text"],
                            "score": 0.0,
                            "type": "negative_cross_author_test",
                            "authors_match": False,
                            "titles_match": False,
                        }
                    )
    else:
        logger.warning(
            "Not enough authors in the test pool to generate cross-author negative test pairs."
        )

    logger.info(f"Total test pairs created: {len(test_pairs)}")
    return test_pairs


# --- 5. Hugging Face Upload Function ---
def upload_to_huggingface(train_data, val_data, test_data, repo_name):
    """
    Converts data splits to a Hugging Face DatasetDict and uploads to the Hugging Face Hub.
    Requires 'huggingface-cli login' to be performed beforehand.
    """
    logger.info(
        f"🚀 Preparing to upload dataset to '{repo_name}' on Hugging Face Hub..."
    )

    if not train_data and not val_data and not test_data:
        logger.warning(
            "No data provided for upload across all splits. Skipping upload."
        )
        return

    # Convert lists of dictionaries to Hugging Face Dataset objects.
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    # Create a DatasetDict from the individual datasets.
    scored_pair_dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    logger.info(f"Pushing dataset '{repo_name}' to the Hub...")
    try:
        scored_pair_dataset_dict.push_to_hub(repo_name)
        logger.info("✅ Upload complete!")
    except Exception as e:
        logger.error(f"Failed to push dataset to Hugging Face Hub: {e}")
        logger.error(
            "Please ensure you are logged in via 'huggingface-cli login' and have sufficient write permissions to the repository."
        )


# --- 6. Main Execution Flow ---
def main():
    """
    Main function to orchestrate the dataset creation and upload process.
    Implements a hybrid splitting strategy:
    - A dedicated pool of original chunks for the test set (no leakage).
    - Remaining data is used for train/validation.
    - Generates various pair types for train/val, and only original-original for test.
    """
    # --- Configuration ---
    file_path = "data/raw_data.jsonl"  # Path to the source JSONL file containing raw data.
    # IMPORTANT: Change this to your desired Hugging Face repository name.
    # Make sure it's unique and you have write access.
    hf_repo_name = "your-username/your-repo-name"  

    logger.info(
        "--- Starting Scored Pair Dataset Creation (Hybrid Split for Style Benchmarking) ---"
    )
    logger.info(f"The final dataset will be uploaded to: {hf_repo_name}")
    logger.info(
        "Please ensure you are logged in to Hugging Face via 'huggingface-cli login' in your terminal."
    )

    # 1. Load and process all raw data.
    all_authors_data_raw = get_dataset(file_path)
    if not all_authors_data_raw:
        logger.error("No raw data loaded. Exiting script.")
        return

    # Process all data once to get flat lists of filled and original chunks.
    all_filled_chunks_full, originals_by_author_full = process_source_data(
        all_authors_data_raw
    )

    # --- Step 1: Dedicate a pool of original chunks for the TEST SET ONLY ---
    # These chunks (and their corresponding filled versions) will be completely excluded
    # from the training and validation sets to prevent any data leakage.

    # Collect all unique original chunks for potential test set candidates.
    all_unique_original_chunks = []
    for author_chunks in originals_by_author_full.values():
        all_unique_original_chunks.extend(author_chunks)

    random.seed(42)  # Set random seed for reproducibility of data splits.
    random.shuffle(all_unique_original_chunks)

    # Define the ratio of unique original chunks to reserve for the test set.
    test_pool_ratio = 0.20  # Increased ratio to try and get more diverse test set.

    # Prioritize selection from multi-title authors
    test_originals_pool = []

    # Identify authors who have multiple titles across the entire dataset.
    multi_title_authors_overall = []
    for author, originals in originals_by_author_full.items():
        titles_for_author = {chunk["title"] for chunk in originals}
        if len(titles_for_author) >= 2:
            multi_title_authors_overall.append(author)

    logger.info(
        f"Found {len(multi_title_authors_overall)} authors with multiple titles in the full dataset."
    )

    # Separate chunks into those from multi-title authors and others
    multi_title_original_chunks = []
    other_original_chunks = []
    for chunk in all_unique_original_chunks:  # Iterate through the shuffled list
        if chunk["author"] in multi_title_authors_overall:
            multi_title_original_chunks.append(chunk)
        else:
            other_original_chunks.append(chunk)

    # Attempt to fill the test pool primarily with chunks from multi-title authors
    target_test_pool_size = int(len(all_unique_original_chunks) * test_pool_ratio)

    # Take as many as possible from multi_title_original_chunks first
    test_originals_pool.extend(multi_title_original_chunks[:target_test_pool_size])

    # If target not met, fill with others
    if len(test_originals_pool) < target_test_pool_size:
        remaining_needed = target_test_pool_size - len(test_originals_pool)
        test_originals_pool.extend(other_original_chunks[:remaining_needed])

    # Ensure a minimum number of chunks in the test pool to allow for pair creation (at least 2).
    if len(test_originals_pool) < 2:
        logger.warning(
            f"Calculated test pool size ({len(test_originals_pool)}) is less than 2. Attempting to take at least 2 chunks if available."
        )
        test_originals_pool = all_unique_original_chunks[
            : min(len(all_unique_original_chunks), 2)
        ]
        if len(test_originals_pool) < 2:
            logger.error(
                "Insufficient unique original chunks in the dataset to create a test set. Please provide more more data or reduce test_pool_ratio. Exiting."
            )
            return

    # Create a set of unique identifiers for all chunks (original and filled) that are part of the test pool.
    # This set will be used to filter out these chunks from the train/val data.
    test_chunk_identifiers = set()
    for chunk in test_originals_pool:
        test_chunk_identifiers.add(
            f"{chunk['author']}_{chunk['title']}_{chunk['chunk_id']}"
        )

    logger.info(
        f"Successfully reserved {len(test_originals_pool)} unique original chunks for the test set (prioritized multi-title authors)."
    )

    # --- Step 2: Prepare data for Training and Validation (excluding test pool content) ---
    # Filter the full set of filled and original chunks to exclude any that are part of the test pool.

    train_val_filled_chunks = []
    train_val_originals_by_author = defaultdict(
        list
    )  # This will be the source for train/val original chunks

    # Populate `train_val_filled_chunks` with chunks whose original counterparts are NOT in the test pool.
    for f_chunk in all_filled_chunks_full:
        chunk_identifier = (
            f"{f_chunk['author']}_{f_chunk['title']}_{f_chunk['chunk_id']}"
        )
        if chunk_identifier not in test_chunk_identifiers:
            train_val_filled_chunks.append(f_chunk)

    # Populate `train_val_originals_by_author` with original chunks NOT in the test pool.
    for author, originals in originals_by_author_full.items():
        for original_chunk in originals:
            chunk_identifier = f"{original_chunk['author']}_{original_chunk['title']}_{original_chunk['chunk_id']}"
            if chunk_identifier not in test_chunk_identifiers:
                train_val_originals_by_author[author].append(original_chunk)

    logger.info(
        f"Filtered: {len(train_val_filled_chunks)} filled chunks and {sum(len(v) for v in train_val_originals_by_author.values())} original chunks available for train/val."
    )

    # --- Step 3: Split Training/Validation Data (no author split here, as per user's request) ---
    # We will now split the *pairs* generated from the `train_val_filled_chunks` and `train_val_originals_by_author`
    # into train and validation sets.

    # Generate all pairs for the train/val pool.
    # IMPORTANT: Set `include_original_original_pairs=False` here, as these are reserved for the test set.
    all_train_val_pairs = create_scored_pairs_for_split(
        train_val_filled_chunks,
        train_val_originals_by_author,
        include_refilled_pairs=True,
        include_original_original_pairs=False,  # Exclude original-original pairs from train/val
    )

    if not all_train_val_pairs:
        logger.error(
            "No training/validation pairs could be generated after filtering. Exiting."
        )
        return

    random.shuffle(all_train_val_pairs)  # Shuffle all train/val pairs for random split.

    # Define train/val split ratio for the generated pairs.
    train_val_pairs_split_ratio = 0.9  # 90% for train, 10% for val.
    train_split_point = int(len(all_train_val_pairs) * train_val_pairs_split_ratio)

    train_data = all_train_val_pairs[:train_split_point]
    val_data = all_train_val_pairs[train_split_point:]

    logger.info(f"Total generated Train/Val pairs: {len(all_train_val_pairs)}")
    logger.info(f"Pairs assigned to Training set: {len(train_data)}")
    logger.info(f"Pairs assigned to Validation set: {len(val_data)}")

    # --- Step 4: Generate Scored Pairs for the Test Split ---
    # Test data: ONLY Perfect Positive (Original-Original) pairs from the dedicated test_originals_pool.
    test_data = create_test_original_pairs(test_originals_pool)

    # Final check for empty datasets before upload.
    if not train_data or not val_data or not test_data:
        logger.error(
            "One or more final dataset splits (train/val/test) are empty. Cannot proceed with upload. Please review data and splitting logic."
        )
        if not train_data:
            logger.error("Final Training set is empty.")
        if not val_data:
            logger.error("Final Validation set is empty.")
        if not test_data:
            logger.error("Final Test set is empty.")
        return

    logger.info("\n--- Final Dataset Creation Summary ---")
    logger.info(f"Total Training pairs:   {len(train_data)}")
    logger.info(f"Total Validation pairs: {len(val_data)}")
    logger.info(f"Total Test pairs:       {len(test_data)}")

    # --- Step 5. Upload the dataset to Hugging Face ---
    upload_to_huggingface(train_data, val_data, test_data, hf_repo_name)


if __name__ == "__main__":
    main()
