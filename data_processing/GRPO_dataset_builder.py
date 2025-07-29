import jsonlines
import logging
import random
from tqdm import tqdm
from datasets import Dataset, load_dataset
from prompts import (
    PromptManager,
)

# --- 1. SET YOUR CONFIGURATION HERE ---
HF_USERNAME = "VibrantVista"
BASE_REPO_NAME = "GRPO_twain-prompts"

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_prompt_dataset(style_examples: list) -> list[dict]:
    """
    Creates every possible pairing of prompts and style texts.
    """
    logger.info("Loading base prompts from PromptManager...")
    prompts = PromptManager.get_twain_prompt()
    dataset_list = []

    logger.info(
        f"Creating all pairs from {len(prompts)} prompts and {len(style_examples)} style examples..."
    )

    # Create every possible pairing using a nested loop
    for style_example in tqdm(style_examples, desc="Processing Style Samples"):
        for prompt_index, prompt_string in enumerate(prompts):
            dataset_list.append(
                {
                    "prompt": prompt_string,  # Use the raw string, NOT a chat template
                    "sample_text": style_example["text"],
                    "prompt_type": prompt_index,
                }
            )

    return dataset_list


def main():
    # --- 2. Load Style Samples ---
    logger.info("Loading style examples from the source dataset...")
    target_dataset = load_dataset("VibrantVista/source_dataset")["full"]
    filtered_style_examples = [
        ex
        for ex in target_dataset
        if ex["author"] == "Twain, Mark"
        and ex["title"] == "The Adventures of Huckleberry Finn"
    ]

    if not filtered_style_examples:
        logger.error("No style examples found. Exiting.")
        return

    # --- 3. Generate the Dataset ---
    prompt_list = create_prompt_dataset(filtered_style_examples)
    num_samples = len(prompt_list)
    logger.info(f"Generated a total of {num_samples} prompt-sample pairs.")

    # --- 4. Shuffle the Data ---
    logger.info("Shuffling the dataset...")
    random.shuffle(prompt_list)
    logger.info("Dataset shuffled.")

    # --- 5. Convert and Upload to Hugging Face Hub ---
    hf_dataset = Dataset.from_list(prompt_list)

    repo_id = f"{HF_USERNAME}/{BASE_REPO_NAME}-{num_samples}"

    logger.info(f"Uploading dataset to Hugging Face Hub at {repo_id}")
    hf_dataset.push_to_hub(
        repo_id,
    )
    logger.info("Dataset successfully uploaded.")


if __name__ == "__main__":
    main()
