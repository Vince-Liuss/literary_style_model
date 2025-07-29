import torch
import os
import jsonlines
import re
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from collections import defaultdict


def prepare_replacement_tasks(
    sample_text, tokenizer, mask_strategy="longest", mask_ratio=0.1
):
    """
    Identifies sentences to be replaced and prepares a structured chat prompt for each task
    using the tokenizer's chat template.
    """
    # Standardize spacing after punctuation for consistent sentence splitting
    text_standardized_spacing = re.sub(r"(?<=[.!?])\s+", " ", sample_text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text_standardized_spacing)
    if not sentences:
        return []

    actual_num_to_mask = max(1, int(len(sentences) * mask_ratio))
    indices_to_mask = []
    last_sentence_idx = len(sentences) - 1

    # Select indices of sentences to mask based on the chosen strategy
    if mask_strategy == "longest":
        sentence_lengths = [(i, len(s)) for i, s in enumerate(sentences)]
        sentence_lengths.sort(key=lambda x: x[1], reverse=True)
        candidate_indices = [
            idx for idx, _ in sentence_lengths if idx != last_sentence_idx
        ]
        indices_to_mask = sorted(candidate_indices[:actual_num_to_mask])
    elif mask_strategy == "random":
        candidate_indices = [i for i in range(len(sentences)) if i != last_sentence_idx]
        k = min(actual_num_to_mask, len(candidate_indices))
        indices_to_mask = sorted(random.sample(candidate_indices, k))
    else:
        raise ValueError("mask_strategy must be 'longest' or 'random'")

    replacement_tasks = []
    for i in indices_to_mask:
        original_sentence = sentences[i]
        text_before = " ".join(sentences[:i]).strip()
        text_after = " ".join(sentences[i + 1 :]).strip()

        # Create a structured message list for the chat template
        messages = [
            {
                "role": "system",
                "content": "You are an expert author. Your task is to complete a text by generating a single sentence that fits logically and stylistically in the provided context. Your response MUST contain ONLY the text for the single new sentence. Do not add labels, markdown, or any other explanatory text.",
            },
            {
                "role": "user",
                "content": f"""**Context Before:**\n{text_before}\n\n**[A SINGLE SENTENCE IS MISSING HERE]**\n\n**Context After:**\n{text_after}\n\nThe new sentence must NOT be similar to the following original sentence: "{original_sentence}" """,
            },
        ]

        # Apply the tokenizer's chat template to get the correctly formatted prompt string
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        replacement_tasks.append(
            {
                "sentence_index": i,
                "original_sentence": original_sentence,
                "prompt_text": prompt_text,
                "word_count": len(original_sentence.strip().split()),
                "original_sentences_list": sentences,
            }
        )

    return replacement_tasks


def process_dataset_in_batches(
    dataset, model, tokenizer, mask_strategies=["longest"], mask_ratios=[0.1, 0.2, 0.3]
):
    """
    Efficiently processes an entire dataset by preparing all prompts, generating text in one batch,
    and then reconstructing the documents.
    """
    print("Phase 1: Preparing all replacement tasks...")
    all_tasks = []
    reconstruction_map = {}
    max_word_count = 0

    for example in tqdm(dataset, desc="Preparing Prompts"):
        text = example["text"]
        if len(text.split()) < 50:
            continue

        for strategy in mask_strategies:
            for ratio in mask_ratios:
                doc_config_id = (
                    f"{example.get('chunk_id', id(example))}_{strategy}_{ratio}"
                )
                tasks_for_doc = prepare_replacement_tasks(
                    text, tokenizer, strategy, ratio
                )
                if not tasks_for_doc:
                    continue

                reconstruction_map[doc_config_id] = {
                    "original_text": text,
                    "author": example["author"],
                    "title": example.get("title", ""),
                    "chunk_id": example.get("chunk_id", ""),
                    "mask_ratio": ratio,
                    "mask_strategy": strategy,
                }

                for task in tasks_for_doc:
                    task["doc_config_id"] = doc_config_id
                    all_tasks.append(task)
                    if task["word_count"] > max_word_count:
                        max_word_count = task["word_count"]

    if not all_tasks:
        print("No tasks were generated. Exiting.")
        return []

    print(
        f"\nPhase 2: Generating {len(all_tasks)} sentence replacements in a single batch..."
    )
    prompts = [
        task["prompt_text"] for task in tqdm(all_tasks, desc="Extracting prompts")
    ]
    max_tokens = max_word_count + 30
    batch_size = 4000

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    with tqdm(total=len(all_tasks), desc="Processing in Batches") as pbar:
        for i in range(0, len(all_tasks), batch_size):
            # Get the current chunk of tasks
            current_batch_tasks = all_tasks[i : i + batch_size]

            # Extract prompts for this chunk only
            prompts = [task["prompt_text"] for task in current_batch_tasks]

            # Generate text for the current batch
            outputs = model.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )

            # Assign the generated text back to the tasks in the current batch
            for task, output in zip(current_batch_tasks, outputs):
                task["replacement"] = output.outputs[0].text.strip()

            # Update the overall progress bar
            pbar.update(len(current_batch_tasks))

    for task, output in zip(all_tasks, outputs):
        task["replacement"] = output.outputs[0].text.strip()

    print("\nPhase 3: Reconstructing all documents...")
    completed_tasks_by_doc = defaultdict(list)
    for task in all_tasks:
        completed_tasks_by_doc[task["doc_config_id"]].append(task)

    final_results = []
    for doc_config_id, completed_tasks in tqdm(
        completed_tasks_by_doc.items(), desc="Reconstructing Texts"
    ):
        sentences = completed_tasks[0]["original_sentences_list"]
        new_sentences = list(sentences)
        for task in completed_tasks:
            new_sentences[task["sentence_index"]] = task["replacement"]

        filled_text = " ".join(new_sentences)
        original_data = reconstruction_map[doc_config_id]
        replacements_log = [
            {
                "original": t["original_sentence"],
                "replacement": t["replacement"],
                "sentence_index": t["sentence_index"],
            }
            for t in completed_tasks
        ]

        final_results.append(
            {
                "original_text": original_data["original_text"],
                "filled_text": filled_text,
                "author": original_data["author"],
                "title": original_data["title"],
                "chunk_id": original_data["chunk_id"],
                "mask_ratio": original_data["mask_ratio"],
                "mask_strategy": original_data["mask_strategy"],
                "replacements": replacements_log,
                "replacement_count": len(replacements_log),
            }
        )

    return final_results


def loading_dataset(dataset_name="VibrantVista/source_dataset"):
    print(f"Loading dataset from {dataset_name}")
    dataset = load_dataset(dataset_name, split="full")
    return dataset


def filter_dataset(dataset, count=1, selection_mode="top"):
    author_counts = defaultdict(int)
    for example in dataset:
        author_counts[example["author"]] += 1
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_authors:
        print("Warning: Dataset is empty or contains no authors.")
        return []

    selected_authors_names = []
    if selection_mode == "all":
        print(f"Selection mode is 'all'. Using all {len(author_counts)} authors found.")
        selected_authors_names = list(author_counts.keys())
    elif selection_mode == "top":
        selected_authors_with_counts = sorted_authors[:count]
        selected_authors_names = [author for author, _ in selected_authors_with_counts]
        print(f"Selecting top {count} authors by frequency:")
        for rank, (author, num_chunks) in enumerate(selected_authors_with_counts, 1):
            print(f"{rank}. {author}: {num_chunks} chunks")
    elif selection_mode == "bottom":
        selected_authors_with_counts = sorted_authors[-count:]
        selected_authors_names = [author for author, _ in selected_authors_with_counts]
        print(f"Selecting bottom {count} authors by frequency:")
        for rank, (author, num_chunks) in enumerate(
            sorted(selected_authors_with_counts, key=lambda x: x[1]), 1
        ):
            print(f"{rank}. {author}: {num_chunks} chunks")
    else:
        raise ValueError("Invalid selection_mode. Must be 'top', 'bottom', or 'all'.")

    filtered_examples = [ex for ex in dataset if ex["author"] in selected_authors_names]
    print(
        f"\nReturned a total of {len(filtered_examples)} chunks from {len(selected_authors_names)} authors."
    )
    return filtered_examples


def main(dataset_name="VibrantVista/source_dataset"):
    # Environment setup
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    # Model Initialization
    model_name = "google/gemma-3-27b-it"  # Using an instruction-tuned model is better for this prompt
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        task="generate",
        max_num_batched_tokens=4096,
        # distributed_executor_backend="ray",
        max_num_seqs=63 if torch.cuda.device_count() > 1 else 23,
        enable_chunked_prefill=True,
    )

    tokenizer = llm.get_tokenizer()
    # Data Loading and Filtering
    dataset = loading_dataset(dataset_name)
    filtered_dataset = filter_dataset(dataset, selection_mode="all")

    # Main Processing
    processed_dataset = process_dataset_in_batches(
        filtered_dataset,
        llm,
        tokenizer,
        mask_strategies=["longest"],
        mask_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )

    # Save Results
    if processed_dataset:
        output_filename = "Full_Replacements_dataset.jsonl"
        with jsonlines.open(output_filename, mode="w") as writer:
            writer.write_all(processed_dataset)
        print(f"\nDataset with replacements saved to {output_filename}")
        print(f"Processed and saved {len(processed_dataset)} final examples.")
    else:
        print("\nNo data was processed, output file not written.")


if __name__ == "__main__":
    main()
