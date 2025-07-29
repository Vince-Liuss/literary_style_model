import torch
import os
import jsonlines
import re
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from prompts import PromptManager
from datasets import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer from Hugging Face Hub.
    """
    logger.info(f"Loading model and tokenizer from {model_name}")
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        task="generate",
        max_model_len=8192,
        max_num_batched_tokens=32768,
        max_num_seqs=200,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
    )

    return llm


def prompt_template():
    """
    Create a prompt template for the model.
    """

    sample_numbers = 40000
    prompts_for_vllm = []
    datasets = []

    for i in tqdm(range(sample_numbers), desc="Generating prompts"):

        dataset_prompt, template_idx = PromptManager.get_sft_prompt()

        # prompt = PromptManager.get_story_prompt(dataset_prompt)

        prompt = PromptManager.get_story_prompt_regular(dataset_prompt)

        prompts_for_vllm.append(prompt)
        sample = {
            "prompt_type": template_idx,
            "prompt": dataset_prompt,
            "response": "",
            "content_reward_score": 0.0,
        }
        datasets.append(sample)

    return prompts_for_vllm, datasets


def build_dataset(prompts_for_vllm, datasets, llm):
    """
    Build the dataset from the prompts.
    """
    logger.info("Building dataset")

    sample_params = SamplingParams(
        max_tokens=1600,
        min_tokens=1400,
        top_k=64,
        top_p=0.95,
        temperature=1.0,
        repetition_penalty=1.0,
        min_p=0.0,
        stop=["\n\nTHE END.", "THE END.", "\nTHE END."],
        n=1,
        skip_special_tokens=True,
    )
    vllm_prompts_input = []

    # Assuming prompts is a list of dictionaries in chat format
    tokenizer = llm.get_tokenizer()
    for original_prompt in tqdm(prompts_for_vllm, desc="Formatting prompts"):
        # Apply the chat template to get the final string format
        formatted_prompt_str = tokenizer.apply_chat_template(
            original_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        vllm_prompts_input.append(formatted_prompt_str)

    outputs = llm.generate(vllm_prompts_input, sample_params)

    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        # Since n=1, output.outputs will have one element
        generated_text = output.outputs[0].text  # Get the generated text

        if generated_text == "":
            logger.warning(f"Empty response for prompt {i}.")
            exit(1)

        end_marker_patterns = ["\n\nTHE END.", "THE END.", "\nTHE END."]
        for pattern in end_marker_patterns:
            if pattern in generated_text:
                generated_text = generated_text.split(pattern)[0] + pattern
                break

        datasets[i]["response"] = generated_text

    return datasets


def create_reward_score(dataset, llm):
    """
    Create a reward score for the dataset.
    """
    logger.info("Creating reward score")

    score_sampling_params = SamplingParams(
        max_tokens=5,  # Single token for the score
        temperature=0.0,  # Fully deterministic
        top_k=1,  # Only select the highest probability token
        top_p=1.0,  # Consider all tokens but with deterministic selection
    )

    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    # build up the input for the reward score
    for i in tqdm(range(len(dataset)), desc="formatting reward score prompts"):
        resp = dataset[i]["response"]
        # Create the prompt for the reward score
        messages = PromptManager.get_content_reward_messages_regular(resp)
        messages = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        formatted_prompts.append(messages)

    outputs = llm.generate(formatted_prompts, score_sampling_params)

    # Process the outputs
    for i, output in enumerate(tqdm(outputs, desc="Processing scores")):
        generated_text = output.outputs[0].text.strip()
        # Extract score using regex
        score_match = re.search(r"\b[1-5]\b", generated_text)
        if not score_match:
            score = 3
        else:
            score = float(score_match.group(0))
        # normalize the score
        score = (score - 1) / 4

        dataset[i]["content_reward_score"] = score

    return dataset


def upload_to_huggingface(dataset, repo_name):
    """
    Upload dataset to Hugging Face Hub in SFT format.
    """
    logger.info("Converting to SFT format and uploading to Hugging Face")

    # Convert to SFT format
    sft_data = []
    for sample in dataset:
        sft_sample = {
            "prompt_type": sample["prompt_type"],
            "prompt": sample["prompt"],
            "completion": sample["response"],
            "quality_score": sample["content_reward_score"],
        }
        sft_data.append(sft_sample)

    # Create HF dataset
    hf_dataset = Dataset.from_list(sft_data)

    hf_dataset = hf_dataset.shuffle(seed=42)

    # Upload to HF
    hf_dataset.push_to_hub(repo_name, private=True)
    logger.info(f"Dataset uploaded to: {repo_name}")


def main():
    # Load the model and tokenizer
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    llm = load_model_and_tokenizer(model_name)
    # Load the prompts
    prompts_for_vllm, dataset = prompt_template()
    # Build the dataset
    dataset = build_dataset(prompts_for_vllm, dataset, llm)

    # Create the reward score
    dataset = create_reward_score(dataset, llm)

    del llm
    torch.cuda.empty_cache()

    # Extract values into lists for statistics
    content_scores = [item["content_reward_score"] for item in dataset]

    # Calculate statistics
    logger.info("The highest content reward score is: ")
    logger.info(max(content_scores))
    logger.info("The lowest content reward score is: ")
    logger.info(min(content_scores))
    logger.info("The average content reward score is: ")
    logger.info(sum(content_scores) / len(content_scores))

    # Save the dataset to a file
    output_file = "SFT_dataset.jsonl"
    with jsonlines.open(output_file, mode="w") as writer:
        for sample in dataset:
            writer.write(sample)

    logger.info(f"Dataset saved to {output_file}")

    upload_to_huggingface(dataset, "VibrantVista/SFT_dataset")


if __name__ == "__main__":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    main()
