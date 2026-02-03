# Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning

This project implements a multi-stage pipeline for training language models to write in the style of classic authors. The approach uses sentence transformers as reward models and Group Relative Policy Optimization (GRPO) reinforcement learning techniques to achieve high-quality literary style transfer in long-form story generation.

## 🎯 Project Overview

This research focuses on capturing the distinctive voices of four classic authors:

1.  **Mark Twain** (*Adventures of Huckleberry Finn*)
2.  **Charles Dickens** (*A Tale of Two Cities*)
3.  **Jane Austen** (*Pride and Prejudice*)
4.  **Thomas Hardy** (*Tess of the d'Urbervilles: A Pure Woman*)

The pipeline consists of three main stages:
1.  **Reward Model Training**: Fine-tune sentence transformers to evaluate literary style similarity.
2.  **Supervised Fine-Tuning (SFT)**: Initial training of language models on high-quality literary examples.
3.  **GRPO Training**: Reinforcement learning optimization serving a "Judge" model via vLLM to guide the generation process.

## 🔗 Model Collection

All trained models and datasets are available in the [VibrantVista Literary Style GRPO Collection](https://huggingface.co/collections/VibrantVista/literary-style-grpo-models).

## 📚 Datasets

This project uses the following datasets available on HuggingFace:

### Training Datasets
- **SFT Dataset**: [`VibrantVista/story-style-SFT-dataset`](https://huggingface.co/datasets/VibrantVista/story-style-SFT-dataset)
  - Used for the Supervised Fine-Tuning stage.
  
- **Style Judge Dataset**: [`VibrantVista/style-judge-dataset`](https://huggingface.co/datasets/VibrantVista/style-judge-dataset)
  - Pairs of texts used to train the reward model (Sentence Transformer) to distinguish authorial styles.

- **GRPO Dataset**: [`VibrantVista/grpo-style-training`](https://huggingface.co/datasets/VibrantVista/grpo-style-training)
  - Prompts and style references for the Reinforcement Learning stage.

## 📁 Project Structure
literary_style_model/
├── config/ # Configuration files
│ ├── ds_config_fsdp.json # DeepSpeed Zero3 config for multi-GPU
│ └── fsdp_config.yaml # Accelerate FSDP config for multi-GPU
├── docker.def
├── LICENSE
├── pyproject.toml
├── README.md
├── data/ # Sample data files
├── data_analysis/ # Analysis and evaluation tools
│ ├── chunk_book.py
│ └── Unified_analysis.py
├── data_processing/ # Dataset creation and processing
│ ├── export_dataset_to_json.py
│ ├── GRPO_dataset_builder.py
│ ├── gutenberg_download.py
│ ├── prompts.py
│ ├── SFT_dataset_builder.py
│ ├── upload_triplet_dataset.py
│ └── Style_evaluation.py
│
└── Train/ # Training scripts and utilities
  ├── Multi_GRPOTrainer.py # Multi-reward GRPO training
  ├── sentence_trainer.py # Sentence transformer training
  └── SFTTrainer.py # Supervised fine-tuning


## 🚀 Quick Start

### Prerequisites

- **Hardware**: Validated on 1x Node with **4x NVIDIA H100** GPUs.
- **Environment**:
  - Python 3.11+
  - **CUDA 12.8**
  - **PyTorch 2.9.0**
- UV package manager (or pip)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd literary_style_model
```
2. Install dependencies:
```uv sync
```
or building Apptainer image:
```bash
apptainer build literary_style_model.sif docker.def
``` 
## 📚 Usage Guide
Stage 1: Train Sentence Transformer Reward Model
Train a sentence transformer to evaluate style similarity using the judge dataset:
```bash
torchrun --nnodes=1 --nproc_per_node=4 Cross_DDP_trainer.py \
        --model_name "model_you_want" \
        --batch_size 16 \
        --grad_accum 1 \
        --trust_remote
```   
Stage 2: Supervised Fine-Tuning (SFT)
Train the base language model using distributed training with accelerate and FSDP. Run the following command:
```bash
accelerate launch --config_file config/fsdp_config.yaml Train/SFTTrainer_multiGPU.py \
    --output_dir "./final_models/sft-Llama-3.1-8B-SFT" \
    --dataset "VibrantVista/story-style-SFT-dataset" \
    --model "rshwndsz/Llama-3.1-8B-SFT" \
    --max_length 4096 \
    --batch_size 2 \
    --accumulation_steps 1 \
    --epochs 1 \
    --use_wandb True
```
Stage 3: GRPO (Group Relative Policy Optimization)
The GRPO stage requires a two-step process: hosting the reward model judge via vLLM and then running the training script.

### 1. Start the Reward Model Server (vLLM)
Open a terminal and start the OpenChat server which serves as the "Judge" for content quality and coherence.
```bash
vllm serve openchat/openchat-3.5-0106 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.15 \
  --trust-remote-code \
  --async-scheduling
```
### 2. Run GRPO Training
In a separate terminal, launch the GRPO trainer. This connects to the vLLM server to score generations.
```bash
accelerate launch Train/Multi_GRPOTrainer.py \
    --model_path "./final_models/sft-Llama-3.1-8B-SFT" \
    --dataset_path "VibrantVista/grpo-style-training" \
    --output_dir "./final_models/Llama-8B-GRPO-style-v1" \
    --epochs 1 \
    --learning_rate 1e-6 \
    --batch_size 4 \
    --grad_acc_steps 8 \
    --num_generations 16 \
    --max_length 2200 \
    --beta 0.0005 \
    --max_grad_norm 0.35 \
    --author_name "Twain, Mark" \
    --log_wandb \
    --use_lora False \
    --loss_type "dr_grpo" \
    --scale_rewards "batch"
```
## 🙏 Acknowledgments
Project Gutenberg for providing classic literature texts.
Hugging Face for model hosting and datasets.
TRL library for GRPO implementation.
Sentence Transformers library for embedding models.

## 📄 License
MIT License

## 📚 Citation
If you use this code or methodology in your research, please cite our work:
```bibtex
@misc{liu2025capturingclassicauthorialstyle,
      title={Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning}, 
      author={Jinlong Liu and Mohammed Bahja and Venelin Kovatchev and Mark Lee},
      year={2025},
      eprint={2512.05747},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.05747}, 
}
```