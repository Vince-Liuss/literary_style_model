# Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning

This project implements a multi-stage pipeline for training language models to write in the style of classic authors (specifically Mark Twain). The approach uses sentence transformers as reward models and Group Relative Policy Optimization (GRPO) reinforcement learning techniques to achieve high-quality literary style transfer in long-form story generation.

## 🎯 Project Overview

The project consists of three main stages:

1. **Reward Model Training**: Fine-tune sentence transformers to evaluate literary style similarity
2. **Supervised Fine-Tuning (SFT)**: Initial training of language models on high-quality literary examples
3. **GRPO Training**: Reinforcement learning optimization using Group Relative Policy Optimization

## 📚 Datasets

This project uses the following datasets, all available on HuggingFace:

### Training Datasets
- **Style Similarity Dataset**: [`VibrantVista/StyleScoredPairsDataset-HybridSplit-Final`](https://huggingface.co/datasets/VibrantVista/StyleScoredPairsDataset-HybridSplit-Final)
  - Used for training sentence transformer reward models
  - Contains pairs of text with style similarity scores

- **SFT Dataset**: [`VibrantVista/SFT_storywriting_1500words`](https://huggingface.co/datasets/VibrantVista/SFT_storywriting_1500words)
  - High-quality 1500-word story examples for supervised fine-tuning

- **GRPO Dataset**: [`VibrantVista/GRPO_twain-prompts-4450`](https://huggingface.co/datasets/VibrantVista/GRPO_twain-prompts-4450)
  - Writing prompts for GRPO training and sample style reference

- **Source Text Chunks**: [`VibrantVista/gutenberg-chunks`](https://huggingface.co/datasets/VibrantVista/gutenberg-chunks)
  - Processed text chunks from Project Gutenberg classics
  - Used for style analysis and dataset creation

## 📁 Project Structure

```
literary_style_model/
├── data/                          # Generated datasets and model outputs
│   └── model_completions.csv
├── data_analysis/                 # Analysis and evaluation tools
│   ├── chunk_book.py             # Text chunking utilities
│   └── Unified_analysis.py       # Style similarity analysis
├── data_processing/               # Dataset creation and processing
│   ├── export_dataset_to_json.py # Dataset export utilities
│   ├── GRPO_dataset_builder.py   # GRPO training dataset creation
│   ├── gutenberg_download.py     # Project Gutenberg text download
│   ├── prompts.py                # Writing prompts for training
│   ├── Refill.py                 # Text replacement/completion tasks
│   ├── SFT_dataset_builder.py    # SFT dataset creation
│   ├── Style_evaluation.py       # Style and content evaluation
│   └── upload_triplet_dataset.py # Dataset uploading utilities
└── Train/                        # Training scripts and utilities
    ├── ds_config.json            # DeepSpeed configuration
    ├── Multi_GRPOTrainer.py      # Multi-reward GRPO training
    ├── sentence_trainer.py       # Sentence transformer training
    ├── SFTTrainer.py             # Supervised fine-tuning
    └── utils.py                  # Training utilities and helpers
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU(s) recommended
- UV package manager (or pip)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd literary_style_model
```

2. Install dependencies:
```bash
uv sync
# or with pip:
# pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export WANDB_API_KEY="your_wandb_key"  # Optional, for experiment tracking
export HF_TOKEN="your_huggingface_token"  # For model uploads
```

## 📚 Usage Guide

### Stage 1: Train Sentence Transformer Reward Model

First, train a sentence transformer to evaluate style similarity:

```bash
python Train/sentence_trainer.py
```

**Key configuration in `sentence_trainer.py`:**
- `model_name`: Base sentence transformer model (e.g., `"sentence-transformers/all-mpnet-base-v2"`)
- `dataset_name`: Style similarity dataset (e.g., `"VibrantVista/StyleScoredPairsDataset-HybridSplit-Final"`)
- `train_batch_size`: Training batch size (default: 64)
- `num_train_epochs`: Number of training epochs (default: 5)

The trained model will be saved and can be used as a reward model in later stages.

### Stage 2: Supervised Fine-Tuning (SFT)

Train the base language model on high-quality literary examples:

```bash
python Train/SFTTrainer.py \
    --dataset "VibrantVista/SFT_dataset" \
    --model "/path/to/your/base-model" \
    --output_dir "./storyteller_model" \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --use_lora \
    --use_wandb \
    --wandb_project "storyteller-sft"
```

**Parameters:**
- `--dataset`: HuggingFace dataset with literary examples
- `--model`: Base model to fine-tune
- `--use_lora`: Enable LoRA for parameter-efficient training
- `--use_wandb`: Enable Weights & Biases logging

### Stage 3: GRPO (Group Relative Policy Optimization)

Fine-tune the model using multiple reward functions with distributed training:

```bash
accelerate launch --num_processes=4 Train/Multi_GRPOTrainer.py \
    --model_path "/path/to/your/sft-model" \
    --reward_model_path "/path/to/your/finetune-style-all-mpnet-base-v2-final" \
    --dataset_path "VibrantVista/GRPO_twain-prompts-4450" \
    --output_dir "/path/to/output/grpo_style_model" \
    --num_iterations 1 \
    --learning_rate 3e-6 \
    --batch_size 4 \
    --grad_acc_steps 2 \
    --num_generations 16 \
    --max_length 1500 \
    --beta 0.035 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --max_grad_norm 1.0 \
    --log_wandb \
    --use_lora False \
    --loss_type "dr_grpo" \
    --scale_rewards False
```

**Key Parameters:**
- `--model_path`: Path to SFT model from Stage 2 (can be HuggingFace model ID)
- `--reward_model_path`: Path to trained sentence transformer from Stage 1
- `--num_generations`: Number of completions per prompt (16 for better comparison)
- `--beta`: KL divergence penalty (0.035 for balanced exploration)
- `--epsilon`: PPO-style clipping parameter (0.2)
- `--epsilon_high`: Upper clipping bound (0.28)
- `--loss_type`: Loss formulation ("dr_grpo" for global normalization)
- `--scale_rewards`: Whether to normalize rewards (False recommended)

## 🔧 Data Processing

### Creating Training Datasets

1. **Download source texts**:
```bash
python data_processing/gutenberg_download.py
```

2. **Build SFT dataset**:
```bash
python data_processing/SFT_dataset_builder.py
```

3. **Create GRPO dataset**:
```bash
python data_processing/GRPO_dataset_builder.py
```

### Style Analysis

Analyze style similarity between texts:
```bash
python data_analysis/Unified_analysis.py
```

## 📊 Evaluation

### Style Evaluation

The project includes comprehensive evaluation metrics:

- **Style Reward**: Cosine similarity between generated text and target author's style
- **Content Reward**: Quality assessment using fine-tuned language models
- **Completeness Reward**: Evaluation of text completion and coherence

Run evaluation:
```bash
python data_processing/Style_evaluation.py
```

## 🎛️ Configuration

### Training Hyperparameters

**Sentence Transformer Training:**
- Learning rate: 2e-5
- Batch size: 64
- Early stopping threshold: 0.003
- Loss function: CosineSimilarityLoss

**SFT Training:**
- Learning rate: 2e-4
- Batch size: 2
- Gradient accumulation: 4 steps
- Max sequence length: 2048

**GRPO Training:**
- Learning rate: 3e-6
- Beta (KL penalty): 0.035
- Epsilon (clipping): 0.2
- Epsilon high (upper clipping): 0.28
- Num generations: 16
- Max completion length: 1500
- Batch size: 4
- Gradient accumulation steps: 2
- Loss type: dr_grpo

### Model Architectures

The project supports various model architectures:
- **Language Models**: Llama 2/3, Qwen, Mistral
- **Sentence Transformers**: all-mpnet-base-v2, all-MiniLM-L6-v2
- **Reward Models**: OpenChat-3.5 for content evaluation

## 🛠️ Advanced Features

### LoRA (Low-Rank Adaptation)

Enable parameter-efficient training:
```python
use_lora = True
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
```

### Multi-GPU Training

The project supports distributed training with Accelerate:
```bash
accelerate launch --num_processes=4 Train/Multi_GRPOTrainer.py [args]
```

For GRPO training, use the accelerate launcher as shown in Stage 3 above. The `--num_processes` parameter should match your available GPU count.

### Experiment Tracking

Integration with Weights & Biases for experiment monitoring:
- Loss curves and reward progression
- Generated text samples
- Model performance metrics
- Hyperparameter tracking

## 📈 Expected Results

After complete training, the model should demonstrate:
- **Style Consistency**: Generated text matches Mark Twain's distinctive voice and mannerisms
- **Content Quality**: Coherent, engaging narratives with proper structure
- **Completeness**: Full, well-formed stories rather than fragments
- **Authenticity**: Appropriate use of vernacular speech and period-appropriate content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Project Gutenberg for providing classic literature texts
- Hugging Face for model hosting and datasets
- TRL library for GRPO implementation
- Sentence Transformers library for embedding models

## 📚 Citation

If you use this code or methodology in your research, please cite our work:

```bibtex
@article{your_paper_2025,
  title={Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning},
  author={[Your Name(s)]},
  journal={[Conference/Journal Name]},
  year={2025},
  note={Under review}
}
```

*Note: This citation format will be updated upon publication acceptance.*