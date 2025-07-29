import random
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from datasets import load_dataset


class UnifiedSimilarityAnalyzer:
    def __init__(self, model_name: str):
        """Initialize with a single sentence transformer model"""
        self.model_name = model_name
        self.chunk_sizes = ["500", "1000", "1500", "2000", "2500", "3000"]
        self.similarity_methods = {
            "cosine": self._cosine_similarity,
        }

        # Load single model
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.\n")

    def load_dataset(self, dataset_name: str):
        """Load dataset from HuggingFace Hub"""
        return load_dataset(dataset_name)

    def _cosine_similarity(self, embeddings1, embeddings2):
        """
        Calculate cosine similarity using sentence_transformers optimized function.
        Automatically handles normalization and uses optimized tensor operations.
        Range: [-1, 1] where 1=perfectly similar, 0=orthogonal, -1=perfectly dissimilar
        """
        return cos_sim(embeddings1, embeddings2).diagonal().numpy()

    def save_pairs(self, pairs, output_dir, analysis_type, chunk_size, seed):
        """Save generated pairs to ensure reproducible results"""
        pairs_dir = Path(output_dir) / "saved_pairs"
        pairs_dir.mkdir(exist_ok=True)

        filename = f"{analysis_type}_{chunk_size}w_seed{seed}.pkl"
        filepath = pairs_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump(pairs, f)

        print(f"Saved {len(pairs)} pairs to {filename}")
        return filepath

    def load_pairs(self, output_dir, analysis_type, chunk_size, seed):
        """Load previously saved pairs if they exist"""
        pairs_dir = Path(output_dir) / "saved_pairs"
        filename = f"{analysis_type}_{chunk_size}w_seed{seed}.pkl"
        filepath = pairs_dir / filename

        if filepath.exists():
            with open(filepath, "rb") as f:
                pairs = pickle.load(f)
            print(f"Loaded {len(pairs)} pairs from {filename}")
            return pairs

        return None
        """Organize chunks by author and title"""
        chunks_by_author = defaultdict(lambda: defaultdict(list))

        for chunk in split_data:
            author = chunk["author"]
            title = chunk["title"]
            chunks_by_author[author][title].append(chunk)

        return chunks_by_author

    def organize_chunks_by_author(self, split_data):
        """Organize chunks by author and title with deterministic ordering"""
        chunks_by_author = defaultdict(lambda: defaultdict(list))

        # Sort data for deterministic ordering
        sorted_data = sorted(
            split_data, key=lambda x: (x["author"], x["title"], x["chunk_label"])
        )

        for chunk in sorted_data:
            author = chunk["author"]
            title = chunk["title"]
            chunks_by_author[author][title].append(chunk)

        return chunks_by_author

    def generate_same_author_pairs(
        self, chunks_by_author, chunk_size, output_dir, seed, max_pairs_per_author=1000
    ):
        """Generate or load same author pairs with deterministic seeding"""
        # Try to load existing pairs first
        pairs = self.load_pairs(output_dir, "same_author", chunk_size, seed)
        if pairs is not None:
            return pairs

        # Generate new pairs with fixed seed
        random.seed(seed)
        np.random.seed(seed)

        # Filter authors with at least 2 titles and sort for deterministic order
        valid_authors = {
            author: books
            for author, books in chunks_by_author.items()
            if len(books) >= 2
        }

        # Sort authors for deterministic processing order
        sorted_authors = sorted(valid_authors.items())

        unique_pairs = []
        used_chunk_labels = set()

        print(
            f"Generating same-author pairs for {len(sorted_authors)} authors with 2+ titles"
        )

        for author, books in tqdm(sorted_authors, desc="Generating same-author pairs"):
            # Reset seed for each author to ensure determinism
            random.seed(seed + hash(author) % 1000)

            # Get all title pairs for this author (sorted for determinism)
            sorted_titles = sorted(books.keys())
            title_pairs = list(combinations(sorted_titles, 2))
            author_pairs = []

            for title1, title2 in title_pairs:
                # Get available chunks (not yet used in this split)
                chunks1 = [
                    c
                    for c in books[title1]
                    if c["chunk_label"] not in used_chunk_labels
                ]
                chunks2 = [
                    c
                    for c in books[title2]
                    if c["chunk_label"] not in used_chunk_labels
                ]

                # Sort chunks for deterministic order
                chunks1.sort(key=lambda x: x["chunk_label"])
                chunks2.sort(key=lambda x: x["chunk_label"])

                # Create pairs from these titles
                min_chunks = min(len(chunks1), len(chunks2))
                if min_chunks > 0:
                    # Use seeded random for reproducible shuffling
                    random.shuffle(chunks1)
                    random.shuffle(chunks2)

                    for i in range(min_chunks):
                        chunk1 = chunks1[i]
                        chunk2 = chunks2[i]

                        # Mark chunks as used in this split
                        used_chunk_labels.add(chunk1["chunk_label"])
                        used_chunk_labels.add(chunk2["chunk_label"])

                        author_pairs.append(
                            {
                                "author": author,
                                "title1": title1,
                                "title2": title2,
                                "chunk1": chunk1,
                                "chunk2": chunk2,
                            }
                        )

            # Sample pairs if too many for this author (with author-specific seed)
            if len(author_pairs) > max_pairs_per_author:
                author_pairs = random.sample(author_pairs, max_pairs_per_author)

            unique_pairs.extend(author_pairs)

        print(f"Generated {len(unique_pairs)} same-author pairs")
        print(f"Used {len(used_chunk_labels)} unique chunks out of total available")

        # Save pairs for future reproducibility
        self.save_pairs(unique_pairs, output_dir, "same_author", chunk_size, seed)

        return unique_pairs

    def generate_cross_author_pairs(
        self,
        chunks_by_author,
        chunk_size,
        output_dir,
        seed,
        max_pairs_per_author_combination=None,
    ):
        """Generate or load cross author pairs with deterministic seeding"""
        # Try to load existing pairs first
        pairs = self.load_pairs(output_dir, "cross_author", chunk_size, seed)
        if pairs is not None:
            return pairs

        # Generate new pairs with fixed seed
        random.seed(seed)
        np.random.seed(seed)

        # Target comparison counts based on same-author analysis
        target_comparisons = {
            "500": 4293,
            "1000": 2052,
            "1500": 1340,
            "2000": 996,
            "2500": 796,
            "3000": 664,
        }
        target_count = target_comparisons.get(chunk_size, 1000)

        authors = list(chunks_by_author.keys())

        # Filter and sort authors for deterministic order
        valid_authors = sorted(
            [author for author in authors if len(chunks_by_author[author]) >= 1]
        )

        if len(valid_authors) < 2:
            print("Need at least 2 authors for cross-author analysis")
            return []

        cross_author_pairs = []
        used_chunk_labels = set()

        print(f"Generating cross-author pairs for {len(valid_authors)} authors")
        print(f"Target comparisons for {chunk_size}-word chunks: {target_count}")

        # Get all possible author pairs (sorted for determinism)
        author_pairs = list(combinations(valid_authors, 2))
        print(f"Generated {len(author_pairs)} author combinations")

        # Calculate pairs per combination to reach target
        if max_pairs_per_author_combination is None:
            pairs_per_combination = max(1, target_count // len(author_pairs))
            print(
                f"Setting {pairs_per_combination} pairs per author combination to reach target"
            )
        else:
            pairs_per_combination = max_pairs_per_author_combination

        for author1, author2 in tqdm(
            author_pairs, desc="Generating cross-author pairs"
        ):
            # Get all chunks from both authors (across all their works)
            chunks1 = []
            for title in sorted(
                chunks_by_author[author1].keys()
            ):  # Sort for determinism
                title_chunks = chunks_by_author[author1][title]
                chunks1.extend(
                    [
                        c
                        for c in sorted(
                            title_chunks, key=lambda x: x["chunk_label"]
                        )  # Sort chunks
                        if c["chunk_label"] not in used_chunk_labels
                    ]
                )

            chunks2 = []
            for title in sorted(
                chunks_by_author[author2].keys()
            ):  # Sort for determinism
                title_chunks = chunks_by_author[author2][title]
                chunks2.extend(
                    [
                        c
                        for c in sorted(
                            title_chunks, key=lambda x: x["chunk_label"]
                        )  # Sort chunks
                        if c["chunk_label"] not in used_chunk_labels
                    ]
                )

            if not chunks1 or not chunks2:
                continue

            # Use seeded random for reproducible shuffling
            random.seed(
                seed + hash((author1, author2)) % 1000
            )  # Pair-specific but deterministic seed
            random.shuffle(chunks1)
            random.shuffle(chunks2)

            # Create pairs between the two authors
            combination_pairs = []
            min_chunks = min(len(chunks1), len(chunks2))
            pairs_to_create = min(min_chunks, pairs_per_combination)

            for i in range(pairs_to_create):
                chunk1 = chunks1[i]
                chunk2 = chunks2[i]

                # Mark chunks as used in this split
                used_chunk_labels.add(chunk1["chunk_label"])
                used_chunk_labels.add(chunk2["chunk_label"])

                combination_pairs.append(
                    {
                        "author1": author1,
                        "author2": author2,
                        "title1": chunk1["title"],
                        "title2": chunk2["title"],
                        "chunk1": chunk1,
                        "chunk2": chunk2,
                    }
                )

            cross_author_pairs.extend(combination_pairs)

        print(f"Generated {len(cross_author_pairs)} cross-author pairs")
        print(f"Used {len(used_chunk_labels)} unique chunks out of total available")

        # Save pairs for future reproducibility
        self.save_pairs(
            cross_author_pairs, output_dir, "cross_author", chunk_size, seed
        )

        return cross_author_pairs

    def encode_text_pairs(self, pairs, batch_size=64):
        """Encode text pairs using the single model with optimized batching for speed"""
        if not pairs:
            return None

        print(f"Encoding texts for {len(pairs)} pairs using {self.model_name}...")

        # Extract all texts for batch processing
        texts1 = [pair["chunk1"]["chunk_text"] for pair in pairs]
        texts2 = [pair["chunk2"]["chunk_text"] for pair in pairs]

        # Use larger batch size and disable progress bar for speed
        embeddings1 = self.model.encode(
            texts1, batch_size=batch_size, show_progress_bar=True
        )
        embeddings2 = self.model.encode(
            texts2, batch_size=batch_size, show_progress_bar=True
        )

        return embeddings1, embeddings2

    def calculate_all_similarities(self, pairs, embeddings, chunk_size, analysis_type):
        """Calculate similarity scores for all methods using pre-computed embeddings - optimized for speed"""
        embeddings1, embeddings2 = embeddings
        all_results = []

        print(f"Computing similarities for {len(self.similarity_methods)} methods...")

        # Pre-compute all similarities at once to avoid duplicate calculations
        similarity_cache = {}
        for method_name, similarity_func in self.similarity_methods.items():
            print(f"  Computing {method_name} similarities...")
            similarity_cache[method_name] = similarity_func(embeddings1, embeddings2)

        # Build results efficiently using cached similarities
        for method_name in self.similarity_methods.keys():
            similarities = similarity_cache[method_name]

            # Vectorized result building for speed
            method_results = []
            for i, (pair, similarity) in enumerate(zip(pairs, similarities)):
                if analysis_type == "same_author":
                    result = {
                        "author": pair["author"],
                        "title1": pair["title1"],
                        "title2": pair["title2"],
                        "chunk_size": int(chunk_size),
                        "model_name": self.model_name,
                        "similarity_method": method_name,
                        "similarity_score": float(similarity),
                        "chunk1_label": pair["chunk1"]["chunk_label"],
                        "chunk2_label": pair["chunk2"]["chunk_label"],
                    }
                else:  # cross_author
                    result = {
                        "author1": pair["author1"],
                        "author2": pair["author2"],
                        "title1": pair["title1"],
                        "title2": pair["title2"],
                        "chunk_size": int(chunk_size),
                        "model_name": self.model_name,
                        "similarity_method": method_name,
                        "similarity_score": float(similarity),
                        "chunk1_label": pair["chunk1"]["chunk_label"],
                        "chunk2_label": pair["chunk2"]["chunk_label"],
                    }
                method_results.append(result)

            all_results.append(pd.DataFrame(method_results))

        return all_results

    def analyze_split(
        self,
        dataset,
        split_name: str,
        output_dir: str,
        analysis_type: str,
        seed: int = 42,
    ):
        """Analyze similarity for a specific split - optimized for speed with reproducible pairs"""
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n=== Analyzing {analysis_type} similarity for split: {split_name} ===")

        # Organize chunks by author
        chunks_by_author = self.organize_chunks_by_author(dataset[split_name])

        # Generate pairs based on analysis type (with saving/loading capability)
        if analysis_type == "same_author":
            pairs = self.generate_same_author_pairs(
                chunks_by_author, split_name, output_dir, seed
            )
        else:  # cross_author
            pairs = self.generate_cross_author_pairs(
                chunks_by_author, split_name, output_dir, seed
            )

        if not pairs:
            print(f"No valid pairs found for split {split_name}")
            return None

        # Encode text pairs with optimized batching for speed
        embeddings = self.encode_text_pairs(pairs, batch_size=64)

        if embeddings is None:
            print(f"Failed to generate embeddings for split {split_name}")
            return None

        # Calculate similarities for all methods with caching to avoid duplicate calculations
        all_results = self.calculate_all_similarities(
            pairs, embeddings, split_name, analysis_type
        )
        all_stats = []

        # Process results for each method efficiently
        for i, method_name in enumerate(self.similarity_methods.keys()):
            results_df = all_results[i]

            # Save results for this method
            output_file = (
                Path(output_dir)
                / f"{analysis_type}_{self.model_name.replace('/', '_')}_{method_name}_{split_name}w.csv"
            )
            results_df.to_csv(output_file, index=False)

            # Generate statistics using vectorized operations for speed
            scores = results_df["similarity_score"].values
            stats = {
                "chunk_size": int(split_name),
                "model_name": self.model_name,
                "similarity_method": method_name,
                "analysis_type": analysis_type,
                "total_comparisons": len(scores),
                "mean_similarity": float(np.mean(scores)),
                "std_similarity": float(np.std(scores)),
                "min_similarity": float(np.min(scores)),
                "max_similarity": float(np.max(scores)),
                "median_similarity": float(np.median(scores)),
                "q25_similarity": float(np.percentile(scores, 25)),
                "q75_similarity": float(np.percentile(scores, 75)),
            }

            if analysis_type == "cross_author":
                stats["unique_author_pairs"] = len(
                    results_df[["author1", "author2"]].drop_duplicates()
                )

            print(f"Method: {method_name} | Chunk: {split_name}w")
            print(f"  Total comparisons: {stats['total_comparisons']}")
            if analysis_type == "cross_author":
                print(f"  Unique author pairs: {stats['unique_author_pairs']}")
            print(
                f"  Mean similarity: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}"
            )

            all_stats.append(stats)

        return all_results, all_stats

    def analyze_all_splits(
        self, dataset_name: str, output_dir: str, analysis_type: str, seed: int = 42
    ):
        """Analyze all splits using single model and all similarity methods - optimized for speed"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        dataset = self.load_dataset(dataset_name)

        all_results = []
        all_stats = []

        # Process each split
        for split_name in self.chunk_sizes:
            if split_name in dataset:
                split_results, split_stats = self.analyze_split(
                    dataset, split_name, output_dir, analysis_type, seed
                )
                if split_results is not None:
                    all_results.extend(split_results)
                    all_stats.extend(split_stats)

        if not all_results:
            print("No results generated!")
            return None, None

        # Combine all results across all methods and splits
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(
            Path(output_dir)
            / f"{analysis_type}_{self.model_name.replace('/', '_')}_all_results.csv",
            index=False,
        )

        # Save summary statistics
        summary_df = pd.DataFrame(all_stats)
        summary_df.to_csv(
            Path(output_dir)
            / f"{analysis_type}_{self.model_name.replace('/', '_')}_summary_stats.csv",
            index=False,
        )

        # Generate detailed tables for this model
        self.generate_method_tables(summary_df, output_dir, analysis_type)

        return combined_df, summary_df

    def generate_method_tables(self, summary_df, output_dir, analysis_type):
        """Store summary data for final table generation"""
        # Just store the data, don't generate tables yet
        pass

    def generate_final_combined_tables(
        self, same_author_summary, cross_author_summary, output_dir
    ):
        """Generate final combined tables grouping same author and cross author by method"""
        output_dir = Path(output_dir)

        print(f"\n{'='*80}")
        print("FINAL COMBINED SIMILARITY ANALYSIS TABLES")
        print(f"{'='*80}")

        # Combine both summaries
        if same_author_summary is not None and cross_author_summary is not None:
            combined_summary = pd.concat(
                [same_author_summary, cross_author_summary], ignore_index=True
            )
        elif same_author_summary is not None:
            combined_summary = same_author_summary
        elif cross_author_summary is not None:
            combined_summary = cross_author_summary
        else:
            print("No data available for table generation")
            return

        # Generate table for each similarity method
        for method in self.similarity_methods.keys():
            method_data = combined_summary[
                combined_summary["similarity_method"] == method
            ].copy()

            if method_data.empty:
                continue

            # Sort by chunk size and analysis type
            method_data = method_data.sort_values(["chunk_size", "analysis_type"])

            # Create formatted table
            table_data = []
            for _, row in method_data.iterrows():
                analysis_label = (
                    "Same Author"
                    if row["analysis_type"] == "same_author"
                    else "Cross Author"
                )
                table_entry = {
                    "Analysis Type": analysis_label,
                    "Chunk Size": f"{int(row['chunk_size'])} words",
                    "Comparisons": f"{int(row['total_comparisons']):,}",
                    "Mean Sim.": f"{row['mean_similarity']:.4f}",
                    "Std. Dev.": f"{row['std_similarity']:.4f}",
                    "Median": f"{row['median_similarity']:.4f}",
                    "Min": f"{row['min_similarity']:.4f}",
                    "Max": f"{row['max_similarity']:.4f}",
                    "Q25": f"{row['q25_similarity']:.4f}",
                    "Q75": f"{row['q75_similarity']:.4f}",
                }
                table_data.append(table_entry)

            table_df = pd.DataFrame(table_data)

            # Save table
            filename = (
                f"combined_table_{self.model_name.replace('/', '_')}_{method}.csv"
            )
            table_path = output_dir / filename
            table_df.to_csv(table_path, index=False)

            # Print table
            print(
                f"\n{self.model_name} - {method.upper()} Similarity (Same Author vs Cross Author)"
            )
            print("-" * 100)
            print(table_df.to_string(index=False))
            print(f"Table saved to: {filename}")
            print()

    def run_analysis(
        self,
        dataset_name: str,
        output_dir: str,
        analysis_type: str = "both",
        seed: int = 42,
    ):
        """Run complete analysis for specified type(s)"""

        same_author_summary = None
        cross_author_summary = None

        if analysis_type == "both":
            # Run same author analysis
            print("\n" + "=" * 60)
            print("RUNNING SAME AUTHOR ANALYSIS")
            print("=" * 60)
            same_results, same_author_summary = self.analyze_all_splits(
                dataset_name, output_dir, "same_author", seed
            )

            # Run cross author analysis
            print("\n" + "=" * 60)
            print("RUNNING CROSS AUTHOR ANALYSIS")
            print("=" * 60)
            cross_results, cross_author_summary = self.analyze_all_splits(
                dataset_name, output_dir, "cross_author", seed
            )

            # Generate final combined tables
            self.generate_final_combined_tables(
                same_author_summary, cross_author_summary, output_dir
            )

            return {
                "same_author": (same_results, same_author_summary),
                "cross_author": (cross_results, cross_author_summary),
            }

        else:
            # Run single analysis type
            results, summary = self.analyze_all_splits(
                dataset_name, output_dir, analysis_type, seed
            )

            # Generate final combined tables (will handle single analysis type)
            if analysis_type == "same_author":
                self.generate_final_combined_tables(summary, None, output_dir)
            else:
                self.generate_final_combined_tables(None, summary, output_dir)

            return {analysis_type: (results, summary)}


if __name__ == "__main__":
    # Available models to test - all work directly with sentence-transformers
    available_models = [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "StyleDistance/styledistance",
        "sentence-transformers/all-distilroberta-v1",
        "",  # Add more models as needed
    ]

    # Configuration - Choose model to test
    config = {
        "model_name": "",  # Change this to test different models
        "dataset_name": "VibrantVista/gutenberg-chunks",
        "output_dir": "./similarity_analysis_results",
        "analysis_type": "both",  # "same_author", "cross_author", or "both"
        "seed": 42,
    }

    print(f"Testing model: {config['model_name']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Analysis type: {config['analysis_type']}")

    # Initialize analyzer with single model
    analyzer = UnifiedSimilarityAnalyzer(config["model_name"])

    # Run analysis
    results = analyzer.run_analysis(
        dataset_name=config["dataset_name"],
        output_dir=config["output_dir"],
        analysis_type=config["analysis_type"],
        seed=config["seed"],
    )

    print(f"\nAnalysis complete! Results saved to {config['output_dir']}")
    print(f"Model tested: {config['model_name']}")
