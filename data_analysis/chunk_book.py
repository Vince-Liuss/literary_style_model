import copy
import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, DatasetDict
from unstructured.cleaners.core import clean_extra_whitespace
from unstructured.partition.text import partition_text
import nltk


def chunk_paragraphs_multiple_sizes(
    paragraphs, chunk_sizes=[500, 1000, 1500, 2000, 2500, 3000]
):
    """Create chunks for multiple word count limits"""
    results = {}

    for max_words in chunk_sizes:
        chunks = []
        current_chunk = []
        current_count = 0

        for para in paragraphs:
            para_text = para.text if hasattr(para, "text") else para
            tokens = nltk.word_tokenize(para_text)
            token_count = len(tokens)

            if current_count + token_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

            current_chunk.append(para_text)
            current_count += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        results[max_words] = chunks

    return results


def clean_title(title):
    """Clean title string by removing chapter/part references"""
    title = re.sub(r"[\r\n]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(
        r",\s*Chapters?\s+\d+\s*(?:to|-)\s*(?:\d+|the Last)",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r",\s*Chapters?\s+\d+", "", title, flags=re.IGNORECASE)
    title = re.sub(r",\s*Part\s+\d+\.?", "", title, flags=re.IGNORECASE)
    return title


class GutenbergCleaner:
    def __init__(self):
        self.cleaning_functions = [
            self.normalize_quotes,
            self.clean_special_chars,
            self.remove_notes,
            self.format_chapter_headings,
            self.remove_chapter_headers,
            self.remove_page_numbers,
            self.handle_illustrations,
            self.remove_urls,
            self.clean_gutenberg_content,
        ]

    def clean(self, text):
        for func in self.cleaning_functions:
            text = func(text)
        return text.strip("_")

    @staticmethod
    def normalize_quotes(text):
        return (
            text.replace('"', '"').replace('"', '"').replace(""", "'").replace(""", "'")
        )

    @staticmethod
    def clean_special_chars(text):
        text = re.sub(r"[_*]+", "", text)
        text = re.sub(r"--+", "—", text)
        text = re.sub(r"[❦]", "", text)
        text = re.sub(r"[^\w\s.,!?;:()-]", "", text)
        return text

    @staticmethod
    def remove_notes(text):
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\(\s*\d+\s*\)", "", text)
        return text

    @staticmethod
    def format_chapter_headings(text):
        pattern = r"^(chapter\s+[\divxlc]+\s*\.?)(.*)$"
        text = re.sub(pattern, r"## \1\n\n\2", text, flags=re.MULTILINE | re.IGNORECASE)
        return text

    @staticmethod
    def remove_chapter_headers(text):
        text = re.sub(
            r"^\s*(chapter|Chapter)\s*(\d+|I{1,3}|IV|IX|V?I{0,3})\s*.*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        text = re.sub(
            r"^.*\b(chapter|Chapter)\b.*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )
        return text

    @staticmethod
    def remove_page_numbers(text):
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[pg\s*\d+\]", "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def handle_illustrations(text):
        text = re.sub(
            r"^\s*illustration\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )
        text = re.sub(
            r"\[illustration(?:\:|\s).*?\]", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"(?is)illustration:.*?(?=\n\n|\Z)", "", text)
        text = re.sub(r"\{Illustration:.*?\}", "", text, flags=re.DOTALL)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
    def clean_gutenberg_content(text):
        lines = text.split("\n")
        cleaned_lines = []
        skip_section = False

        for line in lines:
            if re.match(
                r"^(Contents|VOLUME|CHAPTER|List of Illustrations)",
                line.strip(),
                re.IGNORECASE,
            ):
                skip_section = True
            elif line.strip() and not re.match(r"^\d+\.?\s", line.strip()):
                skip_section = False

            if not skip_section and not re.match(
                r"^(VOLUME|CHAPTER)", line.strip(), re.IGNORECASE
            ):
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def remove_urls(text):
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return url_pattern.sub("", text)


class DatasetBuilder:
    def __init__(self):
        self.cleaner = GutenbergCleaner()
        self.chunk_sizes = [500, 1000, 1500, 2000, 2500, 3000]

    def build_dataset(
        self,
        metadata_file: str,
        data_folder: str,
        dataset_name: str,
        push_to_hub: bool = True,
    ):
        """Build HuggingFace dataset with different chunk size splits"""

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Store data for each split with separate numbering
        splits_data = {str(size): [] for size in self.chunk_sizes}
        # Separate unique_label_dict for each chunk size
        unique_label_dicts = {str(size): {} for size in self.chunk_sizes}

        # Process each book
        for book_metadata in tqdm(metadata, desc="Processing books"):
            gutenberg_id = book_metadata["gutenberg_id"]
            author = book_metadata["author"]
            title = book_metadata["title"]
            subject = book_metadata["subject"]
            filename = book_metadata["filename"]
            file_path = os.path.join(data_folder, filename)

            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as book_file:
                book_text = book_file.read()
                cleaned_text = self.cleaner.clean(book_text)

                # Skip if text is too short
                if len(cleaned_text.split()) < 100:
                    continue

                # Partition text into elements
                elements = partition_text(text=cleaned_text)

                # Create chunks for all sizes
                chunks_dict = chunk_paragraphs_multiple_sizes(
                    elements, self.chunk_sizes
                )

                # Process title
                processed_title = clean_title(title)

                # Store chunks for each size using original numbering logic
                for chunk_size, chunks in chunks_dict.items():
                    # Build unique key as Author_cleanedTitle (same as original)
                    book_key = f"{author}_{processed_title}"
                    if book_key not in unique_label_dicts[str(chunk_size)]:
                        unique_label_dicts[str(chunk_size)][book_key] = 1

                    for chunk in chunks:
                        if len(chunk.split()) < 50:  # Skip very short chunks
                            continue

                        # Assign sequential label for this (author, processed_title) pair (same as original)
                        chunk_label = f"{author}_[{processed_title}]_{unique_label_dicts[str(chunk_size)][book_key]}"
                        unique_label_dicts[str(chunk_size)][book_key] += 1

                        chunk_data = {
                            "chunk_label": chunk_label,
                            "chunk_text": clean_extra_whitespace(chunk),
                            "gutenberg_id": gutenberg_id,
                            "author": author,
                            "title": processed_title,
                            "subject": subject,
                            "chunk_size": chunk_size,
                            "chunk_index": unique_label_dicts[str(chunk_size)][book_key]
                            - 1,
                        }
                        splits_data[str(chunk_size)].append(chunk_data)

        # Create HuggingFace DatasetDict
        dataset_dict = {}
        for chunk_size, data in splits_data.items():
            if data:  # Only create split if data exists
                dataset_dict[chunk_size] = Dataset.from_list(data)
                print(f"Split '{chunk_size}': {len(data)} chunks")

        hf_dataset = DatasetDict(dataset_dict)

        # Push to hub if requested
        if push_to_hub:
            hf_dataset.push_to_hub(dataset_name)
            print(f"Dataset pushed to HuggingFace Hub: {dataset_name}")

        return hf_dataset


if __name__ == "__main__":
    builder = DatasetBuilder()

    # Configuration
    config = {
        "metadata_file": "../books/metadata.json",
        "data_folder": "../books",
        "dataset_name": "VibrantVista/gutenberg-chunks",
        "push_to_hub": True,
    }

    dataset = builder.build_dataset(**config)
