import os
import re
import json
import spacy
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, DatasetDict
from unstructured.cleaners.core import clean_extra_whitespace

# Regex optimizations: Compiled at module level to avoid runtime overhead
ZW_CHARS_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
BIDI_RE = re.compile(r"[\u200E\u200F\u061C]")
NBSP_RE = re.compile(r"[\u00A0\u202F\u2007]")
SOFT_HYPHEN_RE = re.compile(r"\u00AD")
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
DOUBLE_NEWLINE_PATTERN = re.compile(r"\n\s*\n+")
SPACE_PATTERN = re.compile(r"[ \t]+")


def clean_title(title: str) -> str:
    """Sanitizes titles for file/label naming."""
    if not title:
        return "unknown_title"
    title = re.sub(r"[\r\n]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Strip Gutenberg-specific metadata (Chapters X-Y, Part Z)
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
    """
    High-performance regex cleaner.
    Order of operations is tuned to remove largest chunks of noise first.
    """

    CURLY_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}

    # Compile heavy patterns once
    FOOTNOTE_BLOCK = re.compile(
        r"^\s*\[\d+\]\s+(?:\w+\s+){3,}.*?(?=\n\s*(?:\[|{|\d|$))",
        re.MULTILINE | re.DOTALL,
    )
    SIDENOTE = re.compile(
        r"\[(?:Side[\s-]?note|Sidenote)\s*:\s*[^\]]+\]", re.IGNORECASE
    )
    FOOTNOTE_MARKER = re.compile(r"\[\d+\]|\{\d+\}|\(\s*\d+\s*\)")
    CURLY_CITATION = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)
    TRANS_NOTE = re.compile(r"^\s*-{5,}\s*\*?.+?-{5,}\s*$", re.MULTILINE | re.DOTALL)
    SEPARATOR = re.compile(r"^\s*\*(?:\s{2,}\*)+\s*$|^\s*[*\-~#]{5,}\s*$", re.MULTILINE)
    EMPHASIS = re.compile(r"[_*]+(\w+)[_*]+")
    ILLUSTRATION = re.compile(r"\[Illustration[^\]]*\]", re.IGNORECASE)
    CHAPTER_HEADER = re.compile(
        r"^\s*(chapter|Chapter)\s*(\d+|I{1,3}|IV|IX|V?I{0,3})\s*.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    PAGE_NUM = re.compile(r"^\s*\d+\s*$|^\s*\[pg\s*\d+\]", re.MULTILINE | re.IGNORECASE)
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 1. Cheap string replacements first
        text = text.replace("\xa0", " ").replace("\u200b", " ")

        # 2. Heavy Regex (Structure Removal)
        text = self.FOOTNOTE_BLOCK.sub("", text)
        text = self.SIDENOTE.sub("", text)
        text = self.FOOTNOTE_MARKER.sub("", text)
        text = self.CURLY_CITATION.sub("", text)
        text = self.TRANS_NOTE.sub("", text)
        text = self.SEPARATOR.sub("", text)

        # 3. Formatting Normalization
        for old, new in self.CURLY_QUOTES.items():
            text = text.replace(old, new)
        text = text.replace("--", "—")
        text = self.EMPHASIS.sub(r"\1", text)
        text = re.sub(r"[❦†‡※]", "", text)
        text = re.sub(r"^\s*[\*_]+\s*$", "", text, flags=re.MULTILINE)

        # 4. Metadata Stripping
        text = self.ILLUSTRATION.sub("", text)
        text = re.sub(r"(?is)illustration:.*?(?=\n\n|\Z)", "", text)
        text = re.sub(r"\{Illustration:.*?\}", "", text, flags=re.DOTALL)
        text = self.CHAPTER_HEADER.sub("", text)
        text = re.sub(
            r"^.*\b(chapter|Chapter)\b.*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )
        text = self.PAGE_NUM.sub("", text)
        text = self.URL_PATTERN.sub("", text)
        text = self._remove_toc(text)

        # 5. Final Whitespace Collapse
        text = DOUBLE_NEWLINE_PATTERN.sub("\n\n", text)
        text = SPACE_PATTERN.sub(" ", text)
        return text.strip("_").strip()

    @staticmethod
    def _remove_toc(text: str) -> str:
        """Heuristic filter to strip Table of Contents lines."""
        lines = text.split("\n")
        cleaned_lines = []
        skip = False
        for line in lines:
            # Trigger skip on TOC headers
            if re.match(
                r"^(Contents|VOLUME|CHAPTER|List of Illustrations)",
                line.strip(),
                re.IGNORECASE,
            ):
                skip = True
            # Stop skipping when we hit body text (non-numeric/header line)
            elif line.strip() and not re.match(r"^\d+\.?\s", line.strip()):
                skip = False

            if not skip and not re.match(
                r"^(VOLUME|CHAPTER)", line.strip(), re.IGNORECASE
            ):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


class DatasetBuilder:
    """
    Pipeline: Disk Read -> CPU Clean -> GPU Tokenize -> CPU Chunk -> Disk Save.
    Optimized to minimize cross-device transfer and redundant computation.
    """

    def __init__(self):
        self.cleaner = GutenbergCleaner()

        # GPU Activation
        if spacy.prefer_gpu():
            print("🚀 GPU Acceleration: ENABLED")
            try:
                spacy.require_gpu()  # Force allocation
            except:
                pass
        else:
            print("⚠️  GPU Acceleration: DISABLED (Running on CPU)")

        try:
            print("⏳ Loading SpaCy model...")
            # Disable unused pipes to save VRAM and improve speed (30-50% faster)
            self.nlp = spacy.load(
                "en_core_web_md",
                disable=["ner", "tagger", "lemmatizer", "attribute_ruler"],
            )
            self.nlp.max_length = 10000000  # Support full books
            print("✅ SpaCy loaded.")
        except IOError:
            raise RuntimeError("Run `python -m spacy download en_core_web_md` first.")

    def _book_stream(self, books_data, data_folder):
        """
        Generator for nlp.pipe.
        Handles IO and Cleaning parallel to GPU processing in the main loop.
        """
        for book in books_data:
            path = os.path.join(data_folder, book["filename"])
            if not os.path.exists(path):
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()

                # Clean text immediately after read
                cleaned = self.cleaner.clean(raw)

                # Filter tiny files
                if len(cleaned) < 2000:
                    continue  # ~500 words

                yield cleaned, book
            except Exception:
                continue

    def precompute_sentences(self, doc):
        """
        CRITICAL OPTIMIZATION:
        Iterates the heavy SpaCy Doc ONCE. Extracts text and valid word counts.
        Returns lightweight Python objects for the multi-pass chunker.
        """
        sents_data = []

        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue

            # Strict word count: exclude punctuation and whitespace tokens
            # Accessing .is_punct is a C-level call, doing this once per token is vital.
            count = sum(1 for t in sent if not t.is_punct and not t.is_space)

            if count > 0:
                sents_data.append((text, count))

        return sents_data

    def create_chunks_from_precomputed(self, sents_data, max_words):
        """
        Generates chunks using pre-calculated metadata.
        Pure Python logic, extremely fast.
        """
        chunks = []
        buffer_text = []
        curr_count = 0

        for text, count in sents_data:
            if curr_count + count > max_words and buffer_text:
                # Flush buffer
                chunks.append(" ".join(buffer_text))
                buffer_text = [text]
                curr_count = count
            else:
                buffer_text.append(text)
                curr_count += count

        # Flush remaining
        if buffer_text:
            chunks.append(" ".join(buffer_text))

        return chunks

    def build(self, metadata_file, data_folder, hf_output_path):
        meta_path = Path(metadata_file)
        if not meta_path.exists():
            return

        Path(hf_output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "r") as f:
            books_data = json.load(f).get("books", [])

        print(f"\n📋 Processing {len(books_data)} books...")

        TARGET_SIZES = [500, 1000, 1500, 2000, 2500, 3000]

        # Data structures
        split_buffer = defaultdict(list)
        chunk_counters = defaultdict(lambda: defaultdict(int))
        total_chunks = defaultdict(int)
        files_processed = 0

        # Stream processing
        # batch_size=4 is conservative for 16GB VRAM with large books. Increase if VRAM allows.
        # n_process=1 is safe; multiprocessing with GPU requires 'spawn' context (complex).
        stream = self.nlp.pipe(
            self._book_stream(books_data, data_folder), as_tuples=True, batch_size=4
        )

        pbar = tqdm(total=len(books_data), desc="Processing")

        for doc, meta in stream:
            # 1. One-time expensive pass over Doc
            sents_data = self.precompute_sentences(doc)

            if not sents_data:
                pbar.update(1)
                continue

            gid = meta["gutenberg_id"]
            auth = meta["author"]
            # Clean title once
            title = clean_title(meta["title"])
            subj = meta["subjects"][0] if meta["subjects"] else "unknown"

            # 2. Multi-pass chunking (Fast on precomputed data)
            for size in TARGET_SIZES:
                chunks = self.create_chunks_from_precomputed(sents_data, size)

                for txt in chunks:
                    # Final whitespace polish
                    txt = clean_extra_whitespace(txt)
                    if not txt:
                        continue

                    chunk_counters[gid][size] += 1
                    total_chunks[size] += 1

                    label = f"{auth}_[{title}]_{size}_{chunk_counters[gid][size]}"

                    # Store in memory buffer
                    split_buffer[str(size)].append(
                        {
                            "gutenberg_id": gid,
                            "author": auth,
                            "title": title,
                            "subjects": subj,
                            "chunk_label": label,
                            "chunk_text": txt,
                            "target_length": size,
                        }
                    )

            files_processed += 1
            pbar.update(1)

            # Update progress bar with live stats
            stats = " | ".join([f"Sz{k}:{v}" for k, v in total_chunks.items()])
            pbar.set_postfix_str(stats)

        pbar.close()

        # --- Saving ---
        print(f"\n✅ Processed {files_processed} books.")
        print(f"🏗️  Building Hugging Face DatasetDict...")

        hf_splits = {k: Dataset.from_list(v) for k, v in split_buffer.items()}
        final_ds = DatasetDict(hf_splits)

        print(f"💾 Saving to: {hf_output_path}")
        final_ds.save_to_disk(hf_output_path)
        print("✅ Done.")


if __name__ == "__main__":
    # Configuration
    # METADATA = "../data/cleaned_metadata.json"
    DATA_DIR = "../data"
    METADATA = "../data/grpo_target_books.json"
    OUTPUT_DIR = "../data/target_books"
    # OUTPUT_DIR = "../data/benchmark_test_dataset_hf"

    builder = DatasetBuilder()
    builder.build(METADATA, DATA_DIR, OUTPUT_DIR)
