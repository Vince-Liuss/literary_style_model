import os
import re
import json
from collections import defaultdict
from jsonlines import jsonlines
from pathlib import Path
from tqdm import tqdm
from unstructured.cleaners.core import clean_extra_whitespace

# The 'partition_text' import is no longer needed and has been removed.
import spacy  # <-- Import spaCy


def chunk_text_by_sentence(text: str, nlp, max_words=1500):
    """
    Chunks a single large text into smaller chunks by sentence using spaCy.

    Processes the entire text, iterates through its sentences,
    and groups them into chunks up to max_words.
    """
    chunks = []
    current_chunk_sents = []  # Stores sentence *strings*
    current_count = 0

    if not text or not text.strip():
        return []

    # Process the entire text with spaCy once
    doc = nlp(text)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # len(sent) efficiently gives the token count for the sentence
        # (including words and punctuation, similar to nltk.word_tokenize)
        token_count = len(sent)

        # If adding this sentence would exceed the limit and the
        # current chunk isn't empty, finish the current chunk.
        if current_count + token_count > max_words and current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            current_chunk_sents = [sent_text]
            current_count = token_count
        else:
            # Add the sentence to the current chunk
            current_chunk_sents.append(sent_text)
            current_count += token_count

    # Add the last remaining chunk if it's not empty
    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))

    return chunks


def clean_title(title):
    """
    Cleans a title string by:
      - Replacing line breaks with spaces and collapsing whitespace.
      - Removing chapter ranges, single chapter references, and part numbers.
    """
    title = re.sub(r"[\r\n]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Remove chapter references like: ", Chapters 36 to the Last" or ", Chapter 1"
    title = re.sub(
        r",\s*Chapters?\s+\d+\s*(?:to|-)\s*(?:\d+|the Last)",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r",\s*Chapters?\s+\d+", "", title, flags=re.IGNORECASE)
    # Remove part references like: ", Part 1." or ", Part 5"
    title = re.sub(r",\s*Part\s+\d+\.?", "", title, flags=re.IGNORECASE)
    return title


class GutenbergCleaner:
    """Optimized text cleaner for Project Gutenberg books."""

    # Pre-compiled regex patterns (compiled once, reused many times)
    CURLY_QUOTES = {'"': '"', '"': '"', """: "'", """: "'"}

    # Footnote and annotation patterns
    FOOTNOTE_BLOCK_PATTERN = re.compile(
        r"^\s*\[\d+\]\s+(?:\w+\s+){3,}.*?(?=\n\s*(?:\[|{|\d|$))",
        re.MULTILINE | re.DOTALL,
    )
    SIDENOTE_PATTERN = re.compile(
        r"\[(?:Side[\s-]?note|Sidenote)\s*:\s*[^\]]+\]", re.IGNORECASE
    )
    # Match footnote markers: [1], {26}, (1), etc.
    FOOTNOTE_MARKER_PATTERN = re.compile(r"\[\d+\]|\{\d+\}|\(\s*\d+\s*\)")

    # Match extended footnotes/citations in curly braces
    # Matches: {William Wordsworth (English poet, 1770-1850), "Poems...}
    CURLY_BRACE_CITATION_PATTERN = re.compile(
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL
    )

    TRANSLATOR_NOTE_PATTERN = re.compile(
        r"^\s*-{5,}\s*\*?.+?-{5,}\s*$", re.MULTILINE | re.DOTALL
    )
    SEPARATOR_PATTERN = re.compile(
        r"^\s*\*(?:\s{2,}\*)+\s*$|^\s*[*\-~#]{5,}\s*$", re.MULTILINE
    )

    # Formatting patterns
    EMPHASIS_PATTERN = re.compile(r"[_*]+(\w+)[_*]+")
    ILLUSTRATION_PATTERN = re.compile(r"\[Illustration[^\]]*\]", re.IGNORECASE)
    CHAPTER_PATTERN = re.compile(
        r"^\s*(chapter|Chapter)\s*(\d+|I{1,3}|IV|IX|V?I{0,3})\s*.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    PAGE_NUMBER_PATTERN = re.compile(
        r"^\s*\d+\s*$|^\s*\[pg\s*\d+\]", re.MULTILINE | re.IGNORECASE
    )
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    DOUBLE_NEWLINE_PATTERN = re.compile(r"\n\s*\n+")
    SPACE_PATTERN = re.compile(r"[ \t]+")

    def clean(self, text):
        """Apply all cleaning operations in optimized order."""
        if not text:
            return ""

        # Phase 0: Normalize special whitespace
        text = text.replace("\xa0", " ").replace("\u200b", " ")

        # Phase 1: Remove annotations (footnotes, sidenotes, markers, citations)
        text = self._remove_annotations(text)

        # Phase 2: Normalize formatting (quotes, emphasis)
        text = self._normalize_formatting(text)

        # Phase 3: Remove metadata (chapters, illustrations, page numbers)
        text = self._remove_metadata(text)

        # Phase 4: Final whitespace cleanup
        text = self.DOUBLE_NEWLINE_PATTERN.sub("\n\n", text)
        text = self.SPACE_PATTERN.sub(" ", text)
        text = text.strip("_")

        return text.strip()

    def _remove_annotations(self, text):
        """Remove all annotations: footnotes, sidenotes, markers, citations."""
        # Remove multi-line footnote blocks [1] Text...
        text = self.FOOTNOTE_BLOCK_PATTERN.sub("", text)

        # Remove sidenotes [Sidenote: ...]
        text = self.SIDENOTE_PATTERN.sub("", text)

        # Remove inline footnote markers: [1], {26}, (1), etc.
        text = self.FOOTNOTE_MARKER_PATTERN.sub("", text)

        # Remove extended citations in curly braces
        # {William Wordsworth (English poet, 1770-1850), "Poems...}
        text = self.CURLY_BRACE_CITATION_PATTERN.sub("", text)

        # Remove translator/editor notes with dashes
        text = self.TRANSLATOR_NOTE_PATTERN.sub("", text)

        # Remove separator lines (* * * * *)
        text = self.SEPARATOR_PATTERN.sub("", text)

        return text

    def _normalize_formatting(self, text):
        """Normalize quotes and emphasis markers."""
        # Normalize curly quotes
        for old, new in self.CURLY_QUOTES.items():
            text = text.replace(old, new)

        # Convert double hyphens to em dash
        text = text.replace("--", "—")

        # Remove emphasis markers (_word_ or *word*)
        text = self.EMPHASIS_PATTERN.sub(r"\1", text)

        # Remove decorative characters
        text = re.sub(r"[❦†‡※]", "", text)

        # Remove excessive line-start/end markers
        text = re.sub(r"^\s*[\*_]+\s*$", "", text, flags=re.MULTILINE)

        return text

    def _remove_metadata(self, text):
        """Remove structural metadata: illustrations, chapters, page numbers."""
        # Remove illustrations
        text = self.ILLUSTRATION_PATTERN.sub("", text)
        text = re.sub(r"(?is)illustration:.*?(?=\n\n|\Z)", "", text)
        text = re.sub(r"\{Illustration:.*?\}", "", text, flags=re.DOTALL)
        text = re.sub(
            r"^\s*illustration\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )

        # Remove chapter headers
        text = self.CHAPTER_PATTERN.sub("", text)
        text = re.sub(
            r"^.*\b(chapter|Chapter)\b.*$", "", text, flags=re.MULTILINE | re.IGNORECASE
        )

        # Remove page numbers
        text = self.PAGE_NUMBER_PATTERN.sub("", text)

        # Remove URLs
        text = self.URL_PATTERN.sub("", text)

        # Remove table of contents
        text = self._remove_toc(text)

        return text

    @staticmethod
    def _remove_toc(text):
        """Remove table of contents and chapter listings."""
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


class ExportDatasetToJson(object):
    """Outputs datasets to a nested format."""

    def __init__(self):
        self.cleaner = GutenbergCleaner()
        try:
            # Load the spaCy model once
            self.nlp = spacy.load("en_core_web_md")
            self.nlp.max_length = 10000000
        except IOError:
            print("❌ Error: 'en_core_web_md' model not found.")
            print("Please run: python -m spacy download en_core_web_md")
            raise  # Stop execution if model is not available

    def export(
        self,
        metadata_file: str,
        data_folder: str,
        output_file: str,
    ):
        """
        Exports cleaned book data to a JSONL file.
        Assumes the input metadata_file has already been processed by metadata_clean.py.
        """
        metadata_path = Path(metadata_file)

        if not metadata_path.exists():
            print(f"❌ ERROR: Metadata file not found at '{metadata_file}'.")
            print("Please run the metadata_clean.py script first.")
            return

        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "r") as f:
            loaded_data = json.load(f)

        books_data = loaded_data.get("books", [])
        print(f"\n📋 Metadata file contains: {len(books_data)} books")

        # NEW: Check for missing files
        missing_files = []
        for book in books_data:
            file_path = os.path.join(data_folder, book["filename"])
            if not os.path.exists(file_path):
                missing_files.append(book["gutenberg_id"])

        if missing_files:
            print(
                f"⚠️  Missing files for {len(missing_files)} books: {missing_files[:10]}..."
            )

        # Dictionary to track the sequence number for each unique book ID.
        chunk_counters = defaultdict(int)
        files_processed = 0
        chunks_written = 0
        files_skipped = 0

        with jsonlines.open(output_file, mode="w") as writer:
            for book_metadata in tqdm(
                books_data, desc="Processing and exporting books"
            ):
                gutenberg_id = book_metadata["gutenberg_id"]
                file_path = os.path.join(data_folder, book_metadata["filename"])

                if not os.path.exists(file_path):
                    files_skipped += 1
                    continue

                try:
                    with open(
                        file_path, "r", encoding="utf-8", errors="ignore"
                    ) as book_file:
                        book_text = book_file.read()

                    cleaned_text = self.cleaner.clean(book_text)

                    # Skip if cleaned text is too short
                    if len(cleaned_text) < 100:
                        continue

                    # Removed 'partition_text'
                    # elements = partition_text(text=cleaned_text)

                    # Use the new spaCy-based chunking function on the raw text
                    chunks = chunk_text_by_sentence(
                        cleaned_text, self.nlp, max_words=1500
                    )

                    chunks = [clean_extra_whitespace(str(chunk)) for chunk in chunks]

                    processed_title = clean_title(book_metadata["title"])

                    # Select only primary subject (first one)
                    primary_subject = (
                        book_metadata["subjects"][0]
                        if book_metadata["subjects"]
                        else ""
                    )

                    for chunk_text in chunks:
                        if chunk_text:
                            chunk_counters[gutenberg_id] += 1
                            chunk_label = f"{book_metadata['author']}_[{processed_title}]_{chunk_counters[gutenberg_id]}"

                            record = {
                                "gutenberg_id": gutenberg_id,
                                "author": book_metadata["author"],
                                "title": processed_title,
                                "subjects": [primary_subject],  # Single subject only
                                "chunk_label": chunk_label,
                                "chunk_text": chunk_text,
                            }
                            writer.write(record)
                            chunks_written += 1

                    files_processed += 1

                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {str(e)}")
                    continue

        print(f"\n✅ Export completed!")
        print(f"📊 Files processed: {files_processed}")
        print(f"⏭️  Files skipped: {files_skipped}")
        print(f"📝 Total chunks written: {chunks_written}")
        print(f"💾 Output file: {output_file}")


if __name__ == "__main__":

    # For interactive debugging in VS Code:
    defaults = {
        "metadata_file": "../data/cleaned_metadata.json",
        "data_folder": "../data/",
        "output_file": "../data/clean_dataset_all.jsonl",
    }
    ExportDatasetToJson().export(**defaults)
