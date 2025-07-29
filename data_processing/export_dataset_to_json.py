import copy
import os
import re
import fire
import random
from datasets import load_dataset
from jsonlines import jsonlines
from pathlib import Path
from tqdm import tqdm
import json
from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes, remove_punctuation, group_broken_paragraphs
from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements
import nltk



def chunk_paragraphs(paragraphs, max_words=1500):
    chunks = []
    current_chunk = []
    current_count = 0
    for para in paragraphs:
        # Assuming each element is an object with a .text attribute.
        para_text = para.text if hasattr(para, 'text') else para
        # Tokenize paragraph using nltk
        tokens = nltk.word_tokenize(para_text)
        token_count = len(tokens)
        # If adding this paragraph would exceed the limit and current_chunk isn't empty,
        # finish the current chunk and start a new one.
        if current_count + token_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
        # Add the paragraph to the current chunk
        current_chunk.append(para_text)
        current_count += token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def clean_title(title):
    """
    Cleans a title string by:
      - Replacing line breaks with spaces and collapsing whitespace.
      - Removing chapter ranges, single chapter references, and part numbers.
    """
    title = re.sub(r'[\r\n]+', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    # Remove chapter references like: ", Chapters 36 to the Last" or ", Chapter 1"
    title = re.sub(r',\s*Chapters?\s+\d+\s*(?:to|-)\s*(?:\d+|the Last)', '', title, flags=re.IGNORECASE)
    title = re.sub(r',\s*Chapters?\s+\d+', '', title, flags=re.IGNORECASE)
    # Remove part references like: ", Part 1." or ", Part 5"
    title = re.sub(r',\s*Part\s+\d+\.?', '', title, flags=re.IGNORECASE)
    return title


class GutenbergCleaner:
    def __init__(self):
        """Initialize cleaning functions."""
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
        """Apply all cleaning functions to the text."""
        for func in self.cleaning_functions:
            text = func(text)
        return text.strip('_')

    @staticmethod
    def normalize_quotes(text):
        """Normalize curly quotes to straight quotes."""
        return text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

    @staticmethod
    def clean_special_chars(text):
        """Remove emphasis markers, convert double hyphens to em dash, and remove special characters."""
        # Remove underscores and asterisks used for emphasis
        text = re.sub(r'[_*]+', '', text)
        # Replace two or more hyphens with an em dash
        text = re.sub(r'--+', '—', text)
        # Remove specific special characters like ❦
        text = re.sub(r'[❦]', '', text)
        # Remove any remaining non-alphanumeric characters except for common punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        return text

    @staticmethod
    def remove_notes(text):
        """Remove bracketed notes and parenthesized footnote numbers."""
        text = re.sub(r'\[[^\]]+\]', '', text)
        text = re.sub(r'\(\s*\d+\s*\)', '', text)
        return text

    @staticmethod
    def format_chapter_headings(text):
        """Format chapter headings in a markdown style."""
        pattern = r'^(chapter\s+[\divxlc]+\s*\.?)(.*)$'
        text = re.sub(pattern, r'## \1\n\n\2', text,
                      flags=re.MULTILINE | re.IGNORECASE)
        return text

    @staticmethod
    def remove_chapter_headers(text):
        """Remove residual chapter headers if needed."""
        # Remove lines that start with "CHAPTER" or "Chapter" followed by any number or Roman numeral
        text = re.sub(
            r'^\s*(chapter|Chapter)\s*(\d+|I{1,3}|IV|IX|V?I{0,3})\s*.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove lines that contain "CHAPTER" or "Chapter" anywhere, assuming they are part of a heading
        text = re.sub(r'^.*\b(chapter|Chapter)\b.*$', '',
                      text, flags=re.MULTILINE | re.IGNORECASE)

        return text

    @staticmethod
    def remove_page_numbers(text):
        """Remove standalone page numbers and [Pg XX] markers."""
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[pg\s*\d+\]', '', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def handle_illustrations(text):
        """Remove illustration markers and descriptions."""
        # Remove lines that consist solely of "Illustration"
        text = re.sub(r'^\s*illustration\s*$', '', text,
                      flags=re.MULTILINE | re.IGNORECASE)
        # Remove inline illustration markers like "[Illustration: ...]"
        text = re.sub(r'\[illustration(?:\:|\s).*?\]', '',
                      text, flags=re.DOTALL | re.IGNORECASE)
        # Remove multi-line illustration descriptions beginning with "Illustration:"
        text = re.sub(r'(?is)illustration:.*?(?=\n\n|\Z)', '', text)
        # Remove illustration captions enclosed in curly braces
        text = re.sub(r'\{Illustration:.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text

    @staticmethod
    def clean_gutenberg_content(text):
        """Remove table of contents and chapter listings."""
        lines = text.split('\n')
        cleaned_lines = []
        skip_section = False

        for line in lines:
            if re.match(r'^(Contents|VOLUME|CHAPTER|List of Illustrations)', line.strip(), re.IGNORECASE):
                skip_section = True
            elif line.strip() and not re.match(r'^\d+\.?\s', line.strip()):
                skip_section = False

            if not skip_section and not re.match(r'^(VOLUME|CHAPTER)', line.strip(), re.IGNORECASE):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def remove_urls(text):
        """Remove URLs from the text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)


class ExportDatasetToJson(object):
    ''' Outputs datasets to a nested format.
    '''

    def __init__(self):
        self.cleaner = GutenbergCleaner()

    def export(self,
               metadata_file: str,
               data_folder: str,
               output_file: str,
               seed: int = 42,
               nested: bool = False
               ):

        full_books_list = []

        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Dictionary to track the sequence number for each unique (author, title) pair.
        unique_label_dict = {}
        
        for book_metadata in tqdm(metadata, desc="Processing books"):
            gutenberg_id = book_metadata["gutenberg_id"]
            author = book_metadata["author"]
            title = book_metadata["title"]
            subject = book_metadata["subject"]
            filename = book_metadata["filename"]
            file_path = os.path.join(data_folder, filename)

            with open(file_path, 'r') as book_file:
                book_text = book_file.read()
                cleaned_text = self.cleaner.clean(book_text)

                # using unstructured to divide the text into paragraphs
                elements = partition_text(text=cleaned_text)

                # debug to check if the text is being split correctly
                # for i, element in enumerate(elements[:30], start=0):
                #     print(f"Element {i}:")
                #     print(element.text)
                #     print("-" * 40)
                
                #group elements into chunks not exceeding 1500 words.
                chunks = chunk_paragraphs(elements, max_words=1500)
                chunks = [clean_extra_whitespace(str(chunk)) for chunk in chunks]
                
                # debug to check if the text is being split correctly
                # for i, chunk in enumerate(chunks[:30], start=0):
                #     print(f"Chunk {i}:")
                #     print(chunk)
                #     print("-" * 40)
                
                # Process the title to remove chapter/part references for labeling.
                processed_title = clean_title(title)
                
                # Build unique key as Author_cleanedTitle.
                book_key = f"{author}_{processed_title}"
                if book_key not in unique_label_dict:
                    unique_label_dict[book_key] = 1
    
                labeled_chunks = []
                for chunk in chunks:
                    # Assign sequential label for this (author, processed_title) pair.
                    chunk_label = f"{author}_[{processed_title}]_{unique_label_dict[book_key]}"
                    unique_label_dict[book_key] += 1
                    labeled_chunks.append({"label": chunk_label, "text": chunk})             
                
                # Save the full book with labeled chunks.
                full_book_entry = {
                    "gutenberg_id": gutenberg_id,
                    "author": author,
                    "title": processed_title,
                    "subject": subject,
                    "chunks": labeled_chunks
                }
                full_books_list.append(full_book_entry)

        # Write the full books to a JSONL file
        with jsonlines.open(output_file, mode='w') as writer:
            for full_book in full_books_list:
                for chunk in full_book["chunks"]:
                    record = {
                        "gutenberg_id": full_book["gutenberg_id"],
                        "author": full_book["author"],
                        "title": full_book["title"],
                        "subject": full_book["subject"],
                        "chunk_label": chunk["label"],
                        "chunk_text": chunk["text"]
                    }
                    writer.write(record)


if __name__ == '__main__':
    # fire.Fire(ExportDatasetToJson)

    # Debugging
    defaults = {
        "metadata_file": "../books/metadata.json",
        "data_folder": "../books",
        "output_file": "../data/clean_dataset_all.jsonl",
        "seed": 42,
        "nested": False,
    }
    ExportDatasetToJson().export(**defaults)
