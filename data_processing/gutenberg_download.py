import os
import re
import fire
import sqlite3
from pathlib import Path
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.gutenbergcache import GutenbergCacheSettings
import gutenbergpy.textget as g
from tqdm import tqdm
import json


class DownloadGutenberg:
    def download(self, export_dir, subjects_to_include=None):
        """
        Download books from Project Gutenberg based on specified subjects.

        Args:
            export_dir (str): Directory to save downloaded books.
            cache_dir (str): Directory to store the Gutenberg cache.
            subjects_to_include (list): List of subjects to include (optional).
        """
        # Create export directory if it doesn't exist
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        if subjects_to_include is None:
            subjects_to_include = [
                "Fugitive slaves -- Fiction",
                "United States -- Social life and customs -- 19th century -- Fiction",
                "Americans -- Great Britain -- Fiction",
            ]

        GutenbergCache.create(
            refresh=True,
            download=True,
            unpack=True,
            parse=True,
            cache=True,
            deleteTemp=True,
        )

        cache = GutenbergCache.get_cache()
        query = """
        WITH mark_twain_books AS (
            SELECT DISTINCT b.id, s.name AS subject
            FROM books b
            JOIN book_authors ba ON b.id = ba.bookid
            JOIN authors a ON ba.authorid = a.id
            JOIN book_subjects bs ON b.id = bs.bookid
            JOIN subjects s ON bs.subjectid = s.id
            WHERE a.name = 'Twain, Mark'
        )

        SELECT DISTINCT b.gutenbergbookid AS gutenberg_id, t.name AS title, a.name AS author, s.name AS subject
        FROM books b
        JOIN book_authors ba ON b.id = ba.bookid
        JOIN authors a ON ba.authorid = a.id
        JOIN book_subjects bs ON b.id = bs.bookid
        JOIN subjects s ON bs.subjectid = s.id
        JOIN titles t ON b.id = t.bookid
        JOIN languages l ON b.languageid = l.id
        WHERE l.name = 'en'  -- Only English books
        AND (
            s.name IN (SELECT subject FROM mark_twain_books)  -- Same genre
            OR b.id IN (SELECT id FROM mark_twain_books)      -- Include Mark Twain's own books
        )
        ; 
        """

        connection = sqlite3.connect(GutenbergCacheSettings.CACHE_FILENAME)
        cursor = connection.cursor()

        # # Execute the query to get all unique subjects
        # cursor.execute("SELECT DISTINCT name FROM subjects;")
        # subjects = cursor.fetchall()

        # # Display the results
        # all_subjects = [subject[0] for subject in subjects]  # Convert to a list of subject names
        # print("All unique subjects in the database:")
        # for subject in all_subjects:
        #     print(subject)

        # #save the subjects to a file
        # with open("all_subjects.txt", "w") as f:
        #     for subject in all_subjects:
        #         f.write(subject + "\n")

        books = cursor.execute(query).fetchall()
        books = {
            gutenberg_id: {
                "gutenberg_id": gutenberg_id,
                "title": title,
                "author": author,
                "subject": subject,
            }
            for (gutenberg_id, title, author, subject) in books
        }
        # Compile a regex pattern that matches any of the subjects in `subjects_to_include`
        pattern = re.compile(
            r"\b(" + "|".join(subjects_to_include) + r")\b", re.IGNORECASE
        )

        filtered_books = {
            gutenberg_id: book_info
            for gutenberg_id, book_info in books.items()
            if pattern.search(book_info["subject"])
        }

        for gutenberg_id in filtered_books:
            # if filtered_books[gutenberg_id]["author"] == "Twain, Mark":
            print(filtered_books[gutenberg_id])
        print(len(filtered_books))

        # Prepare metadata storage
        metadata = []

        # The code snippet you provided is a part of a Python script that downloads books from Project
        # Gutenberg based on specified subjects. Here's what the specific portion of the code does:
        # Download books
        for gutenberg_id in tqdm(filtered_books, desc="Downloading books"):
            try:
                text = g.get_text_by_id(gutenberg_id)
                clean_text = g.strip_headers(text)
                if isinstance(clean_text, bytes):
                    clean_text = clean_text.decode("utf-8")

                author = filtered_books[gutenberg_id]["author"]
                title = filtered_books[gutenberg_id]["title"]
                # Use gutenberg_id for filename
                filename = f"book_{gutenberg_id}.txt"
                file_path = Path(export_dir) / filename

                # Append metadata for this book
                metadata.append(
                    {
                        "gutenberg_id": filtered_books[gutenberg_id]["gutenberg_id"],
                        "author": author,
                        "title": title,
                        "subject": filtered_books[gutenberg_id]["subject"],
                        "filename": filename,  # Reference to the saved text file
                    }
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
            except Exception as e:
                print(
                    f"Failed to download book {filtered_books[gutenberg_id]} by {filtered_books[gutenberg_id]['author']}: {e}"
                )

        metadata_file = Path(export_dir) / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(DownloadGutenberg)
