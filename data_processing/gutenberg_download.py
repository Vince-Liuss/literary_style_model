import os
import re
import ssl
import sqlite3
import urllib3
from pathlib import Path
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.gutenbergcache import GutenbergCacheSettings
import gutenbergpy.textget as g
from tqdm import tqdm
import json
from collections import defaultdict

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DownloadGutenberg:
    def __init__(self):
        """Initialize the Gutenberg downloader."""
        self.cache_initialized = False

        # Core configuration - change these settings in one place
        self.target_books = [
            ("Adventures of Huckleberry Finn", "Twain, Mark"),
            ("A Tale of Two Cities", "Dickens, Charles"),
            ("Pride and Prejudice", "Austen, Jane"),
            ("Tess of the d'Urbervilles: A Pure Woman", "Hardy, Thomas"),
        ]

        self.core_target_subjects = [
            "Adventure stories",
            "Historical fiction",
            "Young women -- Fiction",
            "Man-woman relationships -- Fiction",
        ]

        # Author name aliases (alternative names for the same author)
        self.author_aliases = {
            "Twain, Mark": ["Clemens, Samuel Langhorne", "Samuel Langhorne Clemens"],
            "Austen, Jane": ["Austen, Miss"],
            "Dickens, Charles": ["Dickens, Charles John Huffam", "Boz"],
            "Hardy, Thomas": ["Hardy, Thomas"],
        }

    def initialize_cache(self):
        """Initialize the Gutenberg cache if not already done."""
        if not self.cache_initialized:
            print("Initializing Gutenberg cache...")
            GutenbergCache.create(
                refresh=True,
                download=True,
                unpack=True,
                parse=True,
                cache=True,
                deleteTemp=True,
            )
            self.cache_initialized = True
            print("Cache initialization complete.")

    def is_target_author(self, author_name):
        """Check if an author name matches any target book author or their aliases."""
        for target_author, aliases in self.author_aliases.items():
            if author_name == target_author or author_name in aliases:
                return target_author
        return None

    def validate_and_cleanup_metadata(
        self, metadata_file, data_folder, output_file=None
    ):
        """
        Combined validation and cleanup of metadata:
        1. First validates metadata against Gutenberg database and downloaded files
        2. Corrects any metadata mismatches
        3. Then applies advanced cleanup logic (2 books per author per subject, 1 subject per book)
        4. Saves corrected and cleaned metadata

        Returns the output file path.
        """
        self.initialize_cache()

        print("\n" + "=" * 80)
        print("VALIDATING AND CLEANING METADATA")
        print("=" * 80)

        # Load metadata
        if not Path(metadata_file).exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return None

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        books_data = data.get("books", data)
        print(f"\nLoaded {len(books_data)} books from metadata file")

        # PHASE 1: VALIDATE METADATA AGAINST DATABASE
        print("\n" + "=" * 80)
        print("PHASE 1: VALIDATING METADATA AGAINST GUTENBERG DATABASE")
        print("=" * 80)

        connection = sqlite3.connect(GutenbergCacheSettings.CACHE_FILENAME)
        cursor = connection.cursor()

        validated_books = []
        validation_issues = []
        files_checked = 0
        files_missing = 0
        metadata_mismatches = 0

        for book_meta in tqdm(books_data, desc="Validating metadata"):
            gutenberg_id = book_meta["gutenberg_id"]
            filename = book_meta.get("filename", f"book_{gutenberg_id}.txt")
            file_path = Path(data_folder) / filename

            # Check if file exists
            if not file_path.exists():
                files_missing += 1
                validation_issues.append(
                    {
                        "type": "missing_file",
                        "gutenberg_id": gutenberg_id,
                        "filename": filename,
                        "stored_title": book_meta.get("title"),
                        "stored_author": book_meta.get("author"),
                    }
                )
                continue

            files_checked += 1

            # Query Gutenberg database for correct book info
            query = """
            SELECT DISTINCT
                b.gutenbergbookid,
                t.name AS title,
                a.name AS author,
                GROUP_CONCAT(s.name, '|') AS subjects
            FROM books b
            JOIN titles t ON b.id = t.bookid
            JOIN book_authors ba ON b.id = ba.bookid
            JOIN authors a ON ba.authorid = a.id
            LEFT JOIN book_subjects bs ON b.id = bs.bookid
            LEFT JOIN subjects s ON bs.subjectid = s.id
            WHERE b.gutenbergbookid = ?
            GROUP BY b.gutenbergbookid
            """

            results = cursor.execute(query, (gutenberg_id,)).fetchall()

            if not results:
                validation_issues.append(
                    {
                        "type": "not_found_in_db",
                        "gutenberg_id": gutenberg_id,
                        "filename": filename,
                        "stored_title": book_meta.get("title"),
                        "stored_author": book_meta.get("author"),
                    }
                )
                continue

            db_id, db_title, db_author, db_subjects_str = results[0]
            db_subjects = (
                [s for s in db_subjects_str.split("|") if s] if db_subjects_str else []
            )

            # Check if metadata matches database
            stored_title = book_meta.get("title", "")
            stored_author = book_meta.get("author", "")

            title_match = stored_title.lower() == db_title.lower()
            author_match = stored_author == db_author or self.is_target_author(
                db_author
            ) == self.is_target_author(stored_author)

            if not title_match or not author_match:
                metadata_mismatches += 1
                validation_issues.append(
                    {
                        "type": "metadata_mismatch",
                        "gutenberg_id": gutenberg_id,
                        "filename": filename,
                        "stored_title": stored_title,
                        "stored_author": stored_author,
                        "correct_title": db_title,
                        "correct_author": db_author,
                        "correct_subjects": db_subjects,
                    }
                )

                # Correct the metadata
                book_copy = book_meta.copy()
                book_copy["title"] = db_title
                book_copy["author"] = db_author
                book_copy["subjects"] = db_subjects
                validated_books.append(book_copy)
            else:
                # Metadata is correct
                book_copy = book_meta.copy()
                if not book_copy.get("subjects"):
                    book_copy["subjects"] = db_subjects
                validated_books.append(book_copy)

        connection.close()

        # Print validation report
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        print(f"\nFiles checked: {files_checked}")
        print(f"Files missing: {files_missing}")
        print(f"Metadata mismatches corrected: {metadata_mismatches}")

        if validation_issues:
            print(f"\n⚠️  VALIDATION ISSUES FOUND ({len(validation_issues)}):")
            print("-" * 80)

            # Group issues by type
            issues_by_type = defaultdict(list)
            for issue in validation_issues:
                issues_by_type[issue["type"]].append(issue)

            for issue_type, issues in issues_by_type.items():
                print(f"\n{issue_type.upper()} ({len(issues)}):")

                if issue_type == "missing_file":
                    for issue in issues:
                        print(f"  - ID {issue['gutenberg_id']}: {issue['filename']}")
                        print(f"    Title: {issue['stored_title']}")
                        print(f"    Author: {issue['stored_author']}")

                elif issue_type == "not_found_in_db":
                    for issue in issues:
                        print(
                            f"  - ID {issue['gutenberg_id']}: NOT FOUND IN GUTENBERG DATABASE"
                        )
                        print(f"    Title: {issue['stored_title']}")
                        print(f"    Author: {issue['stored_author']}")

                elif issue_type == "metadata_mismatch":
                    for issue in issues:
                        print(f"  - ID {issue['gutenberg_id']}: {issue['filename']}")
                        print(
                            f"    Stored:  {issue['stored_title']} by {issue['stored_author']}"
                        )
                        print(
                            f"    Correct: {issue['correct_title']} by {issue['correct_author']}"
                        )
                        if issue["correct_subjects"]:
                            print(
                                f"    Subjects: {', '.join(issue['correct_subjects'][:3])}..."
                            )
        else:
            print(f"\n✅ All metadata is valid and matches Gutenberg database!")

        # PHASE 2: APPLY ADVANCED CLEANUP
        print("\n" + "=" * 80)
        print("PHASE 2: APPLYING ADVANCED CLEANUP")
        print("=" * 80)

        cleaned_metadata = self.apply_advanced_cleanup(validated_books, data_folder)

        # Show target books status
        print("\n" + "=" * 80)
        print("TARGET BOOKS STATUS:")
        print("=" * 80)

        target_book_titles = {title for title, author in self.target_books}
        found_target_books = [
            book for book in cleaned_metadata if book["title"] in target_book_titles
        ]
        missing_target_books = [
            (title, author)
            for title, author in self.target_books
            if title not in {book["title"] for book in cleaned_metadata}
        ]

        if found_target_books:
            print(f"\n✅ TARGET BOOKS FOUND ({len(found_target_books)}):")
            for book in sorted(found_target_books, key=lambda x: x["title"]):
                print(
                    f"  - {book['title']} by {book['author']} (ID: {book['gutenberg_id']})"
                )
                print(f"    File: {book.get('filename')}")
                if book.get("subjects"):
                    print(f"    Subjects: {', '.join(book['subjects'])}")

        if missing_target_books:
            print(f"\n❌ TARGET BOOKS MISSING ({len(missing_target_books)}):")
            for title, author in missing_target_books:
                print(f"  - {title} by {author}")
        else:
            print(f"\n✅ ALL TARGET BOOKS PRESENT")

        # Save corrected and cleaned metadata
        if output_file is None:
            output_file = Path(data_folder).parent / "cleaned_metadata.json"

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "validation_and_cleanup_summary": {
                        "total_books": len(cleaned_metadata),
                        "files_checked": files_checked,
                        "files_missing": files_missing,
                        "metadata_mismatches_corrected": metadata_mismatches,
                        "subjects_used": self.core_target_subjects,
                        "validation_applied": True,
                        "cleanup_applied": True,
                    },
                    "books": cleaned_metadata,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(f"\n✅ Validation and cleanup completed successfully!")
        print(f"📁 Corrected and cleaned metadata saved to: {output_file}")

        # Show final statistics
        self.show_final_statistics(cleaned_metadata)

        return str(output_file)

    def show_target_book_subjects(self):
        """Show all subjects associated with the target books."""
        self.initialize_cache()

        # Build WHERE clause for target books with proper escaping and aliases
        book_conditions = []
        for title, author in self.target_books:
            escaped_title = title.replace("'", "''")
            escaped_author = author.replace("'", "''")
            book_conditions.append(
                f"(t.name = '{escaped_title}' AND a.name = '{escaped_author}')"
            )

            # Add aliases
            for alias in self.author_aliases.get(author, []):
                escaped_alias = alias.replace("'", "''")
                book_conditions.append(
                    f"(t.name = '{escaped_title}' AND a.name = '{escaped_alias}')"
                )

        where_clause = " OR ".join(book_conditions)

        query = f"""
        SELECT DISTINCT 
            t.name AS title,
            a.name AS author,
            s.name AS subject
        FROM books b
        JOIN titles t ON b.id = t.bookid
        JOIN book_authors ba ON b.id = ba.bookid
        JOIN authors a ON ba.authorid = a.id
        JOIN book_subjects bs ON b.id = bs.bookid
        JOIN subjects s ON bs.subjectid = s.id
        WHERE
            {where_clause}
        ORDER BY title, subject;
        """

        connection = sqlite3.connect(GutenbergCacheSettings.CACHE_FILENAME)
        cursor = connection.cursor()
        results = cursor.execute(query).fetchall()
        connection.close()

        # Group by book
        books_subjects = {}
        for title, author, subject in results:
            book_key = f"{title} by {author}"
            if book_key not in books_subjects:
                books_subjects[book_key] = []
            books_subjects[book_key].append(subject)

        print("SUBJECTS FOR EACH TARGET BOOK:")
        print("=" * 80)

        all_subjects = set()
        for book, subjects in books_subjects.items():
            print(f"\n{book}:")
            for i, subject in enumerate(sorted(subjects), 1):
                print(f"  {i:2d}. {subject}")
                all_subjects.add(subject)

        print("\n" + "=" * 80)
        print("ALL UNIQUE SUBJECTS ACROSS ALL BOOKS:")
        print("=" * 80)

        for i, subject in enumerate(sorted(all_subjects), 1):
            print(f"{i:2d}. {subject}")

        print(f"\nTotal unique subjects: {len(all_subjects)}")
        return sorted(all_subjects)

    def get_books_from_subjects(self, target_subjects):
        """Get books that match the specified subjects and include target books."""
        self.initialize_cache()

        # Build WHERE clause for target books with proper escaping and aliases
        book_conditions = []
        for title, author in self.target_books:
            escaped_title = title.replace("'", "''")
            escaped_author = author.replace("'", "''")
            book_conditions.append(
                f"(t.name = '{escaped_title}' AND a.name = '{escaped_author}')"
            )

            # Add aliases
            for alias in self.author_aliases.get(author, []):
                escaped_alias = alias.replace("'", "''")
                book_conditions.append(
                    f"(t.name = '{escaped_title}' AND a.name = '{escaped_alias}')"
                )

        target_books_where = " OR ".join(book_conditions)

        # Build subject conditions with proper escaping
        subject_conditions = []
        for subject in target_subjects:
            escaped_subject = subject.replace("'", "''")
            subject_conditions.append(f"'{escaped_subject}'")
        subjects_in_clause = "(" + ", ".join(subject_conditions) + ")"

        query = f"""
        WITH target_book_ids AS (
            SELECT DISTINCT b.id AS book_id, b.gutenbergbookid
            FROM books b
            JOIN titles t ON b.id = t.bookid
            JOIN book_authors ba ON b.id = ba.bookid
            JOIN authors a ON ba.authorid = a.id
            JOIN languages l ON b.languageid = l.id
            WHERE
                {target_books_where}
                AND l.name = 'en'
                AND b.gutenbergbookid IS NOT NULL
        ),
        books_with_text AS (
            SELECT DISTINCT b.id
            FROM books b
            JOIN downloadlinks dl ON b.id = dl.bookid
            JOIN downloadlinkstype dlt ON dl.downloadtypeid = dlt.id
            WHERE dlt.name LIKE '%text%' 
               OR dlt.name LIKE '%plain%'
               OR dlt.name LIKE '%utf-8%'
               OR dlt.name LIKE '%txt%'
               OR dlt.name LIKE '%epub%'
               OR dlt.name LIKE '%html%'
        ),
        author_subject_book_counts AS (
            -- Count books per author per subject
            SELECT 
                ba.authorid,
                a.name AS author_name,
                s.name AS subject_name,
                COUNT(DISTINCT b.id) AS book_count
            FROM books b
            JOIN book_authors ba ON b.id = ba.bookid
            JOIN authors a ON ba.authorid = a.id
            JOIN book_subjects bs ON b.id = bs.bookid
            JOIN subjects s ON bs.subjectid = s.id
            JOIN languages l ON b.languageid = l.id
            WHERE s.name IN {subjects_in_clause}
            AND l.name = 'en'
            AND b.gutenbergbookid IS NOT NULL
            AND b.id NOT IN (SELECT book_id FROM target_book_ids)
            AND b.id IN (SELECT id FROM books_with_text)
            GROUP BY ba.authorid, a.name, s.name
            HAVING COUNT(DISTINCT b.id) >= 2  -- At least 2 books per subject
        ),
        ranked_books_per_author_subject AS (
            -- Rank books within each author-subject combination
            SELECT 
                b.id,
                b.gutenbergbookid,
                ba.authorid,
                s.name AS subject_name,
                ROW_NUMBER() OVER(
                    PARTITION BY ba.authorid, s.name
                    ORDER BY b.numdownloads DESC, b.gutenbergbookid ASC
                ) as rank_in_author_subject
            FROM books b
            JOIN book_authors ba ON b.id = ba.bookid
            JOIN book_subjects bs ON b.id = bs.bookid
            JOIN subjects s ON bs.subjectid = s.id
            JOIN languages l ON b.languageid = l.id
            WHERE (ba.authorid, s.name) IN (
                SELECT authorid, subject_name FROM author_subject_book_counts
            )
            AND s.name IN {subjects_in_clause}
            AND l.name = 'en'
            AND b.gutenbergbookid IS NOT NULL
            AND b.id NOT IN (SELECT book_id FROM target_book_ids)
            AND b.id IN (SELECT id FROM books_with_text)
        )

        SELECT DISTINCT
            b.gutenbergbookid AS gutenberg_id,
            t.name AS title,
            a.name AS author,
            s.name AS subject
        FROM books b
        JOIN book_authors ba ON b.id = ba.bookid
        JOIN authors a ON ba.authorid = a.id
        JOIN book_subjects bs ON b.id = bs.bookid
        JOIN subjects s ON bs.subjectid = s.id
        JOIN titles t ON b.id = t.bookid
        JOIN languages l ON b.languageid = l.id
        WHERE
            l.name = 'en'
            AND b.gutenbergbookid IS NOT NULL
            AND b.id IN (SELECT id FROM books_with_text)
            AND (
                -- Include target books
                b.gutenbergbookid IN (SELECT gutenbergbookid FROM target_book_ids)
                OR
                -- Include exactly 2 books per author per subject
                (b.id IN (
                    SELECT id 
                    FROM ranked_books_per_author_subject 
                    WHERE rank_in_author_subject <= 2
                ))
            )
        ORDER BY
            author, title;
        """

        connection = sqlite3.connect(GutenbergCacheSettings.CACHE_FILENAME)
        cursor = connection.cursor()
        books_data = cursor.execute(query).fetchall()
        connection.close()

        # Convert to dictionary format
        books = {}
        for gutenberg_id, title, author, subject in books_data:
            if gutenberg_id in books:
                if subject not in books[gutenberg_id]["subjects"]:
                    books[gutenberg_id]["subjects"].append(subject)
            else:
                books[gutenberg_id] = {
                    "gutenberg_id": gutenberg_id,
                    "title": title,
                    "author": author,
                    "subjects": [subject],
                }

        return list(books.values())

    def check_books_count(self, target_subjects):
        """Check and display how many books will be downloaded."""
        print("Fetching books from database...")
        books = self.get_books_from_subjects(target_subjects)

        print(f"\nFound {len(books)} books to download:")
        print("-" * 80)

        target_book_titles = {title for title, author in self.target_books}
        target_books_found = []
        other_books = []

        for book_info in books:
            is_target = book_info["title"] in target_book_titles
            if is_target:
                target_books_found.append(book_info)
            else:
                other_books.append(book_info)

        print(f"\nTARGET BOOKS ({len(target_books_found)}):")
        print("-" * 40)
        for book_info in target_books_found:
            print(
                f"  - {book_info['title']} by {book_info['author']} (ID: {book_info['gutenberg_id']})"
            )
            print(f"    Subjects: {', '.join(book_info['subjects'])}")

        print(f"\nOTHER BOOKS WITH MATCHING SUBJECTS ({len(other_books)}):")
        print("-" * 40)

        # Group other books by author
        authors = {}
        for book_info in other_books:
            author = book_info["author"]
            if author not in authors:
                authors[author] = []
            authors[author].append(book_info)

        for author, author_books in sorted(authors.items()):
            print(f"\n{author} ({len(author_books)} books):")
            for book in author_books:
                print(f"  - {book['title']} (ID: {book['gutenberg_id']})")
                print(f"    Subjects: {', '.join(book['subjects'])}")

        print("-" * 80)
        print(
            f"Total books: {len(books)} (Target: {len(target_books_found)}, Others: {len(other_books)})"
        )

        return books

    def download_books(self, export_dir, target_subjects):
        """Download the books to the specified directory with integrated cleanup."""
        Path(export_dir).mkdir(parents=True, exist_ok=True)

        # Delete old metadata file if it exists
        metadata_file = Path(export_dir) / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
            print(f"Deleted old metadata file: {metadata_file}")

        # Delete all existing book files
        existing_books = list(Path(export_dir).glob("book_*.txt"))
        if existing_books:
            for book_file in existing_books:
                book_file.unlink()
            print(f"Deleted {len(existing_books)} existing book files")

        books = self.get_books_from_subjects(target_subjects)
        print(f"\nStarting fresh download of {len(books)} books to {export_dir}...")

        metadata = []
        successful_downloads = 0
        failed_downloads = 0

        for book_info in tqdm(books, desc="Downloading books"):
            gutenberg_id = book_info["gutenberg_id"]
            filename = f"book_{gutenberg_id}.txt"
            file_path = Path(export_dir) / filename

            try:
                text = g.get_text_by_id(gutenberg_id)
                if not text:
                    raise ValueError("Downloaded text is empty")

                clean_text = g.strip_headers(text)
                if isinstance(clean_text, bytes):
                    clean_text = clean_text.decode("utf-8", errors="ignore")

                if not clean_text or len(clean_text.strip()) < 1000:
                    raise ValueError("Cleaned text is too short")

                # Save the text file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)

                metadata.append(
                    {
                        "gutenberg_id": book_info["gutenberg_id"],
                        "author": book_info["author"],
                        "title": book_info["title"],
                        "subjects": book_info["subjects"],
                        "filename": filename,
                        "file_size": len(clean_text),
                    }
                )
                successful_downloads += 1

            except Exception as e:
                failed_downloads += 1
                continue

        print(f"\nSuccessfully downloaded {successful_downloads} books.")
        if failed_downloads > 0:
            print(f"Failed to download {failed_downloads} books.")

        # Apply advanced cleanup logic (2 books per author per subject, 1 subject per book)
        cleaned_metadata = self.apply_advanced_cleanup(metadata, export_dir)

        # Write final metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "download_summary": {
                        "total_attempted": len(books),
                        "successful_downloads": successful_downloads,
                        "failed_downloads": failed_downloads,
                        "books_after_cleanup": len(cleaned_metadata),
                        "subjects_used": target_subjects,
                        "cleanup_applied": True,
                    },
                    "books": cleaned_metadata,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(f"\nDownload and cleanup complete!")
        print(f"Final book count: {len(cleaned_metadata)}")
        print(f"Metadata saved to: {metadata_file}")

        # Show final statistics
        self.show_final_statistics(cleaned_metadata)

        return cleaned_metadata

    def apply_advanced_cleanup(self, metadata, export_dir):
        """
        Apply advanced cleanup logic:
        1. Each author has exactly 2 books per subject they appear in.
        2. Each book is assigned to ONLY ONE subject.
        3. Files for removed books are deleted from disk.
        4. Deduplicate target books (keep only one edition).
        """
        print("\n" + "=" * 80)
        print("APPLYING ADVANCED CLEANUP - 2 BOOKS PER AUTHOR, 1 SUBJECT PER BOOK")
        print("=" * 80)

        target_book_titles = {title for title, author in self.target_books}
        target_books_meta = [
            book for book in metadata if book["title"] in target_book_titles
        ]
        other_books_meta = [
            book for book in metadata if book["title"] not in target_book_titles
        ]

        # NEW: Deduplicate target books - keep only one per title (highest ID)
        print("\nDeduplicating target books...")
        deduplicated_target_books = {}
        for book in target_books_meta:
            title = book["title"]
            if title not in deduplicated_target_books:
                deduplicated_target_books[title] = book
            else:
                # Keep the book with higher ID (newer/better version)
                if (
                    book["gutenberg_id"]
                    > deduplicated_target_books[title]["gutenberg_id"]
                ):
                    # Delete the old file
                    old_file = (
                        Path(export_dir) / deduplicated_target_books[title]["filename"]
                    )
                    if old_file.exists():
                        old_file.unlink()
                        print(
                            f"  Removed duplicate: {deduplicated_target_books[title]['title']} (ID: {deduplicated_target_books[title]['gutenberg_id']})"
                        )
                    deduplicated_target_books[title] = book
                else:
                    # Delete current file
                    curr_file = Path(export_dir) / book["filename"]
                    if curr_file.exists():
                        curr_file.unlink()
                        print(
                            f"  Removed duplicate: {book['title']} (ID: {book['gutenberg_id']})"
                        )

        target_books_meta = list(deduplicated_target_books.values())

        # Filter target books to only keep core subjects
        for book in target_books_meta:
            if book.get("subjects"):
                # Only keep subjects that are in core_target_subjects
                filtered_subjects = [
                    s for s in book["subjects"] if s in self.core_target_subjects
                ]
                book["subjects"] = (
                    filtered_subjects if filtered_subjects else book["subjects"]
                )

        print(f"  Kept {len(target_books_meta)} target books after deduplication")

        all_books_by_id = {book["gutenberg_id"]: book for book in metadata}

        # Step 1: Group all books by subject and author to find potential pairs
        subject_author_books = defaultdict(lambda: defaultdict(list))
        for book in other_books_meta:
            for subject in book["subjects"]:
                if subject in self.core_target_subjects:  # Only consider core subjects
                    author = book["author"]
                    subject_author_books[subject][author].append(book)

        # Step 2: Identify all books that are part of at least one valid pair
        potential_subject_book_map = defaultdict(list)
        for subject, authors_data in subject_author_books.items():
            for author, books in authors_data.items():
                if len(books) >= 2:
                    sorted_books = sorted(books, key=lambda x: x.get("gutenberg_id", 0))
                    top_two_books = sorted_books[:2]
                    for book in top_two_books:
                        # Only add subjects that are in core_target_subjects
                        if subject in self.core_target_subjects:
                            potential_subject_book_map[book["gutenberg_id"]].append(
                                subject
                            )

        # Step 3: Assign each book to exactly ONE subject (resolve conflicts)
        final_book_subject_map = {}
        for book_id in sorted(potential_subject_book_map.keys()):
            candidate_subjects = sorted(potential_subject_book_map[book_id])
            chosen_subject = candidate_subjects[0]  # First subject alphabetically wins
            final_book_subject_map[book_id] = chosen_subject

        # Step 4: Re-validate pairs after conflict resolution
        final_structure = defaultdict(lambda: defaultdict(list))
        for book_id, subject in final_book_subject_map.items():
            book = all_books_by_id[book_id]
            author = book["author"]
            final_structure[subject][author].append(book)

        final_book_ids_to_keep = set()
        for subject, authors_data in final_structure.items():
            print(f"\nValidating pairs for {subject}:")
            for author, books in authors_data.items():
                if len(books) == 2:
                    print(f"  ✓ Kept valid pair for {author}")
                    for book in books:
                        final_book_ids_to_keep.add(book["gutenberg_id"])
                else:
                    print(
                        f"  ✗ Pair for {author} invalidated by conflict resolution ({len(books)} book remaining)."
                    )

        # Step 5: Delete files for all "other" books that are NOT being kept
        files_deleted_count = 0
        for book in other_books_meta:
            if book["gutenberg_id"] not in final_book_ids_to_keep:
                file_path = Path(export_dir) / book["filename"]
                if file_path.exists():
                    file_path.unlink()
                    files_deleted_count += 1

        if files_deleted_count > 0:
            print(
                f"\nDeleted {files_deleted_count} book files that did not form a valid final pair."
            )

        # Step 6: Build the final, clean metadata list
        final_cleaned_other_books = []
        for book_id in sorted(list(final_book_ids_to_keep)):
            book = all_books_by_id[book_id]
            chosen_subject = final_book_subject_map[book_id]

            book_copy = book.copy()
            book_copy["subjects"] = [
                chosen_subject
            ]  # Assign the single chosen subject (already filtered to core)
            final_cleaned_other_books.append(book_copy)

        final_metadata = target_books_meta + final_cleaned_other_books
        print(f"\nKept {len(final_cleaned_other_books)} 'other' books after cleanup.")
        print(f"Total books in final metadata: {len(final_metadata)}")

        return final_metadata

    def show_final_statistics(self, metadata):
        """Show comprehensive statistics about the final cleaned dataset."""
        target_book_titles = {title for title, author in self.target_books}
        target_books = [
            book for book in metadata if book["title"] in target_book_titles
        ]
        other_books = [
            book for book in metadata if book["title"] not in target_book_titles
        ]

        print("\n" + "=" * 80)
        print("FINAL DATASET STATISTICS")
        print("=" * 80)

        # Analyze by subject - each book should have exactly one subject now
        subject_stats = {}
        for subject in self.core_target_subjects:
            subject_stats[subject] = {
                "authors": defaultdict(list),
                "books": [],
                "total_books": 0,
                "total_authors": 0,
                "valid_authors": 0,
            }

        # Group cleaned books by subject and author
        for book in other_books:
            for subject in book["subjects"]:
                if subject in self.core_target_subjects:
                    author = book["author"]
                    subject_stats[subject]["authors"][author].append(book)
                    subject_stats[subject]["books"].append(book)

        # Calculate statistics
        for subject, stats in subject_stats.items():
            stats["total_books"] = len(stats["books"])
            stats["total_authors"] = len(stats["authors"])
            for author, books in stats["authors"].items():
                if len(books) == 2:
                    stats["valid_authors"] += 1

        print("\nFINAL SUBJECT-SPECIFIC BREAKDOWN:")
        for subject in sorted(subject_stats.keys()):
            stats = subject_stats[subject]
            print(f"  {subject}:")
            print(
                f"    Books: {stats['total_books']}, Authors: {stats['total_authors']}, Valid authors: {stats['valid_authors']}"
            )

            # Verify all numbers are consistent
            if stats["total_books"] != stats["valid_authors"] * 2:
                print(
                    f"    ⚠️  WARNING: Book count ({stats['total_books']}) != Valid authors ({stats['valid_authors']}) * 2"
                )

        # Check for any authors appearing in multiple subjects (should be none after cleanup)
        all_authors = defaultdict(set)
        for book in other_books:
            author = book["author"]
            for subject in book["subjects"]:
                if subject in self.core_target_subjects:
                    all_authors[author].add(subject)

        multi_subject_authors = {
            author: subjects
            for author, subjects in all_authors.items()
            if len(subjects) > 1
        }
        if multi_subject_authors:
            print(
                f"\nAUTHORS APPEARING IN MULTIPLE SUBJECTS ({len(multi_subject_authors)}):"
            )
            for author, subjects in sorted(multi_subject_authors.items()):
                print(f"  - {author}: {', '.join(sorted(subjects))}")
        else:
            print(
                f"\n✅ No authors appear in multiple subjects (as expected after cleanup)"
            )

        # Check for missing target books
        print("\n" + "=" * 80)
        print("TARGET BOOKS STATUS:")
        print("=" * 80)

        found_target_books = {book["title"] for book in target_books}
        missing_target_books = []

        for title, author in self.target_books:
            if title not in found_target_books:
                missing_target_books.append((title, author))

        if target_books:
            print(f"\n✅ TARGET BOOKS FOUND ({len(target_books)}):")
            for book in sorted(target_books, key=lambda x: x["title"]):
                print(
                    f"  - {book['title']} by {book['author']} (ID: {book['gutenberg_id']})"
                )
                print(f"    Subjects: {', '.join(book['subjects'])}")
        else:
            print(f"\n⚠️  NO TARGET BOOKS FOUND IN METADATA")

        if missing_target_books:
            print(f"\n❌ TARGET BOOKS MISSING ({len(missing_target_books)}):")
            for title, author in sorted(missing_target_books):
                print(f"  - {title} by {author}")
                # Check if author aliases exist
                if author in self.author_aliases:
                    print(
                        f"    Alternative author names: {', '.join(self.author_aliases[author])}"
                    )
        else:
            print(f"\n✅ ALL TARGET BOOKS PRESENT")

        print(f"\nFINAL SUMMARY:")
        print(f"Target books: {len(target_books)}/{len(self.target_books)}")
        print(f"Other books: {len(other_books)} (each book appears exactly once)")
        print(f"Total books: {len(metadata)}")

        return subject_stats

    def direct_download_with_subjects(self, export_dir, subjects):
        """Directly download books using specific subjects."""
        print(f"Downloading books with subjects: {subjects}")
        print(f"Export directory: {export_dir}")

        books = self.check_books_count(subjects)

        confirm = (
            input(f"\nProceed with downloading {len(books)} books? (y/n): ")
            .lower()
            .strip()
        )
        if confirm in ["y", "yes"]:
            self.download_books(export_dir, subjects)
        else:
            print("Download cancelled.")

        return books

    def cleanup_metadata(self, metadata_file, data_folder, output_file=None):
        """
        Load metadata from existing metadata.json file and apply advanced cleanup logic.
        No validation - just applies cleanup (2 books per author per subject, 1 subject per book).

        Returns the output file path.
        """
        print("\n" + "=" * 80)
        print("CLEANING UP METADATA FILE")
        print("=" * 80)

        data_path = Path(data_folder)
        if not data_path.exists():
            print(f"❌ Data folder not found: {data_folder}")
            return None

        if not Path(metadata_file).exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return None

        print(f"\nLoading metadata from: {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both raw list and wrapped format, ignore cleanup_applied flag
        books_data = data.get("books", data if isinstance(data, list) else [])
        print(f"Loaded {len(books_data)} books from metadata file")

        # Show metadata before cleanup
        print("\n" + "=" * 80)
        print("CURRENT METADATA (BEFORE CLEANUP):")
        print("=" * 80)

        books_by_author = defaultdict(list)
        for book in books_data:
            books_by_author[book["author"]].append(book)

        for author in sorted(books_by_author.keys()):
            print(f"\n{author}:")
            for book in sorted(books_by_author[author], key=lambda x: x["title"]):
                print(f"  - {book['title']} (ID: {book['gutenberg_id']})")
                if book.get("subjects"):
                    print(f"    Subjects: {', '.join(book['subjects'][:3])}")

        # Show target books status before cleanup
        print("\n" + "=" * 80)
        print("TARGET BOOKS STATUS (BEFORE CLEANUP):")
        print("=" * 80)

        target_book_titles = {title for title, author in self.target_books}
        found_target_books = [
            book for book in books_data if book["title"] in target_book_titles
        ]
        missing_target_books = [
            (title, author)
            for title, author in self.target_books
            if title not in {book["title"] for book in books_data}
        ]

        if found_target_books:
            print(f"\n✅ TARGET BOOKS FOUND ({len(found_target_books)}):")
            for book in sorted(found_target_books, key=lambda x: x["title"]):
                print(
                    f"  - {book['title']} by {book['author']} (ID: {book['gutenberg_id']})"
                )

        if missing_target_books:
            print(f"\n⚠️  TARGET BOOKS MISSING ({len(missing_target_books)}):")
            for title, author in missing_target_books:
                print(f"  - {title} by {author}")
        else:
            print(f"\n✅ ALL TARGET BOOKS PRESENT")

        print("\n" + "=" * 80)
        print("APPLYING ADVANCED CLEANUP")
        print("=" * 80)

        cleaned_metadata = self.apply_advanced_cleanup(books_data, data_folder)

        print("\n" + "=" * 80)
        print("TARGET BOOKS STATUS (AFTER CLEANUP):")
        print("=" * 80)

        found_target_books_after = [
            book for book in cleaned_metadata if book["title"] in target_book_titles
        ]
        missing_target_books_after = [
            (title, author)
            for title, author in self.target_books
            if title not in {book["title"] for book in cleaned_metadata}
        ]

        if found_target_books_after:
            print(f"\n✅ TARGET BOOKS FOUND ({len(found_target_books_after)}):")
            for book in sorted(found_target_books_after, key=lambda x: x["title"]):
                print(
                    f"  - {book['title']} by {book['author']} (ID: {book['gutenberg_id']})"
                )
                print(f"    Subjects: {', '.join(book['subjects'])}")

        if missing_target_books_after:
            print(f"\n❌ TARGET BOOKS MISSING ({len(missing_target_books_after)}):")
            for title, author in missing_target_books_after:
                print(f"  - {title} by {author}")
        else:
            print(f"\n✅ ALL TARGET BOOKS PRESENT")

        if output_file is None:
            output_file = Path(data_folder).parent / "cleaned_metadata.json"

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cleanup_summary": {
                        "total_books_after_cleanup": len(cleaned_metadata),
                        "original_books": len(books_data),
                        "cleanup_applied": True,
                        "subjects_used": self.core_target_subjects,
                    },
                    "books": cleaned_metadata,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(f"\n✅ Cleaned metadata saved to: {output_file}")
        self.show_final_statistics(cleaned_metadata)

        return str(output_file)


def main():
    """Main function to choose what to do."""
    downloader = DownloadGutenberg()

    print("=" * 60)
    print("GUTENBERG BOOK DOWNLOADER & CLEANUP TOOL")
    print("=" * 60)
    print("Target books:")
    for i, (title, author) in enumerate(downloader.target_books, 1):
        print(f"  {i}. {title} by {author}")
    print("\nCore subjects:")
    for i, subject in enumerate(downloader.core_target_subjects, 1):
        print(f"  {i}. {subject}")
    print("=" * 60)

    while True:
        print("\nChoose an option:")
        print("1. Show all subjects from target books")
        print("2. Check book count for specific subjects")
        print("3. Download books for specific subjects")
        print("4. Direct download with core subjects")
        print("5. Clean up metadata file")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            print("\n" + "=" * 60)
            downloader.show_target_book_subjects()

        elif choice == "2":
            print("\nEnter target subjects (comma-separated):")
            subjects_input = input("Subjects: ").strip()
            if subjects_input:
                target_subjects = [s.strip() for s in subjects_input.split(",")]
                print(f"\nUsing subjects: {target_subjects}")
                downloader.check_books_count(target_subjects)
            else:
                print("No subjects entered.")

        elif choice == "3":
            print("\nEnter target subjects (comma-separated):")
            subjects_input = input("Subjects: ").strip()
            if subjects_input:
                target_subjects = [s.strip() for s in subjects_input.split(",")]
                export_dir = input(
                    "Enter export directory (default: ../data/): "
                ).strip()
                if not export_dir:
                    export_dir = "../data/"

                print(f"\nUsing subjects: {target_subjects}")
                print(f"Export directory: {export_dir}")

                books = downloader.check_books_count(target_subjects)
                confirm = (
                    input(f"\nProceed with downloading {len(books)} books? (y/n): ")
                    .lower()
                    .strip()
                )
                if confirm in ["y", "yes"]:
                    downloader.download_books(export_dir, target_subjects)
                else:
                    print("Download cancelled.")
            else:
                print("No subjects entered.")

        elif choice == "4":
            export_dir = input("Enter export directory (default: ../data/): ").strip()
            if not export_dir:
                export_dir = "../data/"

            print(f"\nUsing core subjects: {downloader.core_target_subjects}")
            downloader.direct_download_with_subjects(
                export_dir, downloader.core_target_subjects
            )

        elif choice == "5":
            metadata_file = input(
                "Enter metadata file path (default: ../data/metadata.json): "
            ).strip()
            if not metadata_file:
                metadata_file = "../data/metadata.json"

            data_folder = input("Enter data folder path (default: ../data/): ").strip()
            if not data_folder:
                data_folder = "../data/"

            output_file = input(
                "Enter output file path (default: ../data/cleaned_metadata.json): "
            ).strip()
            if not output_file:
                output_file = None

            confirm = (
                input(
                    f"\nProceed with cleaning up metadata from '{metadata_file}'? (y/n): "
                )
                .lower()
                .strip()
            )
            if confirm in ["y", "yes"]:
                try:
                    downloader.cleanup_metadata(metadata_file, data_folder, output_file)
                except Exception as e:
                    print(f"\n❌ Error during cleanup: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                print("Cleanup cancelled.")

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()
