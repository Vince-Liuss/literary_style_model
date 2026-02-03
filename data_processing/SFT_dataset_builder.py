import os
import json
import random
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List
from textwrap import dedent
from collections import Counter

import httpx
import spacy
import numpy as np
from tqdm.asyncio import tqdm
from datasets import Dataset, load_from_disk
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
import torch
from concurrent.futures import ThreadPoolExecutor


# =========================
# 1) CONFIGURATION (LOCKED)
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("SFT_Distiller")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "openai/gpt-oss-120b"

CHUNKS_PATH = Path("../data/target_books")
TARGET_SPLIT = "1500"

OUTPUT_SFT_DIR = Path("../data/mimicry_sft_dataset")
TEMP_FILE = OUTPUT_SFT_DIR.parent / "mimicry_gold_progress.jsonl"

JUDGE_MODEL_PATH = (
    "VibrantVista/gte-large-en-v1.5-stylejudge"
)
DEVICE = "cuda"

TARGET_TOTAL_SAMPLES = 40000
MAX_WORKERS = 768

STYLE_SCORE_THRESHOLD = 0.4697

ENFORCE_WORD_COUNT = True
MIN_WORDS = 1200
MAX_WORDS = 1500
MAX_OUTPUT_TOKENS = 2200
JUDGE_BATCH_SIZE = 4
API_CONCURRENCY = 768
SCORE_CONCURRENCY = 1


# ======================
# 2) METADATA (REQUIRED)
# ======================
BOOK_CONFIGS = [
    {
        "author": "Twain, Mark",
        "title": "Adventures of Huckleberry Finn",
        "genre": "Adventure stories",
    },
    {
        "author": "Dickens, Charles",
        "title": "A Tale of Two Cities",
        "genre": "Historical fiction",
    },
    {
        "author": "Austen, Jane",
        "title": "Pride and Prejudice",
        "genre": "Man-woman relationships -- Fiction",
    },
    {
        "author": "Hardy, Thomas",
        "title": "Tess of the d'Urbervilles: A Pure Woman",
        "genre": "Young women -- Fiction",
    },
]
BOOK_META: Dict[str, Dict[str, str]] = {b["author"]: b for b in BOOK_CONFIGS}
TARGET_AUTHORS = set(BOOK_META.keys())


# ======================
# 3) JUDGE (BATCHED)
# ======================
nlp = spacy.blank("en")
_ref_emb_cache: Dict[int, Any] = {}


class BatchScoreJudge:
    """
    Accepts many concurrent score() calls.
    Internally batches encode() on GPU and returns only float scores (no GPU tensors escape).
    """

    def __init__(
        self, model_path: str, device: str, batch_size: int = 16, fp16: bool = True
    ):
        dtype = torch.float16 if (fp16 and device.startswith("cuda")) else None
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": dtype} if dtype is not None else None,
        )
        self.batch_size = batch_size
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.processor_task = None

        # Ensure only one encode() runs at once (prevents cuBLAS handle/workspace churn)
        self._exec = ThreadPoolExecutor(max_workers=1)

        # Warm up CUDA context/handles early
        try:
            _ = self.model.encode(
                ["warmup"],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            pass

    def start(self):
        self.processor_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        self.stop_event.set()
        if self.processor_task:
            await self.processor_task
        self._exec.shutdown(wait=True)

    async def _encode_norm(self, texts: List[str]) -> torch.Tensor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._exec,
            lambda: self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,  # enables fast dot similarity
                show_progress_bar=False,
            ),
        )

    async def _process_batches(self):
        while not self.stop_event.is_set():
            items = []
            try:
                items.append(await asyncio.wait_for(self.queue.get(), timeout=0.5))
                while len(items) < self.batch_size:
                    try:
                        items.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                continue

            gen_texts = [it[0] for it in items]
            ref_embs = [it[1] for it in items]  # normalized 1D tensors on GPU
            futures = [it[2] for it in items]

            try:
                with torch.inference_mode():
                    gen_embs = await self._encode_norm(gen_texts)  # [B, D]
                    ref_mat = torch.stack(ref_embs, dim=0)  # [B, D]
                    scores = (gen_embs * ref_mat).sum(
                        dim=1
                    )  # dot == cosine (normalized)
                    scores_cpu = scores.float().cpu().tolist()

                for s, fut in zip(scores_cpu, futures):
                    if not fut.done():
                        fut.set_result(float(s))

            except Exception as e:
                logger.error(f"Encoding failed: {e}")
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                for _ in items:
                    self.queue.task_done()

    async def score(self, gen_text: str, ref_text: str) -> float:
        key = hash(ref_text)
        ref_emb = _ref_emb_cache.get(key)
        if ref_emb is None:
            ref_emb = await asyncio.to_thread(
                self.model.encode,
                ref_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            _ref_emb_cache[key] = ref_emb

        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((gen_text, ref_emb, fut))
        return await fut


# ==========================
# 4) PROMPTS & VALIDATION
# ==========================
INSTRUCTIONS_TEXT = "Reasoning: low.\n\n You are a creative writer skilled in emulating literary styles. And the story length must be strictly followed around 1200-1500 words. and the final line must be THE END. Count the words silently before finalizing."
BAD_META_RE = re.compile(
    r"\b(I am an AI assistant|As an AI|I specialize in)\b", re.IGNORECASE
)


def prepare_generation_prompt(reference_full: str, author: str, title: str) -> str:
    return dedent(
        f"""\
        Write ONE complete, self-contained short story in English (predominantly English prose).
        You MAY include brief non-standard markers that appear in <style_sample> (e.g., Gallicisms, short French phrases, dialect spellings),
        but do not write long non-English passages; keep any non-English insertion to at most 1–2 sentences at a time.

        Style conditioning:
        - Mimic the stylistic features of <style_sample> (cadence, sentence length, diction, imagery density, dialogue style, narration distance).
        - Do NOT quote or closely paraphrase <style_sample>; do not reuse any phrase longer than 5 consecutive words from it.
        - Do NOT reuse unique proper nouns from <style_sample> unless they are generic/common.
        - Do NOT mention the author or book title in the story text. (Author: {author}; Title: {title})

        Length & ending (STRICT):
        - Total output MUST be {MIN_WORDS}–{MAX_WORDS} words INCLUDING any optional title and the final line “THE END.”
        - TARGET 1350–1400 words.
        - By ~1350 words, begin final resolution; reserve ~120–180 words for the ending.
        - The FINAL line MUST be exactly: THE END.
        - After “THE END.” output nothing else.

        Output rules (STRICT):
        - No outlines, bullets, commentary, or meta text about being an AI/model/tokens/reasoning.
        - No epilogue or extra scene after the resolution.
        - Silently estimate word count before finalizing; if outside {MIN_WORDS}–{MAX_WORDS}, revise internally to fit.

        <style_sample>
        {reference_full}
        </style_sample>
        """
    ).strip()


def prepare_dataset_prompt(author: str, title: str) -> str:
    return dedent(
        f"""\
        # Style Target
        Author: {author}
        Title: {title}

        Task: Write an original, polished literary short story in this style.
        Constraints:
        - Do NOT mention the author or title in the story text.
        - Final line must be exactly: THE END.

        Story:
        """
    ).strip()


def _word_count(text: str) -> int:
    doc = nlp.make_doc(text)
    attr_array = doc.to_array([spacy.attrs.IS_PUNCT, spacy.attrs.IS_SPACE])
    return int((attr_array.sum(axis=1) == 0).sum())


async def validate_output(text: str) -> Tuple[bool, str, int]:
    text = (text or "").strip()
    if not text:
        return False, "empty", 0
    if BAD_META_RE.search(text):
        return False, "meta_content", 0
    if not text.endswith("THE END."):
        return False, "no_marker", 0
    wc = await asyncio.to_thread(_word_count, text)
    return True, "ok", wc


async def call_model(client: AsyncOpenAI, gen_prompt: str) -> str:
    resp = await client.responses.create(
        model=MODEL_NAME,
        instructions=INSTRUCTIONS_TEXT,
        input=gen_prompt,
        temperature=1.0,
        # max_output_tokens=MAX_OUTPUT_TOKENS,  # enable if supported by your server
    )
    return (getattr(resp, "output_text", "") or "").strip()


def precompute_ref_embeddings(judge: BatchScoreJudge, ds: Dataset, authors: List[str]):
    logger.info("Pre-computing reference embeddings...")
    unique_texts = set()
    for row in tqdm(ds, desc="Gathering chunks"):
        if row["author"] in authors:
            unique_texts.add(row["chunk_text"])

    unique_texts = list(unique_texts)
    logger.info(f"Encoding {len(unique_texts)} unique reference chunks...")

    all_embeddings = judge.model.encode(
        unique_texts,
        batch_size=JUDGE_BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    for text, emb in zip(unique_texts, all_embeddings):
        _ref_emb_cache[hash(text)] = emb
    logger.info("Reference embeddings cached.")


async def build_and_finalize_40k():
    ds_dict = load_from_disk(str(CHUNKS_PATH))
    ds = ds_dict[TARGET_SPLIT]

    available_authors = sorted(set(ds["author"]))
    available_authors_clean = {a.strip(): a for a in available_authors}

    authors = []
    for target in TARGET_AUTHORS:
        if target in available_authors:
            authors.append(target)
        elif target.strip() in available_authors_clean:
            authors.append(available_authors_clean[target.strip()])
        else:
            logger.warning(f"Target author '{target}' NOT FOUND.")
    if not authors:
        raise RuntimeError("No TARGET_AUTHORS found in dataset.")

    # Partition chunks
    logger.info("Partitioning source chunks...")
    authors_map_temp = {a: {} for a in authors}
    for row in tqdm(ds, desc="Partitioning"):
        row_auth = row.get("author")
        if not row_auth:
            continue
        target_key = (
            row_auth
            if row_auth in authors_map_temp
            else (row_auth.strip() if row_auth.strip() in authors_map_temp else None)
        )
        if target_key:
            txt = row.get("chunk_text", "")
            if txt:
                authors_map_temp[target_key][hash(txt)] = row

    authors_map = {a: list(authors_map_temp[a].values()) for a in authors}

    # Resumption
    per_author_target = TARGET_TOTAL_SAMPLES // len(authors)
    counts = Counter()
    total_existing = 0
    if TEMP_FILE.exists():
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    a = obj.get("author")
                    if a and (a in authors or a.strip() in authors):
                        key = a if a in authors else a.strip()
                        counts[key] += 1
                        total_existing += 1
                except Exception:
                    continue
    logger.info(f"Resuming with {total_existing} existing samples.")

    # Judge
    judge = BatchScoreJudge(
        JUDGE_MODEL_PATH, device="cuda", batch_size=JUDGE_BATCH_SIZE, fp16=True
    )
    await asyncio.to_thread(
        precompute_ref_embeddings, judge, ds, authors
    )  # keep your cache logic (store normalized tensors)
    judge.start()

    # IMPORTANT: httpx pool must allow >= API_CONCURRENCY connections (else you cap concurrency).
    http_client = httpx.AsyncClient(
        timeout=1200.0,
        limits=httpx.Limits(
            max_connections=API_CONCURRENCY + 50,
            max_keepalive_connections=API_CONCURRENCY,
        ),
    )
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, http_client=http_client)

    # Concurrency controls
    api_sem = asyncio.Semaphore(API_CONCURRENCY)
    score_sem = asyncio.Semaphore(SCORE_CONCURRENCY)

    # Queues
    task_q: asyncio.Queue = asyncio.Queue(maxsize=MAX_WORKERS * 2)
    post_q: asyncio.Queue = asyncio.Queue(maxsize=MAX_WORKERS * 4)
    write_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=MAX_WORKERS * 8)

    counts_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    rr_indices = {a: 0 for a in authors}

    ds_prompt_cache = {
        a: prepare_dataset_prompt(a, BOOK_META[a]["title"]) for a in authors
    }

    def _flush_lines(lines: List[str]) -> None:
        with open(TEMP_FILE, "a", encoding="utf-8") as f:
            f.writelines(lines)

    async def writer_task():
        buf: List[str] = []
        FLUSH_EVERY = 256

        while True:
            item = await write_q.get()
            try:
                if item is None:
                    # Flush remaining buffer before exiting
                    if buf:
                        await asyncio.to_thread(_flush_lines, buf)
                    break

                buf.append(item)
                if len(buf) >= FLUSH_EVERY:
                    await asyncio.to_thread(_flush_lines, buf)
                    buf.clear()
            except Exception as e:
                logger.error(f"Writer task failed to flush: {e}")
            finally:
                write_q.task_done()

    async def producer():
        while not stop_event.is_set():
            if task_q.qsize() < MAX_WORKERS:
                needed = []
                async with counts_lock:
                    for a in authors:
                        if counts[a] < per_author_target:
                            needed.append(a)
                if not needed:
                    stop_event.set()
                    break

                for author in needed:
                    if stop_event.is_set():
                        break
                    chunks = authors_map[author]
                    n_chunks = len(chunks)
                    if n_chunks == 0:
                        continue

                    idx = rr_indices[author] % n_chunks
                    prompt_obj = chunks[idx]
                    rr_indices[author] += 1

                    if n_chunks > 1:
                        reward_idx = random.randrange(n_chunks - 1)
                        if reward_idx >= idx:
                            reward_idx += 1
                        reward_obj = chunks[reward_idx]
                    else:
                        reward_obj = prompt_obj

                    task = {
                        "author": author,
                        "prompt_text": prompt_obj["chunk_text"],
                        "reward_text": reward_obj["chunk_text"],
                        "title": BOOK_META[author]["title"],
                        "genre": BOOK_META[author]["genre"],
                    }
                    try:
                        task_q.put_nowait(task)
                    except asyncio.QueueFull:
                        break
            await asyncio.sleep(0.02)

    # Stage 1: generate (keep vLLM saturated)
    async def gen_worker(worker_id: int):
        while not stop_event.is_set() or not task_q.empty():
            try:
                task = await asyncio.wait_for(task_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            try:
                author = task["author"]
                async with counts_lock:
                    if counts[author] >= per_author_target:
                        continue

                gen_prompt = prepare_generation_prompt(
                    task["prompt_text"], author, task["title"]
                )

                async with api_sem:
                    text = await call_model(client, gen_prompt)

                await post_q.put((task, text))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            finally:
                task_q.task_done()

    # Stage 2: validate + score + commit
    # Stage 2: validate + score + commit
    async def post_worker(worker_id: int):
        while not stop_event.is_set() or not post_q.empty():
            try:
                # Use a smaller timeout to ensure we check stop_event frequently during drain
                try:
                    task, text = await asyncio.wait_for(post_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                raise

            try:
                author = task["author"]
                async with counts_lock:
                    if counts[author] >= per_author_target:
                        continue

                ok, _, wc = await validate_output(text)
                if not ok:
                    continue

                async with score_sem:
                    score = await judge.score(text, task["reward_text"])

                if score < STYLE_SCORE_THRESHOLD:
                    continue

                async with counts_lock:
                    # Check again in case another worker filled the quota while we were scoring
                    if counts[author] >= per_author_target:
                        continue
                    counts[author] += 1

                entry = {
                    "prompt": ds_prompt_cache[author],
                    "completion": text,
                    "author": author,
                    "title": task["title"],
                    "genre": task["genre"],
                    "score": float(score),
                    "wc": int(wc),
                }

                await write_q.put(json.dumps(entry, ensure_ascii=False) + "\n")
                pbar.update(1)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Post-worker error: {e}")
            finally:
                post_q.task_done()

    # Run
    pbar = tqdm(total=TARGET_TOTAL_SAMPLES, initial=total_existing, desc="SFT Race")
    writer = asyncio.create_task(writer_task())
    prod_task = asyncio.create_task(producer())

    gen_workers = [asyncio.create_task(gen_worker(i)) for i in range(MAX_WORKERS)]
    post_workers = [
        asyncio.create_task(post_worker(i)) for i in range(max(1, MAX_WORKERS // 4))
    ]

    await prod_task
    logger.info("Target reached. Initiating hard shutdown...")
    for w in gen_workers + post_workers:
        w.cancel()
    await write_q.put(None)
    await writer
    await asyncio.gather(*gen_workers, *post_workers, return_exceptions=True)

    await judge.stop()
    pbar.close()
    await asyncio.wait_for(http_client.aclose(), timeout=30)

    final_data = []
    if TEMP_FILE.exists():
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    final_data.append(json.loads(line))
                except Exception:
                    continue
    if len(final_data) > TARGET_TOTAL_SAMPLES:
        final_data = final_data[:TARGET_TOTAL_SAMPLES]

    final_counts = Counter(d["author"] for d in final_data)
    logger.info("Final dataset distribution per author:")
    for auth in sorted(final_counts.keys()):
        logger.info(f"  {auth}: {final_counts[auth]}")

    random.shuffle(final_data)

    avg_score = np.mean([d["score"] for d in final_data]) if final_data else 0.0
    avg_wc = np.mean([d["wc"] for d in final_data]) if final_data else 0.0
    logger.info(f"Average style score: {avg_score:.4f}, Average wc: {avg_wc:.1f}")

    ds_final = Dataset.from_list(final_data)
    ds_final = ds_final.shuffle(seed=42).flatten_indices()

    OUTPUT_SFT_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds_final.save_to_disk(str(OUTPUT_SFT_DIR))
    logger.info(f"Done. Saved to: {OUTPUT_SFT_DIR} (rows={len(ds_final)})")


if __name__ == "__main__":
    OUTPUT_SFT_DIR.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(build_and_finalize_40k())
