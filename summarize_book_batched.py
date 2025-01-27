#!/usr/bin/env python3

"""
summarize_book_batched.py

Usage:
  python summarize_book_batched.py

Description:
  Reads the .txt file from output/project_lawful_full.txt, then
  performs multi-iteration chunk-based summarization using Anthropic's
  Batches API. Writes final summary to output/final_summary.txt, and
  iteration checkpoints to output/checkpoints/.
"""

import os
import json
import time
from pathlib import Path
from typing import List

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming


# =====================
# CONFIGURABLE PARAMS
# =====================

MAX_CHARS_PER_CHUNK = 15000
CHECKPOINT_DIR = Path("output/checkpoints_batched")  # for iteration_{N}.json

# Batches-supported model names include:
#   - claude-3-5-haiku-20241022
#   - claude-3-5-sonnet-20241022
#   - claude-3-haiku-20240307
#   - claude-3-opus-20240229
BATCH_MODEL = "claude-3-5-haiku-20241022"

# Rough conversion from chars -> tokens
CHARS_PER_TOKEN = 4.0


def chunk_text(text: str, max_chunk_size: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Splits the text into chunks of up to max_chunk_size characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def create_batch_of_chunk_requests(
    chunks: List[str], iteration: int, max_tokens_per_summary: int = 800
) -> str:
    """
    Creates a batch with one request per chunk.
    Returns the batch_id string.
    """
    client = anthropic.Anthropic()

    requests = []
    for i, chunk in enumerate(chunks):
        custom_id = f"it{iteration}-chunk{i+1}"

        # You can add a "system" role for extra context, or keep it simple:
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a helpful assistant that summarizes text.\n\n"
                    "Please summarize the following text in a concise manner:\n\n"
                    f"---\n{chunk}\n---\n"
                )
            }
        ]

        params = MessageCreateParamsNonStreaming(
            model=BATCH_MODEL,
            max_tokens=max_tokens_per_summary,
            messages=messages
        )
        requests.append(Request(custom_id=custom_id, params=params))

    batch = client.messages.batches.create(requests=requests)
    print(f"Created batch {batch.id} with {len(chunks)} requests (iteration={iteration}).")
    return batch.id


def wait_for_batch_completion(batch_id: str, poll_interval_sec: float = 5.0) -> None:
    """
    Poll the Batches API until the batch finishes (or expires/cancels).
    """
    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        print(
            f"  Polling batch {batch_id} => status={status}, "
            f"processing={counts.processing}, succeeded={counts.succeeded}, "
            f"errored={counts.errored}, canceled={counts.canceled}, expired={counts.expired}."
        )

        if status == "ended":
            return
        elif status == "in_progress":
            time.sleep(poll_interval_sec)
        elif status in ("canceled", "expired"):
            raise RuntimeError(f"Batch {batch_id} ended with status {status}")
        else:
            time.sleep(poll_interval_sec)


def retrieve_batch_results(batch_id: str) -> dict[str, str]:
    """
    Retrieve results from a finished batch. Returns a dict: { custom_id -> summary }.
    """
    client = anthropic.Anthropic()
    result_map = {}

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded" and result.result.message:
            # Combine text content segments
            segments = []
            for seg in result.result.message.content:
                if seg["type"] == "text":
                    segments.append(seg["text"])
            summary_text = "".join(segments).strip()
            result_map[cid] = summary_text
        else:
            print(f"  Request {cid} => {rtype}, storing empty summary.")
            result_map[cid] = ""

    return result_map


def summarize_iteration(iteration: int, input_text: str) -> List[str]:
    """
    - Chunk the text
    - Create & process a batch
    - Retrieve partial summaries
    - Write iteration checkpoint
    - Return partial summaries
    """
    chunks = chunk_text(input_text, MAX_CHARS_PER_CHUNK)
    print(f"[Iteration {iteration}] Summarizing {len(chunks)} chunks...")

    batch_id = create_batch_of_chunk_requests(chunks, iteration)
    wait_for_batch_completion(batch_id)
    results = retrieve_batch_results(batch_id)

    partial_summaries = []
    for i in range(len(chunks)):
        cid = f"it{iteration}-chunk{i+1}"
        partial_summaries.append(results.get(cid, ""))

    # Checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"iteration_{iteration}.json"
    checkpoint_file.write_text(json.dumps(partial_summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  => Wrote iteration {iteration} checkpoint -> {checkpoint_file}")
    return partial_summaries


def reconstruct_text_from_summaries(partial_summaries: List[str]) -> str:
    """
    Combine partial summaries into a single text for the next iteration.
    """
    return "\n".join(partial_summaries)


def iterative_summarize_text(full_text: str) -> str:
    """
    Repeatedly chunk + batch-summarize until text fits in 1 chunk.
    Uses iteration-based checkpoints in output/checkpoints_batched/.
    """
    existing_ckpts = sorted(CHECKPOINT_DIR.glob("iteration_*.json"))
    if existing_ckpts:
        # Resume from last checkpoint
        last_ckpt = existing_ckpts[-1]
        iteration_str = last_ckpt.stem.split("_")[-1]
        iteration = int(iteration_str) + 1

        partial_summaries = json.loads(last_ckpt.read_text(encoding="utf-8"))
        current_text = reconstruct_text_from_summaries(partial_summaries)
        print(f"Resuming from iteration {iteration-1}, text length={len(current_text)} chars.")
    else:
        iteration = 1
        current_text = full_text

    while True:
        if len(current_text) <= MAX_CHARS_PER_CHUNK:
            # Final pass
            print(f"[Iteration {iteration}] Final pass => single chunk of {len(current_text)} chars.")
            partial_sums = summarize_iteration(iteration, current_text)
            return partial_sums[0] if partial_sums else ""

        # Normal pass
        partial_sums = summarize_iteration(iteration, current_text)
        iteration += 1
        current_text = reconstruct_text_from_summaries(partial_sums)


def estimate_tokens(chars: int) -> int:
    """
    Rough estimate of tokens from chars.
    """
    return int(chars / CHARS_PER_TOKEN)


def main():
    # Must set your Anthropic key in the environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY in environment.")

    # We'll load from output/project_lawful_full.txt
    output_dir = Path("output")
    text_file = output_dir / "project_lawful_full.txt"
    final_summary_file = output_dir / "final_summary.txt"

    if not text_file.is_file():
        raise FileNotFoundError(f"Text file not found: {text_file}")

    full_text = text_file.read_text(encoding="utf-8")

    total_chars = len(full_text)
    approx = estimate_tokens(total_chars)
    print(f"Loaded {total_chars} chars => ~{approx} tokens from {text_file}.")

    final_summary = iterative_summarize_text(full_text)

    # Save final summary
    final_summary_file.write_text(final_summary, encoding="utf-8")
    print(f"\n=============== Final Summary ===============\n\n{final_summary}\n")
    print(f"Saved final summary => {final_summary_file}")

if __name__ == "__main__":
    main()
