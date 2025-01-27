import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import anthropic
from lxml import etree

# Adjust imports to match your installed Anthropic Python SDK version:
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

# =====================
# CONFIGURABLE PARAMS
# =====================

MAX_CHARS_PER_CHUNK = 15000
CHECKPOINT_DIR = Path("output/checkpoints_xml_h1")
BATCH_MODEL = "claude-3-5-haiku-20241022"  # or any Batches-supported model
CHARS_PER_TOKEN = 4.0

# Prompt template: includes heading context
PROMPT_TEMPLATE = """You are a helpful assistant that summarizes text from a larger document.

Context Heading: {heading}

Please summarize the following text in a concise manner:

---
{chunk}
---
"""


def parse_xml(text: str):
    """
    Parse the given text as XML, returning the root element or None if there's an error.
    """
    parser = etree.XMLParser(remove_comments=True, recover=True)
    try:
        root = etree.fromstring(text, parser=parser)
        return root
    except Exception as e:
        print(f"Warning: parse_xml error: {e}")
        return None


def gather_blocks_with_h1(root) -> List[Tuple[str, str]]:
    """
    Traverse <book> and <section>, tracking the most recent <h1> as the "context heading."
    Return a list of (heading_context, block_html).
    """
    blocks: List[Tuple[str, str]] = []
    current_h1 = "Unknown"

    for section in root.findall(".//section"):
        for element in section:
            tag = element.tag.lower()
            if tag == "h1":
                # Extract text from <h1>
                current_h1 = element.xpath("string()").strip()
            elif tag in {"p", "h2", "h3", "h4", "h5", "h6", "div"}:
                block_str = etree.tostring(element, encoding="unicode", pretty_print=False).strip()
                blocks.append((current_h1, block_str))
            else:
                # skip or handle other tags as you wish
                pass

    return blocks


def chunk_blocks_grouped_by_h1(
    blocks: List[Tuple[str, str]],
    max_chars: int = MAX_CHARS_PER_CHUNK
) -> List[Tuple[str, List[str]]]:
    """
    Produce a list of chunks, each chunk = (heading_context, [list_of_block_html]).
    We never mix blocks with different <h1> contexts in the same chunk,
    and we also enforce the max_chars limit. (Heading is always a string.)
    """
    chunks: List[Tuple[str, List[str]]] = []

    # Start with an empty string—never None—so we avoid type-check errors
    current_context: str = ""
    current_blocks: List[str] = []
    current_len = 0

    for (ctx, block_html) in blocks:
        block_len = len(block_html)
        if ctx != current_context or (current_len + block_len > max_chars):
            if current_blocks:
                chunks.append((current_context, current_blocks))
            current_context = ctx
            current_blocks = [block_html]
            current_len = block_len
        else:
            current_blocks.append(block_html)
            current_len += block_len

    # final chunk
    if current_blocks:
        chunks.append((current_context, current_blocks))

    return chunks


def create_batch_with_context(chunks: List[Tuple[str, List[str]]], iteration: int) -> str:
    """
    Build a single batch of requests for the given iteration.
    Each chunk has (heading_context, [block_htmls]).
    We'll pass a prompt to Claude with that heading and the chunk's HTML.
    """
    client = anthropic.Anthropic()
    requests: List[Request] = []

    for i, (heading, blocks) in enumerate(chunks, start=1):
        combined_html = "\n".join(blocks)
        prompt = PROMPT_TEMPLATE.format(heading=heading, chunk=combined_html)

        custom_id = f"it{iteration}-chunk{i}"
        params = MessageCreateParamsNonStreaming(
            model=BATCH_MODEL,
            max_tokens=1000,  # adjust as needed
            messages=[{"role": "user", "content": prompt}]
        )
        requests.append(Request(custom_id=custom_id, params=params))

    batch = client.messages.batches.create(requests=requests)
    print(f"Created batch {batch.id} with {len(chunks)} requests (iteration={iteration}).")
    return batch.id


def wait_for_batch_completion(batch_id: str, poll_interval_sec: float = 5.0) -> None:
    """
    Poll the Batches API until the batch finishes or fails.
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
        else:
            raise RuntimeError(f"Batch ended with unexpected status: {status}")


def retrieve_batch_results(batch_id: str) -> Dict[str, str]:
    """
    Retrieve the results from the batch. Return { custom_id => summary_text }.

    Each item in results() has item.result, whose .type can be:
      - "succeeded"
      - "errored"
      - "canceled"
      - "expired"

    We only read item.result.message and item.result.message.content
    if item.result.type == "succeeded".
    """
    client = anthropic.Anthropic()
    out: Dict[str, str] = {}

    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            # item.result should have .message, .message.content
            success_result = item.result
            if success_result.message and success_result.message.content:
                segments: List[str] = []
                for seg in success_result.message.content:
                    if seg.type == "text":
                        segments.append(seg.text)
                summary_text = "".join(segments).strip()
                out[item.custom_id] = summary_text
            else:
                out[item.custom_id] = ""
        else:
            # For errored/canceled/expired, store empty
            out[item.custom_id] = ""

    return out


def build_xml_from_summaries(summaries: List[str]) -> str:
    """
    Build minimal well-formed XML from partial summaries, e.g.:
      <book>
        <section iteration="1">
          <summary> ... </summary>
        </section>
        <section iteration="2">
          <summary> ... </summary>
        </section>
      </book>
    """
    sections = []
    for i, summ in enumerate(summaries, start=1):
        escaped_summ = summ  # For robust safety, you could XML-escape here
        section_xml = f'<section iteration="{i}">\n  <summary>{escaped_summ}</summary>\n</section>'
        sections.append(section_xml)

    combined = "<book>\n" + "\n\n".join(sections) + "\n</book>"
    return combined


def summarize_iteration(iteration: int, xml_text: str) -> str:
    """
    Single summarization iteration:
      1) Parse xml_text
      2) gather (heading, block_html)
      3) chunk by heading & size
      4) create + process a batch
      5) build minimal XML from partial summaries
      6) return that new XML
    """
    root = parse_xml(xml_text)
    if not root:
        # fallback: parse error => treat all as one chunk
        return f"<book><section iteration=\"{iteration}\"><summary>(Parse error) {xml_text}</summary></section></book>"

    blocks = gather_blocks_with_h1(root)
    chunked = chunk_blocks_grouped_by_h1(blocks, MAX_CHARS_PER_CHUNK)
    batch_id = create_batch_with_context(chunked, iteration)
    wait_for_batch_completion(batch_id)
    results = retrieve_batch_results(batch_id)

    # Reassemble partial summaries in correct order
    partial_summaries: List[str] = []
    for i in range(len(chunked)):
        cid = f"it{iteration}-chunk{i+1}"
        partial_summaries.append(results.get(cid, ""))

    iteration_xml = build_xml_from_summaries(partial_summaries)
    return iteration_xml


def iterative_summarize_text(full_text: str) -> str:
    """
    Main iterative loop:
    - If the text is small enough for 1 chunk => final pass
    - Otherwise do partial summarization
    - Store checkpoint each iteration
    - Stop when final
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(CHECKPOINT_DIR.glob("iteration_*.xml"))

    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        iteration_num = int(last_ckpt.stem.split("_")[-1]) + 1
        current_text = last_ckpt.read_text(encoding="utf-8")
        print(f"Resuming from iteration {iteration_num-1}, loaded {last_ckpt}")
    else:
        iteration_num = 1
        current_text = full_text

    while True:
        root = parse_xml(current_text)
        if not root:
            print("Parse error => single pass summarization")
            final_xml = summarize_iteration(iteration_num, current_text)
            path_ckpt = CHECKPOINT_DIR / f"iteration_{iteration_num}.xml"
            path_ckpt.write_text(final_xml, encoding="utf-8")
            return final_xml

        blocks = gather_blocks_with_h1(root)
        total_len = sum(len(b[1]) for b in blocks)
        if total_len <= MAX_CHARS_PER_CHUNK:
            # final pass
            print(f"[Iteration {iteration_num}] Single-chunk pass with {len(blocks)} blocks (~{total_len} chars).")
            final_xml = summarize_iteration(iteration_num, current_text)
            path_ckpt = CHECKPOINT_DIR / f"iteration_{iteration_num}.xml"
            path_ckpt.write_text(final_xml, encoding="utf-8")
            return final_xml
        else:
            # partial pass
            print(f"[Iteration {iteration_num}] Summarizing {len(blocks)} blocks (~{total_len} chars)...")
            out_xml = summarize_iteration(iteration_num, current_text)
            ckpt_path = CHECKPOINT_DIR / f"iteration_{iteration_num}.xml"
            ckpt_path.write_text(out_xml, encoding="utf-8")
            iteration_num += 1
            current_text = out_xml


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


def main():
    if len(sys.argv) != 2:
        print("Usage: python summarize_book_batched_xml_h1.py path/to/book.xml")
        sys.exit(1)

    xml_file = Path(sys.argv[1])
    if not xml_file.is_file():
        raise FileNotFoundError(f"XML file not found: {xml_file}")

    # Make sure Anthropic API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY in environment.")

    # Just ensuring the key is in environment for the client
    os.environ["ANTHROPIC_API_KEY"] = api_key

    full_text = xml_file.read_text(encoding="utf-8")
    total_chars = len(full_text)
    approx = estimate_tokens(total_chars)
    print(f"Loaded {total_chars} chars => ~{approx} tokens from {xml_file}.\n")

    final_summary = iterative_summarize_text(full_text)

    # Write final
    final_path = Path("output/final_summary_with_h1.xml")
    final_path.write_text(final_summary, encoding="utf-8")
    print(f"\n================ FINAL SUMMARY (XML) ================\n{final_summary[:1000]}...\n")
    print(f"Wrote final summary => {final_path}")


if __name__ == "__main__":
    main()
