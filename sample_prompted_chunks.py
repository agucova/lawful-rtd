#!/usr/bin/env python3

"""
sample_prompted_chunks.py

Usage:
  python sample_prompted_chunks.py path/to/book.xml

Description:
  Reads a structured XML, finds blocks (<p>, <h2> etc.), tracks the
  most recent <h1> as the "context heading," and chunk them up to
  MAX_CHARS_PER_CHUNK. Then prints some sample prompted chunks
  where we incorporate that "context heading" into the prompt.

Requires:
  pip install lxml
"""

import sys
import random
from pathlib import Path
from typing import List, Tuple
from lxml import etree

MAX_CHARS_PER_CHUNK = 15000
NUM_SAMPLES_TO_SHOW = 3

PROMPT_TEMPLATE = """You are a helpful assistant that summarizes text from a larger document.

Context Heading: {context}

Please summarize the following text in a concise manner:

---
{chunk}
---
"""

def parse_xml_file(xml_path: Path):
    parser = etree.XMLParser(remove_comments=True, recover=True)
    return etree.parse(str(xml_path), parser=parser).getroot()

def gather_blocks(root) -> List[Tuple[str, str]]:
    """
    Traverse the <book>, track the nearest <h1> as 'context heading'.
    Return a list of (context_h1, block_html).

    Example:
      - If you see <h1>Chapter 3</h1>, store last_h1 = "Chapter 3".
      - For each subsequent <p> or <h2> etc., yield (last_h1, that_element_html).
    """
    blocks = []
    current_h1 = "Unknown"

    for section in root.findall(".//section"):
        for element in section:
            tag = element.tag.lower()
            # If it's <h1>, update the context
            if tag == "h1":
                # Extract text from this heading
                current_h1 = element.xpath("string()").strip()
                # We might also want to yield the H1 itself as a block, if you like:
                # blocks.append( (current_h1, etree.tostring(element, encoding="unicode").strip()) )
            elif tag in {"p", "h2", "h3", "h4", "h5", "h6", "div"}:
                block_str = etree.tostring(element, encoding="unicode").strip()
                blocks.append((current_h1, block_str))

    return blocks

def chunk_blocks_with_context(
    blocks: List[Tuple[str, str]], max_chars=15000
) -> List[Tuple[str, List[str]]]:
    """
    We'll produce a list of chunks. Each chunk is a tuple:
      (context_h1, [block_htmls])

    BUT: multiple blocks might have different contexts. We need to decide how
    to handle that if a single chunk merges blocks from different <h1> headings.

    Simplest approach: whenever the heading changes, we start a new chunk.
    That ensures each chunk has a single context heading. Then we also watch
    the max_chars limit to avoid too-large chunks.

    This means we can't combine blocks from different <h1>s into a single chunk.
    If that's acceptable, do it this way. Alternatively, you might incorporate
    multiple headings in the chunk prompt, but let's keep it simple.
    """
    chunks = []
    current_blocks = []
    current_len = 0
    current_context = None

    for (ctx, block_html) in blocks:
        # If context changed, or if adding block would exceed size, push the current chunk out
        if ctx != current_context or (current_len + len(block_html) > max_chars):
            # finalize old chunk
            if current_blocks:
                # store chunk
                chunks.append((current_context, current_blocks))
            # reset
            current_blocks = [block_html]
            current_context = ctx
            current_len = len(block_html)
        else:
            current_blocks.append(block_html)
            current_len += len(block_html)

    # finalize last chunk
    if current_blocks:
        chunks.append((current_context, current_blocks))
    return chunks


def main():
    if len(sys.argv) != 2:
        print("Usage: python sample_prompted_chunks_xml_with_context.py path/to/book.xml")
        sys.exit(1)

    xml_path = Path(sys.argv[1])
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    root = parse_xml_file(xml_path)
    # Gather (h1_context, block_html)
    blocks = gather_blocks(root)
    print(f"Gathered {len(blocks)} blocks with heading context.\n")

    # Build chunked data
    chunked = chunk_blocks_with_context(blocks, max_chars=MAX_CHARS_PER_CHUNK)
    print(f"Created {len(chunked)} chunks, each chunk has a single context heading.\n")

    # We'll sample a few random chunks
    if len(chunked) <= NUM_SAMPLES_TO_SHOW:
        sample_indices = list(range(len(chunked)))
    else:
        sample_indices = sorted(random.sample(range(len(chunked)), NUM_SAMPLES_TO_SHOW))

    for i, idx in enumerate(sample_indices, start=1):
        context_h1, block_list = chunked[idx]
        chunk_text = "\n".join(block_list)

        prompt = PROMPT_TEMPLATE.format(context=context_h1, chunk=chunk_text)
        print("="*80)
        print(f"Sample {i} (chunk index={idx}):\n")
        print(prompt)
        print("="*80)
        print()

if __name__ == "__main__":
    main()
