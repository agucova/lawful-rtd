#!/usr/bin/env python3

import sys
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup, Comment

# List of block-level tags to keep
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "div"}
# List of inline tags to keep
INLINE_TAGS = {"b", "strong", "i", "em", "u"}
# If you want to keep <ul>, <li> or others, add them here.

def clean_soup(body: BeautifulSoup) -> None:
    """
    In-place clean up of a BeautifulSoup `body`:
      - Remove comments
      - Remove disallowed tags entirely
      - Strip unwanted attributes
      - Convert <br> to a line break or keep it as <br> if you prefer
      - Keep only certain block-level or inline tags, flatten the rest
    """

    # 1) Remove comments
    for comment in body.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 2) Remove <script> and <style> blocks entirely
    for bad_tag in body.find_all(["script", "style"]):
        bad_tag.decompose()

    # 3) Traverse all elements
    #    - If the tag is not in BLOCK_TAGS or INLINE_TAGS or <br>,
    #      we'll unwrap it (i.e., replace the tag with its text children).
    #    - If the tag is allowed, strip all attributes (like style, id, class).
    for element in body.find_all():
        # Convert <br> to a line break or keep them as <br>
        if element.name == "br":
            # Option A: Just leave <br> as is, but strip attributes
            element.attrs = {}
            continue

        if element.name not in BLOCK_TAGS and element.name not in INLINE_TAGS and element.name != "br":
            # "unwrap" => replace the tag with its children (plain text or inline)
            element.unwrap()
        else:
            # Strip attributes
            element.attrs = {}

def extract_epub_semantic_optimized(epub_path: Path) -> str:
    """
    Manually unzip the EPUB, parse each .html/.xhtml file,
    preserve paragraphs/headings/bold/etc. in a cleaned form.
    Then wrap each file's content in <section file="..."> ... </section>,
    and combine under <book> ... </book>.
    """

    with zipfile.ZipFile(epub_path, 'r') as zf:
        all_files = zf.namelist()
        # Accept .html, .htm, .xhtml
        html_files = [f for f in all_files if f.lower().endswith((".html", ".htm", ".xhtml"))]

        sections = []

        # Sort for stable ordering (some ePubs might name chapters in alpha order)
        for fname in sorted(html_files):
            raw_bytes = zf.read(fname)
            try:
                text_str = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_str = raw_bytes.decode("latin-1", errors="replace")

            soup = BeautifulSoup(text_str, "html.parser")

            # Clean up or remove <script>, <style>, comments, etc.
            # We'll look for a <body> if present:
            body = soup.find("body")
            if not body:
                # fallback: just operate on the entire soup
                body = soup

            clean_soup(body)

            # Convert the final body part back to HTML
            # If there's a <body> we can just gather its children, or do body.prettify()
            cleaned_html = "".join(str(x) for x in body.children).strip()

            # Wrap in <section>
            section_block = f"<section file=\"{fname}\">\n{cleaned_html}\n</section>"
            sections.append(section_block)

        # Combine everything in a <book> root
        combined = "<book>\n" + "\n\n".join(sections) + "\n</book>"
        return combined

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_epub_semantic_optimized.py input.epub output.xml")
        sys.exit(1)

    epub_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not epub_path.is_file():
        raise FileNotFoundError(f"EPUB not found at {epub_path}")

    final_result = extract_epub_semantic_optimized(epub_path)
    output_path.write_text(final_result, encoding="utf-8")

    print(f"Wrote structured, cleaned HTML-like output to {output_path}")

if __name__ == "__main__":
    main()
