#!/usr/bin/env python3

import sys
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup

def extract_epub_semantic(epub_path: Path) -> str:
    """
    Unzip the EPUB manually and parse each .xhtml/.html file with BeautifulSoup.
    Return a single XML/HTML string that has some minimal structure.

    For example:
      <book>
        <section file="chapter1.xhtml">
          <heading level="1">Chapter 1</heading>
          <paragraph>Some text here.</paragraph>
          <heading level="2">A subheading</heading>
          <paragraph>Another paragraph.</paragraph>
        </section>
        <section file="chapter2.xhtml">
          ...
        </section>
      </book>
    """

    with zipfile.ZipFile(epub_path, 'r') as zf:
        all_files = zf.namelist()
        # Filter to HTML-ish file extensions:
        html_files = [f for f in all_files if f.lower().endswith((".html", ".xhtml", ".htm"))]

        # We'll store everything in a list of <section> blocks
        sections = []

        for fname in sorted(html_files):
            raw_html = zf.read(fname)  # read bytes
            try:
                text_str = raw_html.decode("utf-8")
            except UnicodeDecodeError:
                # fallback if some file is in another encoding
                text_str = raw_html.decode("latin-1", errors="replace")

            soup = BeautifulSoup(text_str, "html.parser")

            # Build up a string that has headings + paragraphs, etc.
            content_blocks = []
            # We'll walk the DOM top-down
            for element in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol"]):
                if element.name in ["h1", "h2", "h3"]:
                    level = element.name[-1]  # "1", "2", or "3"
                    heading_text = element.get_text(strip=True)
                    content_blocks.append(f"<heading level=\"{level}\">{heading_text}</heading>")
                elif element.name == "p":
                    p_text = element.get_text(separator=" ", strip=True)
                    # Skip empty paragraphs
                    if p_text:
                        content_blocks.append(f"<paragraph>{p_text}</paragraph>")
                elif element.name in ["ul", "ol"]:
                    # If you'd like to preserve list structure:
                    li_list = []
                    for li in element.find_all("li"):
                        li_text = li.get_text(" ", strip=True)
                        if li_text:
                            li_list.append(li_text)
                    if li_list:
                        if element.name == "ul":
                            tag = "unordered_list"
                        else:
                            tag = "ordered_list"
                        items_str = "".join(f"<list_item>{x}</list_item>" for x in li_list)
                        content_blocks.append(f"<{tag}>{items_str}</{tag}>")

            # If you want to keep entire file structure, wrap in <section>
            if content_blocks:
                joined_content = "\n  ".join(content_blocks)
                section_xml = f"<section file=\"{fname}\">\n  {joined_content}\n</section>"
                sections.append(section_xml)

        # Final doc
        doc_body = "\n\n".join(sections)
        final_xml = f"<book>\n{doc_body}\n</book>"
        return final_xml

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_epub_semantic.py input.epub output.xml")
        sys.exit(1)

    epub_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not epub_path.is_file():
        raise FileNotFoundError(f"EPUB not found at {epub_path}")

    xml_content = extract_epub_semantic(epub_path)
    output_path.write_text(xml_content, encoding="utf-8")
    print(f"Wrote structured XML to {output_path}")

if __name__ == "__main__":
    main()
