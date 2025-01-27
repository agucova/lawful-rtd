#!/usr/bin/env python3

import sys
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup

def extract_epub_lenient(epub_path: Path) -> str:
    """
    Unzip the EPUB manually and parse each .xhtml/.html file with BeautifulSoup.
    Keep most of the body tags, removing only <script> and <style>.
    Return a big combined string with <section> elements for each file.

    Example final structure:

    <book>
      <section file="chapter1.xhtml">
        ... original body content (with headings, paragraphs, br, etc.) ...
      </section>

      <section file="chapter2.xhtml">
        ... ...
      </section>
    </book>
    """

    with zipfile.ZipFile(epub_path, 'r') as zf:
        all_files = zf.namelist()

        # We'll consider .html, .htm, and .xhtml as textual pages
        html_files = [f for f in all_files if f.lower().endswith((".html", ".htm", ".xhtml"))]

        sections = []

        for fname in sorted(html_files):
            raw_bytes = zf.read(fname)
            try:
                html_str = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # fallback if needed
                html_str = raw_bytes.decode("latin-1", errors="replace")

            soup = BeautifulSoup(html_str, "html.parser")

            # Remove script/style tags, but keep everything else
            for bad_tag in ["script", "style"]:
                for t in soup.find_all(bad_tag):
                    t.decompose()

            # If there's a <body>, we'll pretty-print just that. Otherwise, entire doc.
            body = soup.find("body")
            if body:
                # prettify() can re-indent tags; if you want the raw HTML, do body.decode_contents()
                content_html = body.prettify()
            else:
                # fallback: no <body> found => entire soup
                content_html = soup.prettify()

            # Wrap each file in a <section> block
            section = f"<section file=\"{fname}\">\n{content_html}\n</section>"
            sections.append(section)

        final = "<book>\n" + "\n\n".join(sections) + "\n</book>"
        return final

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_epub_lenient.py input.epub output.xml")
        sys.exit(1)

    epub_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not epub_path.is_file():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    final_xml = extract_epub_lenient(epub_path)
    output_path.write_text(final_xml, encoding="utf-8")

    print(f"Wrote more lenient structured HTML/XML to {output_path}")

if __name__ == "__main__":
    main()
