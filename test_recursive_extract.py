from pathlib import Path
from typing import List, Tuple
from lxml import etree
from rich import print as rprint
import re

RECURSIVE_LEVEL_MAX_SECTIONS = 4

def parse_xml(text: str):
    """Parse the given text as XML, returning the root element or None if there's an error."""
    parser = etree.XMLParser(remove_comments=True, recover=True, remove_blank_text=True)
    try:
        root = etree.fromstring(text, parser=parser)
        return root
    except Exception as e:
        print(f"Warning: parse_xml error: {e}")
        return None

def print_element_tree(element, level=0):
    """Print the full XML tree structure with all attributes and text content."""
    indent = "  " * level
    print(f"{indent}Element: {element.tag}")
    print(f"{indent}Attributes: {dict(element.attrib)}")
    if element.text and element.text.strip():
        print(f"{indent}Text: {element.text[:100].strip()}...")
    for child in element:
        print_element_tree(child, level + 1)

def gather_blocks_recursive(root) -> List[Tuple[str, List[str]]]:
    """
    For recursive levels: gather sections into coherent groups for synthesis.
    Returns list of (context, list_of_section_xmls).
    """
    print("\nDEBUG: Full XML structure:")
    print_element_tree(root)

    # First try to get sections using direct children
    sections = list(root.iterchildren("section"))
    print(f"\nDEBUG: Found {len(sections)} sections")

    # Debug each section
    valid_sections = []
    for i, section in enumerate(sections, 1):
        print(f"\nDEBUG: Processing section {i}:")

        # Get all text content within the section
        full_text = "".join(section.itertext()).strip()

        # Try to extract analysis and summary using text patterns
        analysis_match = re.search(r'<chunk_analysis>(.*?)</chunk_analysis>', full_text, re.DOTALL)
        analysis_text = analysis_match.group(1).strip() if analysis_match else ""

        # Get summary text (everything after the last </chunk_analysis>)
        summary_text = re.split(r'</chunk_analysis>', full_text)[-1].strip()

        if analysis_text or summary_text:
            section_text = ""
            if analysis_text:
                section_text += f"<chunk_analysis>{analysis_text}</chunk_analysis>"
            if summary_text:
                section_text += summary_text

            if section_text.strip():
                valid_sections.append(section_text)
                print(f"DEBUG: Added section with {len(section_text)} chars")
                print(f"First 100 chars: {section_text[:100]}...")

    print(f"\nDEBUG: Collected {len(valid_sections)} valid sections")

    # Group sections for processing
    chunks: List[Tuple[str, List[str]]] = []
    for i in range(0, len(valid_sections), RECURSIVE_LEVEL_MAX_SECTIONS):
        group = valid_sections[i:i + RECURSIVE_LEVEL_MAX_SECTIONS]
        if group:
            context = f"Sections {i+1} to {i+len(group)} of {len(valid_sections)}"
            chunks.append((context, group))
            print(f"DEBUG: Created chunk {len(chunks)} with {len(group)} sections")

    return chunks

def main():
    # Load iteration 1 checkpoint
    checkpoint_path = Path("output/checkpoints/iteration_1.xml")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    xml_content = checkpoint_path.read_text(encoding='utf-8')
    root = parse_xml(xml_content)
    if root is None:
        print("Error parsing XML")
        return

    # Test the gather_blocks_recursive function
    chunks = gather_blocks_recursive(root)

    print("\nRESULTS:")
    print(f"Total chunks created: {len(chunks)}")
    for i, (context, sections) in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Context: {context}")
        print(f"Number of sections: {len(sections)}")
        print(f"First section preview (100 chars): {sections[0][:100]}...")

if __name__ == "__main__":
    main()
