import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import anthropic
import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.prompt import Confirm
from lxml import etree
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datetime import datetime

from prompts import FIRST_LEVEL_PROMPT, RECURSIVE_PROMPT

# =====================
# CONFIGURABLE PARAMS
# =====================

FIRST_LEVEL_MAX_CHARS = 10000
RECURSIVE_LEVEL_MAX_SECTIONS = 4  # Number of sections to combine in recursive steps
CHECKPOINT_DIR = Path("output/checkpoints_xml_h1")
BATCH_MODEL = "claude-3-5-haiku-20241022"
CHARS_PER_TOKEN = 3.5

app = typer.Typer(help="Project Lawful summarization script with recursive task decomposition")

def parse_xml(text: str):
    """Parse the given text as XML, returning the root element or None if there's an error."""
    parser = etree.XMLParser(remove_comments=True, recover=True)
    try:
        root = etree.fromstring(text, parser=parser)
        return root
    except Exception as e:
        print(f"Warning: parse_xml error: {e}")
        return None

def gather_blocks_with_h1(root, test_mode: bool = False) -> List[Tuple[str, str]]:
    """
    For first iteration: traverse and gather blocks based on h1 headers.
    Returns list of (heading_context, block_html).
    In test mode, only processes first ~50k words.
    """
    blocks: List[Tuple[str, str]] = []
    current_h1 = "Unknown"
    total_words = 0
    WORD_LIMIT = 150000  # Test mode word limit

    for section in root.findall(".//section"):
        for element in section:
            tag = element.tag.lower()
            if tag == "h1":
                current_h1 = element.xpath("string()").strip()
            elif tag in {"p", "h2", "h3", "h4", "h5", "h6", "div"}:
                block_str = etree.tostring(element, encoding="unicode", pretty_print=False).strip()
                blocks.append((current_h1, block_str))

                if test_mode:
                    # Rough word count estimation
                    text_content = element.xpath("string()").strip()
                    words_in_block = len(text_content.split())
                    total_words += words_in_block

                    if total_words > WORD_LIMIT:
                        print(f"Test mode: Stopping after ~{total_words:,} words ({len(blocks)} blocks collected)")
                        return blocks

    return blocks

def gather_blocks_recursive(root) -> List[Tuple[str, List[str]]]:
    sections = root.findall(".//section")
    total_sections = len(sections)
    print(f"DEBUG: Found {total_sections} sections to process recursively")

    # Collect all valid summaries with their analyses
    valid_sections = []
    for section in sections:
        analysis = section.find(".//analysis")
        summary = section.find(".//summary")

        if summary is not None:
            # Combine analysis and summary if both exist
            section_text = ""
            if analysis is not None:
                # Get complete analysis content including nested elements
                analysis_text = "".join(analysis.itertext()).strip()
                if analysis_text:
                    section_text += f"<chunk_analysis>{analysis_text}</chunk_analysis>"

            # Get complete summary content including nested elements
            summary_text = "".join(summary.itertext()).strip()
            if summary_text:
                section_text += summary_text

            if section_text.strip():
                valid_sections.append(section_text)

    print(f"DEBUG: Collected {len(valid_sections)} valid sections with content")

    # If we only have one section, don't create a chunk
    if len(valid_sections) <= 1:
        print("DEBUG: Only one section found - no need for recursive processing")
        return []

    # Group sections for processing
    chunks: List[Tuple[str, List[str]]] = []
    for i in range(0, len(valid_sections), RECURSIVE_LEVEL_MAX_SECTIONS):
        group = valid_sections[i:i + RECURSIVE_LEVEL_MAX_SECTIONS]
        if len(group) > 1:  # Only create chunks if we have multiple sections to combine
            context = f"Sections {i+1} to {i+len(group)} of {len(valid_sections)}"
            chunks.append((context, group))
            print(f"DEBUG: Created chunk {len(chunks)} with {len(group)} sections")

    return chunks


def chunk_blocks_grouped_by_h1(
    blocks: List[Tuple[str, str]],
    max_chars: int = FIRST_LEVEL_MAX_CHARS
) -> List[Tuple[str, List[str]]]:
    """
    First iteration chunking: group blocks by h1 header while respecting max_chars.
    """
    chunks: List[Tuple[str, List[str]]] = []
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

    if current_blocks:
        chunks.append((current_context, current_blocks))

    return chunks

def estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    """Estimate number of tokens from text using character count."""
    return int(len(text) / chars_per_token)

def calculate_batch_cost(input_tokens: int, estimated_output_tokens: int = 1000) -> float:
    """
    Calculate batch cost using Haiku batch pricing.
    Input: $0.125 per million tokens
    Output: $0.625 per million tokens
    """
    input_cost = (input_tokens / 1_000_000) * 0.125
    output_cost = (estimated_output_tokens / 1_000_000) * 0.625
    return input_cost + output_cost

def display_batch_preview(chunks: List[Tuple[str, List[str]]], test_mode: bool) -> None:
    """Display a preview of what will be processed and request confirmation."""
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Chapter")
    table.add_column("Chunks")
    table.add_column("Input Tokens (est.)")
    table.add_column("Cost (est.)")

    total_chunks = 0
    total_input_tokens = 0
    chapters_seen = set()

    # Assume each summary will be around 1000 tokens
    estimated_output_tokens_per_chunk = 1000

    for heading, blocks in chunks:
        chapters_seen.add(heading)
        chunk_count = len(blocks)
        input_tokens = estimate_tokens("".join(blocks))

        total_chunks += chunk_count
        total_input_tokens += input_tokens

        chunk_cost = calculate_batch_cost(
            input_tokens,
            estimated_output_tokens_per_chunk * chunk_count
        )

        table.add_row(
            heading,
            str(chunk_count),
            f"{input_tokens:,}",
            f"${chunk_cost:.4f}"
        )

    total_cost = calculate_batch_cost(
        total_input_tokens,
        estimated_output_tokens_per_chunk * total_chunks
    )

    rprint("\n[bold blue]🔍 Batch Processing Preview:[/bold blue]")
    rprint(f"Mode: {'[yellow]TEST MODE[/yellow]' if test_mode else '[green]FULL MODE[/green]'}")
    rprint(f"Unique Chapters: {len(chapters_seen)}")
    rprint(f"Total Chunks: {total_chunks}")
    rprint(f"Total Input Tokens (est.): {total_input_tokens:,}")
    rprint(f"Total Output Tokens (est.): {total_chunks * estimated_output_tokens_per_chunk:,}")
    rprint(f"[bold]Estimated Total Cost: ${total_cost:.4f}[/bold]")
    rprint("[dim](Based on Haiku batch pricing: $0.125/MTok input, $0.625/MTok output)[/dim]")

    console.print("\nDetailed Breakdown:")
    console.print(table)

    if not Confirm.ask("\nProceed with processing?"):
        raise typer.Abort()

def create_batch_with_context(
    chunks: List[Tuple[str, List[str]]],
    iteration: int,
    dry_run: bool = False
) -> str:
    """Build a single batch of requests for the given iteration."""
    if dry_run:
        return "dry-run-batch-id"

    client = anthropic.Anthropic()
    requests: List[Request] = []

    for i, (heading, blocks) in enumerate(chunks, start=1):
        combined_text = "\n".join(blocks)

        if iteration == 1:
            prompt = FIRST_LEVEL_PROMPT.format(
                chapter_title=heading,
                chunk=combined_text
            )
        else:
            prompt = RECURSIVE_PROMPT.format(
                heading=heading,
                chunk=combined_text
            )

        custom_id = f"it{iteration}-chunk{i}"
        params = MessageCreateParamsNonStreaming(
            model=BATCH_MODEL,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": "<chunk_analysis>"
                }
            ]
        )
        requests.append(Request(custom_id=custom_id, params=params))

    batch = client.messages.batches.create(requests=requests)
    print(f"Created batch {batch.id} with {len(chunks)} requests (iteration={iteration}).")
    return batch.id

def create_progress_display():
    """Create a rich progress display for batch processing."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        expand=True
    )
    return progress

def wait_for_batch_completion(batch_id: str, poll_interval_sec: float = 5.0) -> None:
    """Poll the Batches API until the batch finishes or fails, with improved progress display."""
    client = anthropic.Anthropic()
    start_time = datetime.now()

    with create_progress_display() as progress:
        # Create the main progress task
        task_id = progress.add_task("Processing batch...", total=None)

        while True:
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts

            # Calculate total and completed
            total = (counts.processing + counts.succeeded +
                    counts.errored + counts.canceled + counts.expired)
            completed = counts.succeeded + counts.errored + counts.canceled + counts.expired

            # Update progress description with detailed status
            elapsed = datetime.now() - start_time
            status_text = (
                f"[bold]Batch {batch_id}[/bold] ({elapsed.total_seconds():.0f}s)\n"
                f"Status: [cyan]{status}[/cyan]\n"
                f"✓ Completed: {counts.succeeded}\n"
                f"⚠ Errors: {counts.errored}\n"
                f"⏳ Processing: {counts.processing}\n"
                f"✗ Canceled/Expired: {counts.canceled + counts.expired}"
            )

            # Update progress
            if total > 0:
                progress.update(task_id, total=total, completed=completed,
                              description=status_text)

            if status == "ended":
                # Show final status before returning
                progress.update(task_id, description=f"{status_text}\n[green]Complete![/green]")
                return
            elif status == "in_progress":
                time.sleep(poll_interval_sec)
            else:
                progress.update(task_id, description=f"{status_text}\n[red]Error: Unexpected status[/red]")
                raise RuntimeError(f"Batch ended with unexpected status: {status}")

def retrieve_batch_results(batch_id: str) -> Dict[str, str]:
    """Retrieve results from the batch. Return {custom_id => summary_text}."""
    client = anthropic.Anthropic()
    out: Dict[str, str] = {}

    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
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
            out[item.custom_id] = ""

    return out

def build_xml_from_summaries(summaries: List[str]) -> str:
    root = etree.Element("book")

    for i, summ in enumerate(summaries, start=1):
        if not summ.strip():
            continue

        section = etree.SubElement(root, "section")
        section.set("iteration", str(i))

        summary = etree.SubElement(section, "summary")
        summary.text = summ.strip()

    return etree.tostring(root, encoding="unicode", pretty_print=True)

def summarize_iteration(
    iteration: int,
    xml_text: str,
    test_mode: bool = False,
    dry_run: bool = False
) -> str:
    """Single summarization iteration with different handling for first vs recursive levels."""
    root = parse_xml(xml_text)
    if not root:
        return f"<book><section iteration=\"{iteration}\"><summary>(Parse error) {xml_text}</summary></section></book>"

    # Debug logging
    print(f"\nDEBUG: Iteration {iteration}")
    print(f"Input XML sections: {len(root.findall('.//section'))}")

    # Use different gathering strategy based on iteration
    if iteration == 1:
        blocks = gather_blocks_with_h1(root, test_mode)
        chunked = chunk_blocks_grouped_by_h1(blocks, FIRST_LEVEL_MAX_CHARS)
    else:
        chunked = gather_blocks_recursive(root)
        print(f"DEBUG: Recursive chunks: {len(chunked)}")
        for i, (ctx, sections) in enumerate(chunked):
            print(f"DEBUG: Chunk {i+1} context: {ctx}")
            print(f"DEBUG: Number of sections: {len(sections)}")

    if not chunked:
        print("WARNING: No chunks generated for this iteration!")
        return xml_text

    # Display preview and get confirmation
    display_batch_preview(chunked, test_mode)

    batch_id = create_batch_with_context(chunked, iteration, dry_run)
    if not dry_run:
        wait_for_batch_completion(batch_id)
        results = retrieve_batch_results(batch_id)
    else:
        # Mock results for dry run
        results = {f"it{iteration}-chunk{i+1}": "(dry run summary)" for i in range(len(chunked))}

    partial_summaries: List[str] = []
    for i in range(len(chunked)):
        cid = f"it{iteration}-chunk{i+1}"
        partial_summaries.append(results.get(cid, ""))

    iteration_xml = build_xml_from_summaries(partial_summaries)
    return iteration_xml

def iterative_summarize_text(
    full_text: str,
    test_mode: bool = False,
    dry_run: bool = False,
    checkpoint_dir: Path = CHECKPOINT_DIR
) -> str:
    """
    Main iterative loop with checkpoint handling.
    """
    existing_ckpts = sorted(checkpoint_dir.glob("iteration_*.xml"))

    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        iteration_num = int(last_ckpt.stem.split("_")[-1]) + 1
        current_text = last_ckpt.read_text(encoding="utf-8")
        rprint(f"[blue]Resuming from iteration {iteration_num-1}[/blue]")
    else:
        iteration_num = 1
        current_text = full_text

    while True:
        root = parse_xml(current_text)
        if not root:
            rprint("[yellow]Parse error => single pass summarization[/yellow]")
            final_xml = summarize_iteration(
                iteration_num,
                current_text,
                test_mode,
                dry_run
            )
            path_ckpt = checkpoint_dir / f"iteration_{iteration_num}.xml"
            path_ckpt.write_text(final_xml, encoding="utf-8")
            return final_xml

        # Get necessary information for both first and recursive levels
        blocks = gather_blocks_with_h1(root, test_mode) if iteration_num == 1 else []
        total_len = sum(len(b[1]) for b in blocks) if iteration_num == 1 else 0
        sections = root.findall(".//section") if iteration_num > 1 else []

        if iteration_num == 1:
            should_continue = total_len > FIRST_LEVEL_MAX_CHARS
        else:
            # Continue if we have more than RECURSIVE_LEVEL_MAX_SECTIONS sections
            section_count = len(sections)
            should_continue = section_count > RECURSIVE_LEVEL_MAX_SECTIONS
            print(f"DEBUG: Found {section_count} sections in iteration {iteration_num}")
            print(f"DEBUG: Should continue? {should_continue}")

        if not should_continue:
            rprint(f"[bold blue]Iteration {iteration_num}:[/bold blue] Final pass...")
            final_xml = summarize_iteration(
                iteration_num,
                current_text,
                test_mode,
                dry_run
            )
            path_ckpt = checkpoint_dir / f"iteration_{iteration_num}.xml"
            path_ckpt.write_text(final_xml, encoding="utf-8")
            return final_xml
        else:
            if iteration_num == 1:
                rprint(f"[bold blue]Iteration {iteration_num}:[/bold blue] Processing {len(blocks)} blocks (~{total_len:,} chars)")
            else:
                rprint(f"[bold blue]Iteration {iteration_num}:[/bold blue] Synthesizing {len(sections)} sections")

            out_xml = summarize_iteration(
                iteration_num,
                current_text,
                test_mode,
                dry_run
            )
            ckpt_path = checkpoint_dir / f"iteration_{iteration_num}.xml"
            ckpt_path.write_text(out_xml, encoding="utf-8")
            iteration_num += 1
            current_text = out_xml

@app.command()
def summarize(
    input_file: Path = typer.Argument(
        Path("output") / "project_lawful.xml",
        help="Path to the input XML file",
        exists=True,
        dir_okay=False,
        readable=True
    ),
    test_mode: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Run in test mode (only process first 4 chapters)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Preview what would be processed without making API calls"
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="Directory for output files and checkpoints"
    ),
    checkpoint_dir: Path = typer.Option(
        Path("output/checkpoints"),
        "--checkpoint-dir",
        "-c",
        help="Directory for checkpoint files"
    ),
):
    """
    Summarize Project Lawful using recursive task decomposition.
    """
    # Validate environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        raise typer.BadParameter("Please set ANTHROPIC_API_KEY in environment.")

    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load and process file
    rprint(f"\n[bold green]Loading[/bold green] {input_file}")
    full_text = input_file.read_text(encoding="utf-8")
    total_chars = len(full_text)
    approx_tokens = int(total_chars / CHARS_PER_TOKEN)

    rprint(f"Loaded {total_chars:,} chars => ~{approx_tokens:,} tokens")
    if test_mode:
        rprint("[yellow]Running in TEST MODE - only processing first 150k words[/yellow]")
    if dry_run:
        rprint("[yellow]Running in DRY RUN mode - no API calls will be made[/yellow]")

    final_summary = iterative_summarize_text(
        full_text,
        test_mode=test_mode,
        dry_run=dry_run,
        checkpoint_dir=checkpoint_dir
    )

    final_path = output_dir / "final_summary.xml"
    final_path.write_text(final_summary, encoding="utf-8")

    rprint("\n[bold green]✓ Summary completed![/bold green]")
    rprint(f"Final summary written to: {final_path}")

if __name__ == "__main__":
    app()
