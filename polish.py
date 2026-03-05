"""
Final formatting pass over OCR output text files.
Run after engine.py to normalize page headers, collapse excess blank lines,
and trim whitespace. Reads from output_text (or given folder), writes back
or to a separate folder.
"""
import os
import re
import argparse


def format_document(content: str) -> str:
    """
    Apply consistent formatting: page headers, blank lines, trim.
    """
    if not content or not content.strip():
        return content

    lines = content.splitlines()
    out = []
    i = 0
    prev_blank = False
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        # Normalize page header: "--- PAGE N ---" with exactly one blank line after
        if re.match(r"^---\s*PAGE\s+\d+\s*---\s*$", stripped):
            # Ensure we don't double blank before header
            while out and out[-1].strip() == "":
                out.pop()
            out.append("--- PAGE %s ---" % re.search(r"\d+", stripped).group())
            out.append("")
            prev_blank = True
            i += 1
            continue

        is_blank = not stripped
        if is_blank:
            # At most one blank line in a row
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(stripped)
            prev_blank = False
        i += 1

    # Trim leading/trailing blank lines
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()

    # Ensure file ends with single newline
    return "\n".join(out) + "\n" if out else ""


def process_folder(input_folder: str, output_folder: str = None) -> None:
    """
    Format all .txt files in input_folder. If output_folder is None, overwrite in place.
    """
    if output_folder is None:
        output_folder = input_folder
    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".txt")]
    if not txt_files:
        print("No .txt files in", input_folder)
        return

    for name in sorted(txt_files):
        path = os.path.join(input_folder, name)
        out_path = os.path.join(output_folder, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            formatted = format_document(content)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(formatted)
            print("[OK]", name)
        except Exception as e:
            print("[Error]", name, ":", e)


def main():
    parser = argparse.ArgumentParser(
        description="Polish OCR output: normalize page headers and blank lines."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="output_text",
        help="Folder containing .txt files (default: output_text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output folder (default: overwrite input files)",
    )
    args = parser.parse_args()
    process_folder(args.input_dir, args.output)


if __name__ == "__main__":
    main()
