#!/usr/bin/env python3
"""Sync corrected all_transcriptions.txt back to individual text files.

After you edit the combined file, run this to split it back:
    python3 sync_corrections.py

Then run the data preparation pipeline:
    python3 prepare_call_center_data.py
"""

import re
from pathlib import Path

COMBINED = Path("/root/stt/call_center_output/all_transcriptions.txt")
TEXTS_DIR = Path("/root/stt/call_center_output/texts")
RAW_DIR = Path("/root/stt/call_center_output/raw_responses")

def main():
    content = COMBINED.read_text(encoding="utf-8")

    # Split by file headers
    pattern = r"={80}\nFILE: (.+?)\.mp3\s+\|.*?\n={80}\n"
    parts = re.split(pattern, content)

    # parts[0] is before first header (empty), then alternating: stem, text, stem, text...
    count = 0
    for i in range(1, len(parts), 2):
        stem = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if not stem or not text:
            continue

        txt_path = TEXTS_DIR / f"{stem}.txt"
        txt_path.write_text(text, encoding="utf-8")
        count += 1

    print(f"Synced {count} text files to {TEXTS_DIR}")

    # Also update the raw JSON responses with corrected text
    # (prepare_call_center_data.py uses raw JSONs for word timestamps,
    #  but having corrected text available is useful for reference)
    print(f"\nNow run:")
    print(f"  python3 prepare_call_center_data.py")


if __name__ == "__main__":
    main()
