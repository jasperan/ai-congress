#!/usr/bin/env python3
"""AI Congress CLI - run from project root.

Usage:
    python cli.py              # Interactive menu
    python cli.py chat "prompt" --model phi3:3.8b --stream
    python cli.py models       # List models
    python cli.py pull gemma3  # Pull model
"""

import os
import sys


def main():
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from ai_congress.cli.main import app, interactive_menu

    if len(sys.argv) > 1:
        app()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
