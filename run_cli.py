#!/usr/bin/env python3
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ai_congress.cli.main import app

if __name__ == "__main__":
    app()
