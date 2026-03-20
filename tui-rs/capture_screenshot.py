#!/usr/bin/env python3
"""Capture tmux pane as PNG screenshot via ansi2html + Playwright."""
import subprocess
import sys
import os

def capture(session_name: str, output_path: str):
    # Capture pane with ANSI escapes
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", session_name, "-ep"],
        capture_output=True, text=True
    )
    ansi_text = result.stdout

    # Convert ANSI to HTML
    from ansi2html import Ansi2HTMLConverter
    conv = Ansi2HTMLConverter(dark_bg=True, font_size="14px", scheme="dracula")
    html = conv.convert(ansi_text)

    # Add custom styling for better look
    html = html.replace("</head>", """
    <style>
        body { margin: 0; padding: 20px; }
        .ansi2html-content {
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            line-height: 1.3;
            padding: 16px;
            border-radius: 8px;
        }
    </style>
    </head>""")

    # Write temp HTML
    html_path = f"/tmp/tui_capture_{os.getpid()}.html"
    with open(html_path, "w") as f:
        f.write(html)

    # Screenshot with Playwright
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 700})
        page.goto(f"file://{html_path}")
        page.wait_for_timeout(500)
        page.screenshot(path=output_path, full_page=False)
        browser.close()

    os.unlink(html_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    capture(sys.argv[1], sys.argv[2])
