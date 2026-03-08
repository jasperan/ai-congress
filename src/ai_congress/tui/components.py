"""
Pi-style stateless TUI components for AI Congress.

Every function takes a Rich Console as its first argument and renders
directly -- no return values, no retained state.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import EVENT_ICONS, PI_COLORS


# ── Utility ─────────────────────────────────────────────────────────────

def dynamic_border(console: Console, label: str = "", style: str = "pi.border") -> None:
    """Print a horizontal rule spanning the full terminal width."""
    width = console.width or 80
    if label:
        pad = width - len(label) - 4  # 2 spaces + 2 dashes padding
        if pad < 2:
            pad = 2
        left = 2
        right = pad - left
        line = f"{'─' * left} {label} {'─' * right}"
    else:
        line = "─" * width
    console.print(line, style=style)


# ── Startup Banner ──────────────────────────────────────────────────────

def startup_banner(
    console: Console,
    version: str = "0.1.0",
    mode: str = "multi_model",
    models: Optional[List[str]] = None,
    temperature: float = 0.7,
) -> None:
    """Render a pi-style startup header for AI Congress."""
    models = models or []

    cyan = PI_COLORS["cyan"]
    banner_text = Text()
    ascii_lines = [
        "    _    ___    ____                                   ",
        "   / \\  |_ _|  / ___|___  _ __   __ _ _ __ ___  ___ ___",
        "  / _ \\  | |  | |   / _ \\| '_ \\ / _` | '__/ _ \\/ __/ __|",
        " / ___ \\ | |  | |__| (_) | | | | (_| | | |  __/\\__ \\__ \\",
        "/_/   \\_\\___|  \\____\\___/|_| |_|\\__, |_|  \\___||___/___/",
        "                                |___/                    ",
    ]
    for i, line in enumerate(ascii_lines):
        banner_text.append(line, style=cyan)
        if i < len(ascii_lines) - 1:
            banner_text.append("\n")
    banner_lines = [banner_text]

    info_lines = [
        "",
        f"  [{PI_COLORS['dim_gray']}]Multi-Agent LLM Swarm Decision System[/]",
        "",
        f"  [{PI_COLORS['custom_label']}]version[/]  [{PI_COLORS['accent']}]{version}[/]",
        f"  [{PI_COLORS['custom_label']}]mode[/]     [{PI_COLORS['accent']}]{mode}[/]",
        f"  [{PI_COLORS['custom_label']}]temp[/]     [{PI_COLORS['accent']}]{temperature}[/]",
    ]
    if models:
        model_str = ", ".join(models)
        info_lines.append(
            f"  [{PI_COLORS['custom_label']}]models[/]   [{PI_COLORS['accent']}]{model_str}[/]"
        )

    # Add info lines to the banner Text
    info_markup = "\n".join(info_lines)
    banner_text.append("\n")

    # Use a Group to combine Text + markup
    from rich.console import Group
    body = Group(banner_text, info_markup)

    panel = Panel(
        body,
        border_style=PI_COLORS["cyan"],
        padding=(1, 2),
    )
    console.print(panel)


# ── Model Response Panel ────────────────────────────────────────────────

def model_response_panel(
    console: Console,
    model_name: str,
    response: str,
    confidence: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> None:
    """Render a single model's response in a bordered panel."""
    ts = timestamp or datetime.now().strftime("%H:%M:%S")
    conf_str = f"  confidence {confidence:.1%}" if confidence is not None else ""

    title = (
        f"[{PI_COLORS['cyan']}]{EVENT_ICONS['model_complete']}[/] "
        f"[{PI_COLORS['blue']}]{model_name}[/]"
        f"[{PI_COLORS['dim_gray']}]  {ts}{conf_str}[/]"
    )

    panel = Panel(
        response,
        title=title,
        title_align="left",
        border_style=PI_COLORS["dark_gray"],
        padding=(0, 1),
    )
    console.print(panel)


# ── Voting Panel ────────────────────────────────────────────────────────

def voting_panel(
    console: Console,
    votes: Dict[str, Any],
    confidence: float = 0.0,
    method: str = "weighted_majority",
) -> None:
    """Render voting results in a table panel."""
    table = Table(
        show_header=True,
        header_style=f"bold {PI_COLORS['md_heading']}",
        border_style=PI_COLORS["dark_gray"],
        expand=True,
    )
    table.add_column("Model", style=PI_COLORS["cyan"])
    table.add_column("Vote", style=PI_COLORS["accent"])
    table.add_column("Weight", style=PI_COLORS["yellow"], justify="right")

    for model, info in votes.items():
        if isinstance(info, dict):
            vote_val = str(info.get("vote", info.get("response", "N/A")))[:80]
            weight = f"{info.get('weight', 0):.2f}"
        else:
            vote_val = str(info)[:80]
            weight = "—"
        table.add_row(model, vote_val, weight)

    title = (
        f"[{PI_COLORS['yellow']}]{EVENT_ICONS['voting_complete']}[/] "
        f"[{PI_COLORS['md_heading']}]Voting Results[/]"
        f"[{PI_COLORS['dim_gray']}]  method={method}  confidence={confidence:.1%}[/]"
    )

    panel = Panel(table, title=title, title_align="left", border_style=PI_COLORS["yellow"])
    console.print(panel)


# ── Result Panel ────────────────────────────────────────────────────────

def result_panel(
    console: Console,
    answer: str,
    confidence: float = 0.0,
    timestamp: Optional[str] = None,
) -> None:
    """Render the final consensus answer."""
    ts = timestamp or datetime.now().strftime("%H:%M:%S")

    title = (
        f"[{PI_COLORS['green']}]{EVENT_ICONS['consensus_reached']}[/] "
        f"[bold {PI_COLORS['green']}]Final Answer[/]"
        f"[{PI_COLORS['dim_gray']}]  confidence={confidence:.1%}  {ts}[/]"
    )

    panel = Panel(
        answer,
        title=title,
        title_align="left",
        border_style=PI_COLORS["green"],
        padding=(1, 2),
    )
    console.print(panel)


# ── Status Line ─────────────────────────────────────────────────────────

def status_line(
    console: Console,
    mode: str = "multi_model",
    models_count: int = 0,
    temperature: float = 0.7,
) -> None:
    """Compact single-line status display."""
    parts = [
        f"[{PI_COLORS['custom_label']}]mode[/] [{PI_COLORS['accent']}]{mode}[/]",
        f"[{PI_COLORS['custom_label']}]models[/] [{PI_COLORS['accent']}]{models_count}[/]",
        f"[{PI_COLORS['custom_label']}]temp[/] [{PI_COLORS['accent']}]{temperature}[/]",
    ]
    console.print("  ".join(parts))


# ── Footer (for Rich Live) ─────────────────────────────────────────────

def render_footer(
    mode: str = "multi_model",
    models_count: int = 0,
    elapsed_seconds: float = 0.0,
    width: int = 80,
) -> Text:
    """Return a 2-line Text object suitable for Rich Live footer."""
    elapsed = f"{elapsed_seconds:.1f}s"

    line1 = Text()
    line1.append("─" * width, style=PI_COLORS["dark_gray"])

    line2 = Text()
    line2.append(" mode ", style=f"bold {PI_COLORS['cyan']}")
    line2.append(f"{mode} ", style=PI_COLORS["gray"])
    line2.append(" models ", style=f"bold {PI_COLORS['cyan']}")
    line2.append(f"{models_count} ", style=PI_COLORS["gray"])
    line2.append(" elapsed ", style=f"bold {PI_COLORS['cyan']}")
    line2.append(elapsed, style=PI_COLORS["gray"])

    # Pad line2 to width
    pad = width - len(line2.plain)
    if pad > 0:
        line2.append(" " * pad)

    result = Text()
    result.append_text(line1)
    result.append("\n")
    result.append_text(line2)
    return result
