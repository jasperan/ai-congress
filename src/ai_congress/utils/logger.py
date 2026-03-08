"""
Rich Logger for AI Congress - Pi-style Command Center Console Output
"""
import logging
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler

from ..tui.theme import PI_THEME, PI_COLORS, EVENT_ICONS
from ..tui.components import dynamic_border

# Global Rich console with pi-style theme
console = Console(theme=PI_THEME, highlighter=NullHighlighter(), stderr=True)


def get_rich_logger(name: str = "ai_congress", level: int = logging.INFO) -> logging.Logger:
    """Get a logger with Rich formatting"""
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Use RichHandler for structured logging
    handler = RichHandler(console=console, rich_tracebacks=True, show_time=False, show_level=False)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def debug_action(action: str, entity: str, message: str, verbosity: int = 2):
    """
    Print debug action in pi-style command center format

    Format: [DBG][datetime][ACTION][entity] message
    verbosity: 0=minimal, 1=normal, 2=verbose, 3=debug
    """
    if verbosity >= 3:  # Only show debug if verbosity is 3
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = f"[pi.dim][DBG][/pi.dim][pi.muted][{timestamp}][/pi.muted]"
        action_tag = f"[pi.accent][{action.upper()}][/pi.accent]"
        entity_tag = f"[pi.agent][{entity}][/pi.agent]"

        # Use different styles for different actions
        if "GENERATED" in action.upper():
            console.print(f"{tag}{action_tag}{entity_tag} {message}", style="pi.success")
        elif "ERROR" in action.upper() or "FAILED" in action.upper():
            console.print(f"{tag}{action_tag}{entity_tag} {message}", style="pi.error")
        elif "VOTING" in action.upper():
            console.print(f"{tag}{action_tag}{entity_tag} {message}", style="pi.warning")
        else:
            console.print(f"{tag}{action_tag}{entity_tag} {message}", style="pi.dim")


def swarm_status_panel(title: str, items: list, headers: list = ["Entity", "Status"], verbosity: int = 2):
    """Display swarm status in a pi-style Rich panel with table"""
    if verbosity >= 1:
        dynamic_border(console, EVENT_ICONS["info"], style="pi.border.dim")

        table = Table(
            show_header=True,
            header_style=f"bold {PI_COLORS['md_heading']}",
            border_style=PI_COLORS["dark_gray"],
        )
        for header in headers:
            table.add_column(header, style=PI_COLORS["accent"])

        for item in items:
            table.add_row(*item)

        panel = Panel(
            table,
            title=f"[pi.heading]{title}[/pi.heading]",
            border_style=PI_COLORS["cyan"],
        )
        console.print(panel)

        dynamic_border(console, style="pi.border.dim")


def info_message(action: str, entity: str, message: str, verbosity: int = 2):
    """Print info message in pi-style formatted output"""
    if verbosity >= 1:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(
            f"[pi.success]{EVENT_ICONS['info']}[/pi.success] "
            f"[pi.muted][{timestamp}][/pi.muted]"
            f"[pi.accent][{action.upper()}][/pi.accent]"
            f"[pi.agent][{entity}][/pi.agent] "
            f"[pi.success]{message}[/pi.success]"
        )


def error_message(action: str, entity: str, message: str, verbosity: int = 0):
    """Print error message (always shown) in pi-style"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[pi.error]{EVENT_ICONS['error']}[/pi.error] "
        f"[pi.muted][{timestamp}][/pi.muted]"
        f"[pi.accent][{action.upper()}][/pi.accent]"
        f"[pi.agent][{entity}][/pi.agent] "
        f"[pi.error]{message}[/pi.error]"
    )


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for logging to avoid clogging"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
