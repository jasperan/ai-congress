"""
Rich Logger for AI Congress - Command Center Style Console Output
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

# Global Rich console
console = Console(highlighter=NullHighlighter(), stderr=True)


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
    Print debug action in command center format

    Format: [DBG][datetime][ACTION][entity] message
    verbosity: 0=minimal, 1=normal, 2=verbose, 3=debug
    """
    if verbosity >= 3:  # Only show debug if verbosity is 3
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[bold cyan][DBG][/bold cyan][{timestamp}][{action.upper()}][{entity}] {message}"

        # Use different colors for different actions
        if "GENERATED" in action.upper():
            console.print(formatted, style="green")
        elif "ERROR" in action.upper() or "FAILED" in action.upper():
            console.print(formatted, style="red")
        elif "VOTING" in action.upper():
            console.print(formatted, style="yellow")
        else:
            console.print(formatted, style="dim")


def swarm_status_panel(title: str, items: list, headers: list = ["Entity", "Status"], verbosity: int = 2):
    """Display swarm status in a Rich panel with table"""
    if verbosity >= 1:
        table = Table(show_header=True, header_style="bold magenta")
        for header in headers:
            table.add_column(header)

        for item in items:
            table.add_row(*item)

        panel = Panel(table, title=f"[bold blue]{title}[/bold blue]", border_style="blue")
        console.print(panel)


def info_message(action: str, entity: str, message: str, verbosity: int = 2):
    """Print info message in formatted style"""
    if verbosity >= 1:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[bold green][INFO][/bold green][{timestamp}][{action.upper()}][{entity}] {message}"
        console.print(formatted, style="green")


def error_message(action: str, entity: str, message: str, verbosity: int = 0):
    """Print error message (always shown)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[bold red][ERR][/bold red][{timestamp}][{action.upper()}][{entity}] {message}"
    console.print(formatted, style="red")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for logging to avoid clogging"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
