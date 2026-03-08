"""
Pi-style dark theme for AI Congress TUI.

Color palette and Rich Theme inspired by pi-coding-agent aesthetics:
dark backgrounds, cyan/blue accents, muted grays, and vibrant highlights.
"""

from rich.theme import Theme

# ── Pi-style color palette ──────────────────────────────────────────────
PI_COLORS = {
    "cyan": "#00d7ff",
    "blue": "#5f87ff",
    "green": "#b5bd68",
    "red": "#cc6666",
    "yellow": "#ffff00",
    "gray": "#808080",
    "dim_gray": "#666666",
    "dark_gray": "#505050",
    "accent": "#8abeb7",
    "md_heading": "#f0c674",
    "md_link": "#81a2be",
    "custom_label": "#9575cd",
    "thinking_high": "#b294bb",
    "thinking_xhigh": "#d183e8",
}

# ── Rich Theme mapping ─────────────────────────────────────────────────
PI_THEME = Theme(
    {
        # Borders & structure
        "pi.border": f"bold {PI_COLORS['cyan']}",
        "pi.border.dim": PI_COLORS["dark_gray"],
        "pi.accent": PI_COLORS["accent"],

        # Semantic
        "pi.success": f"bold {PI_COLORS['green']}",
        "pi.error": f"bold {PI_COLORS['red']}",
        "pi.warning": f"bold {PI_COLORS['yellow']}",
        "pi.muted": PI_COLORS["gray"],
        "pi.dim": PI_COLORS["dim_gray"],

        # Headings & labels
        "pi.heading": f"bold {PI_COLORS['md_heading']}",
        "pi.label": PI_COLORS["custom_label"],
        "pi.link": PI_COLORS["md_link"],

        # Agent / model names
        "pi.agent": f"bold {PI_COLORS['cyan']}",
        "pi.model": f"bold {PI_COLORS['blue']}",

        # Message types
        "pi.msg.user": f"bold {PI_COLORS['green']}",
        "pi.msg.assistant": PI_COLORS["accent"],
        "pi.msg.system": PI_COLORS["dim_gray"],

        # Swarm events
        "pi.event.start": f"bold {PI_COLORS['cyan']}",
        "pi.event.vote": f"bold {PI_COLORS['yellow']}",
        "pi.event.consensus": f"bold {PI_COLORS['green']}",
        "pi.event.error": f"bold {PI_COLORS['red']}",
        "pi.event.thinking": PI_COLORS["thinking_high"],
        "pi.event.thinking_intense": PI_COLORS["thinking_xhigh"],

        # Footer / status bar
        "pi.footer": f"on {PI_COLORS['dark_gray']}",
        "pi.footer.key": f"bold {PI_COLORS['cyan']}",
        "pi.footer.value": PI_COLORS["gray"],

        # Streaming
        "pi.stream.chunk": PI_COLORS["accent"],
        "pi.stream.cursor": f"bold {PI_COLORS['cyan']}",
    }
)

# ── Event icons ─────────────────────────────────────────────────────────
EVENT_ICONS = {
    "model_responding": ">>",
    "model_complete": "<<",
    "voting_started": "[VOTE]",
    "voting_complete": "[RESULT]",
    "consensus_reached": "[OK]",
    "consensus_failed": "[!!]",
    "swarm_started": "[START]",
    "swarm_complete": "[DONE]",
    "error": "[ERR]",
    "warning": "[WARN]",
    "info": "[i]",
    "thinking": "[...]",
    "streaming": "[~]",
}

# ── Message icons ───────────────────────────────────────────────────────
MESSAGE_ICONS = {
    "user": ">",
    "assistant": "<",
    "system": "*",
    "error": "!",
    "debug": "#",
}
