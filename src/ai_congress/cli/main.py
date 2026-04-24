"""
AI Congress CLI - Command-line interface for LLM swarm
"""
import asyncio
import os
import sys
import typer
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.prompt import Prompt, Confirm

import questionary

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator
from ..core.triads import (
    TriadError,
    describe_triad,
    list_triads,
    resolve_triad,
)
from ..utils.config_loader import load_config
from ..tui.theme import PI_THEME, PI_COLORS, EVENT_ICONS
from ..tui.components import (
    dynamic_border,
    startup_banner,
    model_response_panel,
    result_panel,
    status_line,
)

app = typer.Typer()
console = Console(theme=PI_THEME)

# Initialize components
config = load_config()
model_registry = ModelRegistry(config.ollama)
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(
    model_registry, voting_engine, config.ollama,
    openai_config=config.openai,
)


@app.callback()
def callback():
    """AI Congress - LLM Swarm with Ensemble Decision Making"""
    pass


# Move async functions OUT of the commands so they can be reused
async def run_chat_logic(prompt, models, mode, temperature, stream, verbose, personalities, reasoning, inference_backend="ollama", triad=None):
    try:
        # Apply the chosen inference backend
        swarm.inference_backend = inference_backend

        # Load model weights
        await model_registry.load_benchmark_weights("config/models_benchmark.json")

        # --triad is sugar for deliberation mode with the triad's agents
        triad_agents = None
        if triad:
            try:
                available = [
                    m.get("name") for m in await model_registry.list_available_models()
                ]
                triad_agents = resolve_triad(
                    triad,
                    fallback_model=config.agents.base_model,
                    available_models=available or None,
                )
                mode = "deliberation"
                info = describe_triad(triad)
                console.print(
                    f"[pi.accent]triad[/pi.accent] "
                    f"[pi.dim]{triad}[/pi.dim]: {info['description']}"
                )
                console.print(
                    "[pi.dim]members: "
                    + ", ".join(f"{a['role']}@{a['model']}" for a in triad_agents)
                    + "[/pi.dim]"
                )
            except TriadError as exc:
                console.print(f"[pi.error]{exc}[/pi.error]")
                return

        if stream and mode != "deliberation":
            # Streaming mode
            dynamic_border(console, "streaming", style="pi.border")
            console.print(f"[pi.accent]Streaming individual responses...[/pi.accent]")
            await stream_chat(prompt, models, mode, temperature, verbose, personalities, reasoning)
        else:
            # Non-streaming
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                disable=not verbose
            ) as progress:
                task_label = (
                    f"[{PI_COLORS['cyan']}]Deliberating across {len(triad_agents)} members...[/]"
                    if triad_agents else
                    f"[{PI_COLORS['cyan']}]Querying models...[/]"
                )
                task = progress.add_task(task_label, total=None)

                if mode == "deliberation":
                    if triad_agents is None:
                        # Allow --mode deliberation without a triad: build agents
                        # from --model list with neutral role prompts.
                        if not models:
                            console.print("[pi.error]deliberation needs --triad or at least 2 models via -m[/pi.error]")
                            return
                        triad_agents = [
                            {
                                "role": f"member_{i+1}",
                                "name": f"member_{i+1}@{m}",
                                "model": m,
                                "system_prompt": (
                                    f"You are council member {i+1}. Offer an independent analysis "
                                    "of the user's question. Be concrete and flag what you're unsure about."
                                ),
                            }
                            for i, m in enumerate(models)
                        ]
                    result = await swarm.deliberation_swarm(
                        agents=triad_agents,
                        prompt=prompt,
                        temperature=temperature,
                    )
                elif mode == "multi_model":
                    result = await swarm.multi_model_swarm(models=models, prompt=prompt, temperature=temperature, reasoning_mode=reasoning)
                elif mode == "multi_request":
                    result = await swarm.multi_request_swarm(model=models[0] if models else "mistral:7b", prompt=prompt, temperature=temperature)
                elif mode == "hybrid":
                    result = await swarm.hybrid_swarm(models=models, prompt=prompt, temperature=temperature, stream=stream)
                elif mode == "personality":
                    # Load personalities
                    import json
                    if os.path.exists("config/personalities.json"):
                        with open("config/personalities.json", 'r') as f:
                            all_personalities = json.load(f)
                    else:
                        console.print("[pi.error]Error: config/personalities.json not found[/pi.error]")
                        return

                    if personalities:
                         selected = [p for p in all_personalities if p['name'] in personalities]
                    else:
                         selected = all_personalities[:4]

                    result = await swarm.personality_swarm(personalities=selected, prompt=prompt, base_model=config.agents.base_model, temperature=temperature)
                else:
                    console.print(f"[pi.error]Unsupported mode: {mode}[/pi.error]")
                    return

                progress.update(task, completed=True)

            # Display results
            if result.get('final_answer', '').startswith("Error"):
                console.print(f"[pi.error]{result.get('final_answer')}[/pi.error]")
                return

            # Deliberation mode has a richer verdict that LEADS with what the
            # council couldn't resolve -- print it verbatim and return.
            if result.get('mode') == 'deliberation':
                dynamic_border(console, "deliberation verdict", style="pi.border")
                console.print(result.get('verdict') or result.get('final_answer', ''))
                dynamic_border(console, style="pi.border.dim")
                return

            if verbose:
                dynamic_border(console, "model responses", style="pi.border.dim")
                for response in result.get('responses', []):
                    if response['success']:
                        model_response_panel(
                            console,
                            model_name=response['model'],
                            response=response['response'][:200] + "..." if len(response['response']) > 200 else response['response'],
                        )

            # Final answer
            result_panel(
                console,
                answer=result['final_answer'],
                confidence=result.get('confidence', 0),
            )

    except Exception as e:
        console.print(f"[pi.error]Error: {e}[/pi.error]")

async def stream_chat(prompt, models, mode, temperature, verbose, personalities=None, reasoning=None):
    """Stream individual responses in real-time"""
    from rich.live import Live
    # Placeholder -- keep existing stream_chat logic if present.
    # The original was a stub (pass), so provide a basic streaming impl.
    pass

@app.command()
def chat(
    prompt: str = typer.Argument(..., help="The prompt to send to the swarm"),
    models: List[str] = typer.Option(["phi3:3.8b", "mistral:7b"], "--model", "-m", help="Models to use in swarm"),
    personalities: List[str] = typer.Option([], "--personality", "-p", help="Personalities to use in personality mode"),
    mode: str = typer.Option("multi_model", "--mode", help="Swarm mode: multi_model, multi_request, hybrid, personality, deliberation"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature for models"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream responses in real-time"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    reasoning: Optional[str] = typer.Option(None, "--reasoning", "-r", help="Reasoning mode: cot, react"),
    triad: Optional[str] = typer.Option(None, "--triad", help="Run a 3-member deliberation. Use --triad list to see available names"),
):
    """Interactive chat with LLM swarm"""
    if triad == "list":
        names = list_triads()
        console.print("[pi.accent]Available triads:[/pi.accent]")
        for name in names:
            info = describe_triad(name)
            console.print(f"  [pi.dim]{name}[/pi.dim]: {info['description']}")
        return
    asyncio.run(run_chat_logic(prompt, models, mode, temperature, stream, verbose, personalities, reasoning, triad=triad))


@app.command()
def triads():
    """List all available deliberation triads"""
    names = list_triads()
    dynamic_border(console, "deliberation triads", style="pi.border")
    for name in names:
        info = describe_triad(name)
        roles = " + ".join(info["roles"])
        console.print(f"  [{PI_COLORS['cyan']}]{name}[/] [pi.dim]({roles})[/pi.dim]")
        console.print(f"    [pi.dim]{info['description']}[/pi.dim]")
    dynamic_border(console, style="pi.border.dim")

@app.command()
def models():
    """List available models (Ollama + OpenAI if configured)"""
    async def list_models():
        try:
            available = await model_registry.list_available_models()

            dynamic_border(console, "available models", style="pi.border")

            table = Table(
                show_header=True,
                header_style=f"bold {PI_COLORS['md_heading']}",
                border_style=PI_COLORS["dark_gray"],
            )
            table.add_column("Name", style=PI_COLORS["cyan"])
            table.add_column("Backend", style=PI_COLORS["accent"])
            table.add_column("Size (GB)", style=PI_COLORS["thinking_high"], justify="right")
            table.add_column("Weight", style=PI_COLORS["yellow"], justify="right")

            for model in available:
                size_gb = f"{model['size'] / (1024**3):.1f}" if model['size'] else "N/A"
                weight = f"{model_registry.get_model_weight(model['name']):.2f}"
                table.add_row(model['name'], "ollama", size_gb, weight)

            # Show OpenAI model if configured
            if swarm.openai_client is not None:
                table.add_row(
                    config.openai.model or "remote",
                    "openai",
                    "N/A",
                    "N/A",
                )

            console.print(table)

            dynamic_border(console, style="pi.border.dim")
        except Exception as e:
            console.print(f"[pi.error]Error listing models: {e}[/pi.error]")
    asyncio.run(list_models())

@app.command()
def pull(model_name: str = typer.Argument(..., help="Name of model to pull")):
    """Pull a model from Ollama library"""
    async def pull_model():
        try:
            with console.status(
                f"[{PI_COLORS['cyan']}]Pulling {model_name}...[/]"
            ):
                success = await model_registry.pull_model(model_name)
            if success:
                console.print(f"[pi.success]Successfully pulled {model_name}[/pi.success]")
            else:
                console.print(f"[pi.error]Failed to pull {model_name}[/pi.error]")
        except Exception as e:
            console.print(f"[pi.error]Error: {e}[/pi.error]")
    asyncio.run(pull_model())


def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    startup_banner(console)


def interactive_menu():
    while True:
        print_header()

        status_line(console, mode="interactive", models_count=2, temperature=0.7)
        console.print()

        choices = [
            questionary.Choice("Start Swarm Chat", value="chat"),
            questionary.Choice("List Available Models", value="models"),
            questionary.Choice("Pull New Model", value="pull"),
            questionary.Separator(),
            questionary.Choice("Exit", value="exit")
        ]

        choice = questionary.select(
            "Select Activity:",
            choices=choices,
            use_arrow_keys=True
        ).ask()

        if choice == "exit":
            dynamic_border(console, "goodbye", style="pi.border")
            sys.exit(0)

        elif choice == "models":
            models()
            input("\nPress Enter to continue...")

        elif choice == "pull":
            model_name = Prompt.ask(
                f"[{PI_COLORS['accent']}]Enter model name to pull (e.g., gemma3)[/]"
            )
            pull(model_name)
            input("\nPress Enter to continue...")

        elif choice == "chat":
            # Interactive chat config
            dynamic_border(console, "new chat", style="pi.border")
            prompt = Prompt.ask(f"\n[{PI_COLORS['green']}]Enter your prompt[/]")

            # Inference backend selection
            backend_choices = [
                questionary.Choice("Ollama (local models, default)", value="ollama"),
            ]
            if swarm.openai_client is not None:
                openai_label = f"OpenAI-compatible ({config.openai.model or 'remote'})"
                backend_choices.append(
                    questionary.Choice(openai_label, value="openai")
                )
            inference_backend = questionary.select(
                "Select Inference Backend:", choices=backend_choices
            ).ask()

            modes = [
                questionary.Choice("Multi-Model (Default)", value="multi_model"),
                questionary.Choice("Multi-Request (Temperature sampling)", value="multi_request"),
                questionary.Choice("Hybrid (Ensemble)", value="hybrid"),
                questionary.Choice("Personality Swarm", value="personality")
            ]

            mode = questionary.select("Select Swarm Mode:", choices=modes).ask()

            stream = Confirm.ask("Stream responses?", default=True)

            # Pick agent labels based on backend
            if inference_backend == "openai":
                openai_model = config.openai.model or "gpt"
                models_list = [f"{openai_model}:agent-1", f"{openai_model}:agent-2"]
            else:
                models_list = ["phi3:3.8b", "mistral:7b"]
            if mode == "multi_model":
                 # Maybe ask for models?
                 pass

            backend_label = f"{'openai' if inference_backend == 'openai' else 'ollama'}"
            console.print(f"[pi.dim]Running swarm in {mode} mode via {backend_label}...[/pi.dim]")
            dynamic_border(console, "swarm executing", style="pi.border.dim")

            asyncio.run(run_chat_logic(
                prompt=prompt,
                models=models_list,
                mode=mode,
                temperature=0.7,
                stream=stream,
                verbose=True,
                personalities=[],
                reasoning=None,
                inference_backend=inference_backend,
            ))

            input("\nPress Enter to continue...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        app()
    else:
        interactive_menu()
