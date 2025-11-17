"""
CLI Main - Beautiful command-line interface with Rich
"""
import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
import yaml

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator

console = Console()


def load_config():
    """Load configuration from YAML"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return {}


@click.group()
def cli():
    """AI Congress - LLM Swarm with Ensemble Decision Making"""
    pass


@cli.command()
@click.option('--mode', default='multi_model', help='Swarm mode: multi_model, multi_request, hybrid')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
def chat(mode, verbose):
    """Interactive chat with LLM swarm"""

    console.print(Panel.fit(
        "[bold cyan]AI Congress - Interactive Chat[/bold cyan]\n"
        "Type your questions and see the swarm decide!\n"
        "Commands: /models, /mode, /quit",
        border_style="cyan"
    ))

    # Initialize components
    config = load_config()
    registry = ModelRegistry()
    voting = VotingEngine()
    swarm = SwarmOrchestrator(registry, voting)

    # List models
    with console.status("[bold green]Loading models..."):
        models = asyncio.run(registry.list_available_models())

    console.print(f"\n[green]✓[/green] Found {len(models)} models\n")

    # Model selection
    selected_models = select_models_interactive(models)

    if not selected_models:
        console.print("[red]No models selected. Exiting.[/red]")
        return

    console.print(f"\n[cyan]Selected models:[/cyan] {', '.join(selected_models)}\n")

    # Chat loop
    while True:
        try:
            prompt = Prompt.ask("\n[bold blue]You[/bold blue]")

            if not prompt or prompt == "/quit":
                break

            if prompt == "/models":
                show_models_table(models, registry)
                continue

            if prompt == "/mode":
                mode = Prompt.ask(
                    "Select mode",
                    choices=["multi_model", "multi_request", "hybrid"],
                    default=mode
                )
                console.print(f"[green]Mode set to: {mode}[/green]")
                continue

            # Process with swarm
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Querying {len(selected_models)} models...",
                    total=None
                )

                if mode == "multi_model":
                    result = asyncio.run(
                        swarm.multi_model_swarm(selected_models, prompt)
                    )
                else:
                    result = asyncio.run(
                        swarm.multi_request_swarm(selected_models[0], prompt)
                    )

            # Display results
            display_swarm_results(result, verbose)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def select_models_interactive(models):
    """Interactive model selection"""
    console.print("[cyan]Select models for the swarm:[/cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model Name", style="cyan")
    table.add_column("Size", justify="right")

    for i, model in enumerate(models, 1):
        size_gb = model.get('size', 0) / (1024**3)
        table.add_row(str(i), model['name'], f"{size_gb:.1f} GB")

    console.print(table)

    selection = Prompt.ask(
        "\nEnter model numbers (comma-separated) or 'all'",
        default="1,2,3"
    )

    if selection.lower() == 'all':
        return [m['name'] for m in models]

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        return [models[i]['name'] for i in indices if 0 <= i < len(models)]
    except:
        console.print("[red]Invalid selection[/red]")
        return []


def display_swarm_results(result, verbose):
    """Display swarm results with Rich formatting"""

    # Show individual model responses if verbose
    if verbose >= 2:
        console.print("\n[bold yellow]━━━ Model Responses ━━━[/bold yellow]")
        for resp in result.get('responses', []):
            if resp['success']:
                console.print(Panel(
                    resp['response'],
                    title=f"[cyan]{resp['model']}[/cyan]",
                    border_style="blue"
                ))

    # Show voting breakdown
    if verbose >= 1:
        vote_breakdown = result.get('vote_breakdown', {})
        if vote_breakdown:
            console.print("\n[bold yellow]━━━ Vote Breakdown ━━━[/bold yellow]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Response", style="cyan")
            table.add_column("Weight", justify="right")
            table.add_column("Models", style="dim")

            for resp_key, details in vote_breakdown.items():
                models_str = ", ".join(details.get('models', []))
                table.add_row(
                    details['original'][:50] + "...",
                    f"{details['weight']:.2f}",
                    models_str
                )

            console.print(table)

    # Show final answer
    console.print("\n[bold green]━━━ Congress Decision ━━━[/bold green]")
    console.print(Panel(
        result['final_answer'],
        title=f"[green]Final Answer[/green] (Confidence: {result.get('confidence', 0):.0%})",
        border_style="green",
        padding=(1, 2)
    ))


def show_models_table(models, registry):
    """Show available models in a table"""
    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Weight", justify="right", style="yellow")

    for model in models:
        size_gb = model.get('size', 0) / (1024**3)
        weight = registry.get_model_weight(model['name'])
        table.add_row(
            model['name'],
            f"{size_gb:.1f} GB",
            f"{weight:.2f}"
        )

    console.print(table)


@cli.command()
def models():
    """List available models"""
    registry = ModelRegistry()
    models = asyncio.run(registry.list_available_models())
    show_models_table(models, registry)


@cli.command()
@click.argument('model_name')
def pull(model_name):
    """Pull a new model from Ollama"""
    console.print(f"[cyan]Pulling model: {model_name}[/cyan]")

    with console.status(f"[bold green]Downloading {model_name}..."):
        registry = ModelRegistry()
        success = asyncio.run(registry.pull_model(model_name))

    if success:
        console.print(f"[green]✓ Model {model_name} pulled successfully[/green]")
    else:
        console.print(f"[red]✗ Failed to pull {model_name}[/red]")


if __name__ == '__main__':
    cli()
