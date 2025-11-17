"""
AI Congress CLI - Command-line interface for LLM swarm
"""
import asyncio
import typer
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator
from ..utils.config_loader import load_config

app = typer.Typer()
console = Console()

# Initialize components
config = load_config()
model_registry = ModelRegistry(config.ollama)
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(model_registry, voting_engine, config.ollama)


@app.callback()
def callback():
    """AI Congress - LLM Swarm with Ensemble Decision Making"""
    pass


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="The prompt to send to the swarm"),
    models: List[str] = typer.Option(["phi3:3.8b", "mistral:7b"], "--model", "-m", help="Models to use in swarm"),
    mode: str = typer.Option("multi_model", "--mode", help="Swarm mode: multi_model, multi_request, hybrid"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature for models"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream responses in real-time"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Interactive chat with LLM swarm"""
    async def run_chat():
        try:
            # Load model weights
            await model_registry.load_benchmark_weights("config/models_benchmark.json")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                disable=not verbose
            ) as progress:
                task = progress.add_task("Querying models...", total=None)

                if mode == "multi_model":
                    result = await swarm.multi_model_swarm(
                        models=models,
                        prompt=prompt,
                        temperature=temperature
                    )
                elif mode == "multi_request":
                    result = await swarm.multi_request_swarm(
                        model=models[0] if models else "mistral:7b",
                        prompt=prompt,
                        temperatures=[0.3, 0.7, 1.0, 1.2]
                    )
                else:
                    result = await swarm.hybrid_swarm(
                        models=models,
                        prompt=prompt,
                        temperatures=[0.5, 0.9],
                        stream=stream
                    )

                progress.update(task, completed=True)

            # Display results
            if result['final_answer'].startswith("Error"):
                console.print(f"[red]{result['final_answer']}[/red]")
                return

            # Show individual responses if verbose
            if verbose:
                table = Table(title="Model Responses")
                table.add_column("Model", style="cyan")
                table.add_column("Response", style="white")

                for response in result.get('responses', []):
                    if response['success']:
                        table.add_row(
                            response['model'],
                            response['response'][:100] + "..." if len(response['response']) > 100 else response['response']
                        )

                console.print(table)

            # Show final answer
            panel = Panel(
                result['final_answer'],
                title=f"[bold green]Final Answer (Confidence: {result.get('confidence', 0):.1%})[/bold green]",
                border_style="green"
            )
            console.print(panel)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    asyncio.run(run_chat())


@app.command()
def models():
    """List available Ollama models"""
    async def list_models():
        try:
            models = await model_registry.list_available_models()

            table = Table(title="Available Models")
            table.add_column("Name", style="cyan")
            table.add_column("Size (GB)", style="magenta")
            table.add_column("Weight", style="yellow")

            for model in models:
                size_gb = f"{model['size'] / (1024**3):.1f}" if model['size'] else "N/A"
                weight = f"{model_registry.get_model_weight(model['name']):.2f}"
                table.add_row(model['name'], size_gb, weight)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error listing models: {e}[/red]")

    asyncio.run(list_models())


@app.command()
def pull(
    model_name: str = typer.Argument(..., help="Name of model to pull")
):
    """Pull a model from Ollama library"""
    async def pull_model():
        try:
            with console.status(f"[bold green]Pulling {model_name}...[/bold green]"):
                success = await model_registry.pull_model(model_name)

            if success:
                console.print(f"[green]Successfully pulled {model_name}[/green]")
            else:
                console.print(f"[red]Failed to pull {model_name}[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    asyncio.run(pull_model())


if __name__ == "__main__":
    app()
