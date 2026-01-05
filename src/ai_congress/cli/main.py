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
    personalities: List[str] = typer.Option([], "--personality", "-p", help="Personalities to use in personality mode"),
    mode: str = typer.Option("multi_model", "--mode", help="Swarm mode: multi_model, multi_request, hybrid, personality"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature for models"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream responses in real-time"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    reasoning: Optional[str] = typer.Option(None, "--reasoning", "-r", help="Reasoning mode: cot, react")
):
    """Interactive chat with LLM swarm"""
    async def run_chat():
        try:
            # Load model weights
            await model_registry.load_benchmark_weights("config/models_benchmark.json")

            if stream:
                # Streaming mode: Show individual responses as they complete
                console.print("[bold blue]Streaming individual responses...[/bold blue]")
                await stream_chat(prompt, models, mode, temperature, verbose, personalities, reasoning)
            else:
                # Non-streaming: Show progress and final result
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
                            temperature=temperature,
                            reasoning_mode=reasoning
                        )
                    elif mode == "multi_request":
                        result = await swarm.multi_request_swarm(
                            model=models[0] if models else "mistral:7b",
                            prompt=prompt,
                            temperatures=[0.3, 0.7, 1.0, 1.2]
                        )
                    elif mode == "hybrid":
                        result = await swarm.hybrid_swarm(
                            models=models,
                            prompt=prompt,
                            temperatures=[0.5, 0.9],
                            stream=stream
                        )
                    elif mode == "personality":
                        # Load personalities
                        import json
                        import os
                        if os.path.exists("config/personalities.json"):
                            with open("config/personalities.json", 'r') as f:
                                all_personalities = json.load(f)
                        else:
                            console.print("[red]Error: config/personalities.json not found[/red]")
                            return

                        if personalities:
                            selected_personalities = [p for p in all_personalities if p['name'] in personalities]
                        else:
                            selected_personalities = all_personalities[:4]  # Default first 4

                        result = await swarm.personality_swarm(
                            personalities=selected_personalities,
                            prompt=prompt,
                            base_model=config.agents.base_model,
                            temperature=temperature
                        )
                    else:
                        console.print(f"[red]Unsupported mode: {mode}[/red]")
                        return

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

    async def stream_chat(prompt, models, mode, temperature, verbose, personalities=None, reasoning=None):
        """Stream individual responses in real-time"""
        import asyncio
        from rich.live import Live

        if mode == "personality":
            # Load personalities
            import json
            import os
            if os.path.exists("config/personalities.json"):
                with open("config/personalities.json", 'r') as f:
                    all_personalities = json.load(f)
            else:
                console.print("[red]Error: config/personalities.json not found[/red]")
                return

            if personalities:
                selected_personalities = [p for p in all_personalities if p['name'] in personalities]
            else:
                selected_personalities = all_personalities[:4]  # Default first 4

            # Create live table
            table = Table(title="Personality Swarm Status")
            table.add_column("Personality", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Response Preview", style="white")

            status_dict = {}
            for p in selected_personalities:
                table.add_row(p['name'], "Queued", "")
                status_dict[p['name']] = {"status": "Queued", "preview": ""}

            def update_table(event_type, entity_name, content=None, full_response=None):
                if event_type == 'start':
                    status_dict[entity_name]["status"] = "Generating..."
                elif event_type == 'chunk':
                    status_dict[entity_name]["preview"] += content
                    status_dict[entity_name]["preview"] = status_dict[entity_name]["preview"][:100]  # Limit preview
                elif event_type == 'complete':
                    status_dict[entity_name]["status"] = "Complete"
                    status_dict[entity_name]["preview"] = full_response[:100] + "..." if len(full_response) > 100 else full_response

                # Rebuild table
                table.rows.clear()
                for name, data in status_dict.items():
                    table.add_row(name, data["status"], data["preview"])

            with Live(console, refresh_per_second=4) as live:
                live.update(table)
                result = await swarm.personality_swarm(
                    personalities=selected_personalities,
                    prompt=prompt,
                    base_model=config.agents.base_model,
                    temperature=temperature,
                    stream=True,
                    update_callback=update_table
                )

            # After live, show final result
            if result['final_answer'].startswith("Error"):
                console.print(f"[red]{result['final_answer']}[/red]")
                return

            panel = Panel(
                result['final_answer'],
                title=f"[bold green]Final Congress Decision (Confidence: {result.get('confidence', 0):.1%})[/bold green]",
                border_style="green"
            )
            console.print(panel)
            return

        # Other modes
        tasks = {}
        task_to_entity = {}

        if mode == "multi_model":
            for model in models:
                task = swarm.query_model(model, prompt, temperature, reasoning_mode=reasoning)
                tasks[task] = model
                task_to_entity[task] = model
        elif mode == "multi_request":
            base_model = models[0] if models else "mistral:7b"
            temperatures = [0.3, 0.7, 1.0, 1.2]
            for temp in temperatures:
                task = swarm.query_model(base_model, prompt, temp)
                tasks[task] = f"{base_model}@{temp}"
                task_to_entity[task] = f"{base_model}@{temp}"
        else:  # hybrid
            hybrid_models = models or await swarm.model_registry.get_top_models(2)
            temperatures = [0.5, 0.9]
            for model in hybrid_models:
                for temp in temperatures:
                    task = swarm.query_model(model, prompt, temp)
                    tasks[task] = f"{model}@{temp}"
                    task_to_entity[task] = f"{model}@{temp}"

        # Execute concurrently and stream as completed
        responses = []
        completed_count = 0
        total_tasks = len(tasks)

        for coro in asyncio.as_completed(tasks.keys()):
            response = await coro
            completed_count += 1
            entity_name = task_to_entity[coro]

            if response['success']:
                # Show individual response
                panel = Panel(
                    response['response'][:200] + "..." if len(response['response']) > 200 else response['response'],
                    title=f"[cyan]{entity_name}[/cyan]",
                    border_style="cyan"
                )
                console.print(panel)
                console.print(f"[dim]Completed {completed_count}/{total_tasks}[/dim]\n")
                responses.append(response)
            else:
                console.print(f"[red]{entity_name}: Error - {response.get('error', 'Unknown')}[/red]")

        # Now compute final answer using the responses
        if responses:
            successful = responses
            if mode == "multi_model":
                weights = [model_registry.get_model_weight(r['model']) for r in successful]
                texts = [r['response'] for r in successful]
                model_names = [r['model'] for r in successful]
            elif mode == "multi_request":
                weights = [1.0 / (r['temperature'] + 0.1) for r in successful]
                texts = [r['response'] for r in successful]
                model_names = [f"{base_model}@{r['temperature']}" for r in successful]
            else:
                weights = [model_registry.get_model_weight(r['model']) * (1.0 / (r['temperature'] + 0.1)) for r in successful]
                texts = [r['response'] for r in successful]
                model_names = [f"{r['model']}@{r['temperature']}" for r in successful]

            final_answer, confidence, vote_breakdown = swarm.voting_engine.weighted_majority_vote(
                texts, weights, model_names
            )

            # Compute semantic confidence
            semantic_confidence = await swarm.semantic_confidence(texts, model_names, prompt)

            # Show final answer
            panel = Panel(
                final_answer,
                title=f"[bold green]Final Congress Decision (Confidence: {confidence:.1%}, Semantic: {semantic_confidence:.1%})[/bold green]",
                border_style="green"
            )
            console.print(panel)

            if verbose:
                console.print(f"\n[dim]Vote breakdown: {vote_breakdown}[/dim]")
        else:
            console.print("[red]No successful responses from any model.[/red]")

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
