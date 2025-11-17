
# Create additional critical implementation files

# 1. SwarmOrchestrator example
swarm_orchestrator_code = '''"""
Swarm Orchestrator - Coordinates concurrent LLM requests and aggregates responses
"""
import asyncio
from typing import List, Dict, Optional
import ollama
from .voting_engine import VotingEngine
from .model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrates LLM swarm with concurrent model querying"""
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        voting_engine: VotingEngine,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.model_registry = model_registry
        self.voting_engine = voting_engine
        self.ollama_client = ollama.AsyncClient(host=ollama_base_url)
        self.max_concurrent = 10
    
    async def query_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """Query a single model asynchronously"""
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            logger.debug(f"Querying {model_name} with temperature {temperature}")
            
            response = await self.ollama_client.chat(
                model=model_name,
                messages=messages,
                options={'temperature': temperature}
            )
            
            return {
                'model': model_name,
                'response': response['message']['content'],
                'temperature': temperature,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error querying {model_name}: {e}")
            return {
                'model': model_name,
                'response': '',
                'temperature': temperature,
                'success': False,
                'error': str(e)
            }
    
    async def multi_model_swarm(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        Query multiple different models concurrently
        
        Returns:
            {
                'responses': List[Dict],
                'final_answer': str,
                'confidence': float,
                'vote_breakdown': Dict
            }
        """
        logger.info(f"Starting multi-model swarm with {len(models)} models")
        
        # Create concurrent tasks
        tasks = [
            self.query_model(model, prompt, temperature, system_prompt)
            for model in models
        ]
        
        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)
        
        # Filter successful responses
        successful = [r for r in responses if r['success']]
        
        if not successful:
            logger.error("No successful responses from swarm")
            return {
                'responses': responses,
                'final_answer': "Error: No models responded successfully",
                'confidence': 0.0,
                'vote_breakdown': {}
            }
        
        # Get model weights
        weights = [
            self.model_registry.get_model_weight(r['model']) 
            for r in successful
        ]
        
        # Vote on best response
        texts = [r['response'] for r in successful]
        model_names = [r['model'] for r in successful]
        
        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, model_names
        )
        
        logger.info(f"Multi-model swarm completed. Confidence: {confidence:.2f}")
        
        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'vote_breakdown': vote_breakdown,
            'models_used': model_names,
            'weights': weights
        }
    
    async def multi_request_swarm(
        self,
        model: str,
        prompt: str,
        temperatures: List[float] = [0.3, 0.7, 1.0, 1.2],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Query same model multiple times with different temperatures
        
        Returns similar structure to multi_model_swarm
        """
        logger.info(f"Starting multi-request swarm with model {model}, {len(temperatures)} temperatures")
        
        # Create concurrent tasks with different temperatures
        tasks = [
            self.query_model(model, prompt, temp, system_prompt)
            for temp in temperatures
        ]
        
        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)
        
        # Filter successful
        successful = [r for r in responses if r['success']]
        
        if not successful:
            return {
                'responses': responses,
                'final_answer': "Error: No successful responses",
                'confidence': 0.0
            }
        
        # Use temperature-based weighting
        texts = [r['response'] for r in successful]
        temps = [r['temperature'] for r in successful]
        
        final_answer = self.voting_engine.temperature_ensemble(texts, temps)
        
        return {
            'responses': responses,
            'final_answer': final_answer,
            'temperatures_used': temps,
            'model': model
        }
    
    async def hybrid_swarm(
        self,
        models: List[str],
        prompt: str,
        temperatures: List[float] = [0.5, 0.9],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Hybrid: Multiple models, each queried with multiple temperatures
        
        This is the most comprehensive swarm mode
        """
        logger.info(f"Starting hybrid swarm: {len(models)} models x {len(temperatures)} temps")
        
        # Create all combinations
        tasks = []
        for model in models:
            for temp in temperatures:
                tasks.append(self.query_model(model, prompt, temp, system_prompt))
        
        # Execute all concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def limited_query(task):
            async with semaphore:
                return await task
        
        responses = await asyncio.gather(*[limited_query(task) for task in tasks])
        
        # Process results
        successful = [r for r in responses if r['success']]
        
        if not successful:
            return {
                'responses': responses,
                'final_answer': "Error: No successful responses",
                'confidence': 0.0
            }
        
        # Weight by both model performance and temperature
        weights = []
        for r in successful:
            model_weight = self.model_registry.get_model_weight(r['model'])
            temp_weight = 1.0 / (r['temperature'] + 0.1)  # Lower temp = higher weight
            combined_weight = model_weight * temp_weight
            weights.append(combined_weight)
        
        texts = [r['response'] for r in successful]
        model_names = [f"{r['model']}@{r['temperature']}" for r in successful]
        
        final_answer, confidence, vote_breakdown = self.voting_engine.weighted_majority_vote(
            texts, weights, model_names
        )
        
        logger.info(f"Hybrid swarm completed. Confidence: {confidence:.2f}")
        
        return {
            'responses': responses,
            'final_answer': final_answer,
            'confidence': confidence,
            'vote_breakdown': vote_breakdown,
            'total_queries': len(tasks),
            'successful_queries': len(successful)
        }
    
    async def stream_swarm_response(
        self,
        models: List[str],
        prompt: str,
        websocket = None
    ):
        """
        Stream responses from swarm in real-time
        Useful for WebSocket connections
        """
        async def stream_model(model_name: str):
            try:
                messages = [{'role': 'user', 'content': prompt}]
                
                async for chunk in await self.ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                ):
                    content = chunk['message']['content']
                    
                    if websocket:
                        await websocket.send_json({
                            'type': 'model_chunk',
                            'model': model_name,
                            'content': content
                        })
                    
                    yield content
                    
            except Exception as e:
                logger.error(f"Streaming error for {model_name}: {e}")
        
        # Stream from all models concurrently
        # Implementation depends on specific WebSocket library
        pass
'''

with open('swarm_orchestrator_example.py', 'w') as f:
    f.write(swarm_orchestrator_code)

# 2. FastAPI main.py example
fastapi_main_code = '''"""
FastAPI Main Application
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio
import logging
from pydantic import BaseModel

from ..core.model_registry import ModelRegistry
from ..core.voting_engine import VotingEngine
from ..core.swarm_orchestrator import SwarmOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Congress API",
    description="LLM Swarm with Ensemble Decision Making",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry()
voting_engine = VotingEngine()
swarm = SwarmOrchestrator(model_registry, voting_engine)


# Pydantic models
class ChatRequest(BaseModel):
    prompt: str
    models: List[str]
    mode: str = "multi_model"  # multi_model, multi_request, hybrid
    temperature: float = 0.7
    temperatures: Optional[List[float]] = None
    system_prompt: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    size: int
    weight: float


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting AI Congress API...")
    await model_registry.list_available_models()
    await model_registry.load_benchmark_weights("config/models_benchmark.json")
    logger.info("Startup complete")


@app.get("/")
async def root():
    return {"message": "AI Congress API", "status": "running"}


@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available Ollama models"""
    models = await model_registry.list_available_models()
    
    return [
        ModelInfo(
            name=m['name'],
            size=m.get('size', 0),
            weight=model_registry.get_model_weight(m['name'])
        )
        for m in models
    ]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat request through swarm"""
    try:
        if request.mode == "multi_model":
            result = await swarm.multi_model_swarm(
                models=request.models,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature
            )
        elif request.mode == "multi_request":
            temps = request.temperatures or [0.3, 0.7, 1.0]
            result = await swarm.multi_request_swarm(
                model=request.models[0] if request.models else "mistral:7b",
                prompt=request.prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        elif request.mode == "hybrid":
            temps = request.temperatures or [0.5, 0.9]
            result = await swarm.hybrid_swarm(
                models=request.models,
                prompt=request.prompt,
                temperatures=temps,
                system_prompt=request.system_prompt
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            prompt = data.get('prompt')
            models = data.get('models', ['mistral:7b'])
            mode = data.get('mode', 'multi_model')
            
            # Send acknowledgment
            await websocket.send_json({
                'type': 'start',
                'message': f'Processing with {len(models)} models...'
            })
            
            # Process swarm request
            if mode == "multi_model":
                result = await swarm.multi_model_swarm(
                    models=models,
                    prompt=prompt
                )
            else:
                result = await swarm.multi_request_swarm(
                    model=models[0],
                    prompt=prompt
                )
            
            # Send individual model responses
            for response in result.get('responses', []):
                if response['success']:
                    await websocket.send_json({
                        'type': 'model_response',
                        'model': response['model'],
                        'content': response['response']
                    })
                    await asyncio.sleep(0.1)  # Small delay for readability
            
            # Send final answer
            await websocket.send_json({
                'type': 'final_answer',
                'content': result['final_answer'],
                'confidence': result.get('confidence', 0),
                'vote_breakdown': result.get('vote_breakdown', {})
            })
            
            await websocket.send_json({'type': 'end'})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })


@app.post("/api/models/pull/{model_name}")
async def pull_model(model_name: str):
    """Pull a new model from Ollama"""
    success = await model_registry.pull_model(model_name)
    if success:
        return {"message": f"Model {model_name} pulled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to pull model")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
'''

with open('fastapi_main_example.py', 'w') as f:
    f.write(fastapi_main_code)

# 3. CLI main.py example with Rich
cli_main_code = '''"""
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
        "[bold cyan]AI Congress - Interactive Chat[/bold cyan]\\n"
        "Type your questions and see the swarm decide!\\n"
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
    
    console.print(f"\\n[green]✓[/green] Found {len(models)} models\\n")
    
    # Model selection
    selected_models = select_models_interactive(models)
    
    if not selected_models:
        console.print("[red]No models selected. Exiting.[/red]")
        return
    
    console.print(f"\\n[cyan]Selected models:[/cyan] {', '.join(selected_models)}\\n")
    
    # Chat loop
    while True:
        try:
            prompt = Prompt.ask("\\n[bold blue]You[/bold blue]")
            
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
            console.print("\\n[yellow]Interrupted[/yellow]")
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
        "\\nEnter model numbers (comma-separated) or 'all'",
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
        console.print("\\n[bold yellow]━━━ Model Responses ━━━[/bold yellow]")
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
            console.print("\\n[bold yellow]━━━ Vote Breakdown ━━━[/bold yellow]")
            
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
    console.print("\\n[bold green]━━━ Congress Decision ━━━[/bold green]")
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
'''

with open('cli_main_example.py', 'w') as f:
    f.write(cli_main_code)

print("✅ Created swarm_orchestrator_example.py")
print("✅ Created fastapi_main_example.py")
print("✅ Created cli_main_example.py")
print("\n🎯 All core implementation examples ready!")
