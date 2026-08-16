"""
API request/response schemas (pydantic models).
"""
from typing import List, Optional, Dict
from pydantic import BaseModel

from ..utils.config_loader import load_config

config = load_config()


class Personality(BaseModel):
    name: str
    system_prompt: str


class PersonalityCreate(BaseModel):
    name: str
    system_prompt: str


class ChatRequest(BaseModel):
    prompt: str
    models: List[str]
    mode: str = config.swarm.default_mode
    temperature: float = 0.7
    temperatures: Optional[List[float]] = None
    system_prompt: Optional[str] = None
    personalities: Optional[List[Personality]] = None  # For personality mode
    history: Optional[List[Dict[str, str]]] = None  # Conversation history for context
    use_rag: bool = False  # Enable RAG
    search_web: bool = False  # Enable web search
    document_ids: Optional[List[str]] = None  # Specific documents for RAG
    voting_mode: str = "classic"  # classic | semantic
    inference_backend: str = "ollama"  # ollama | openai


class EnhancedChatRequest(BaseModel):
    prompt: str
    models: List[str]
    temperature: float = 0.7
    enable_decomposition: bool = True
    enable_debate: bool = True
    use_rag: bool = False
    document_ids: Optional[List[str]] = None
    search_web: bool = False


class FeedbackRequest(BaseModel):
    session_id: str
    model: str
    feedback: str  # "positive" or "negative"
    response_text: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    size: int
    weight: float
    backend: str = "ollama"


class WebSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = None
    search_type: str = "web"  # web or news


class BrowseRequest(BaseModel):
    url: str
    extract_clean_text: bool = True


class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


class PrecedentSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    top_k: int = 5
    min_similarity: float = 0.75
