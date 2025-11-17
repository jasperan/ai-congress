# Open WebUI Features Integration - Implementation Summary

## Overview
Successfully integrated Open WebUI-inspired features into the AI Congress system, including voice input, document RAG with Oracle 26ai vector store, web search, web browsing, and image generation.

## ✅ Completed Features

### 1. Voice Input (Faster-Whisper)
- **Backend**: `src/ai_congress/integrations/voice.py`
  - Local Whisper model via Faster-Whisper
  - Support for multiple audio formats (WAV, MP3, etc.)
  - Language detection and transcription
- **API Endpoint**: `POST /api/audio/transcribe`
- **Frontend**: `frontend/src/components/Voice/VoiceInput.svelte`
  - Browser audio recording with MediaRecorder API
  - Real-time recording indicator
  - Automatic transcription and text insertion
- **Setup**: Whisper model auto-downloads on first use (configured in `startup.sh`)

### 2. Document Upload & Processing
- **Backend**: `src/ai_congress/integrations/documents.py`
  - **Supported formats**: PDF, DOCX, TXT, PPTX, CSV, JSON, Markdown, XLSX
  - Text chunking with configurable size and overlap
  - Metadata extraction (filename, page numbers, timestamps)
- **API Endpoints**:
  - `POST /api/documents/upload` - Upload and process documents
  - `GET /api/documents/list` - List all uploaded documents
  - `DELETE /api/documents/{document_id}` - Delete documents
- **Frontend**: 
  - `frontend/src/components/Documents/DocumentUpload.svelte` - Drag-drop upload
  - `frontend/src/components/Documents/DocumentList.svelte` - Document management

### 3. Oracle Database 26ai Vector Store
- **Backend**: `src/ai_congress/integrations/oracle_vector_store.py`
  - Based on agentic_rag OraDBVectorStore.py
  - TLS connection support
  - Vector similarity search with cosine distance
  - CRUD operations for document vectors
- **Features**:
  - Automatic table and index creation
  - Batch vector insertion
  - Top-K similarity search
  - Document deletion
- **Configuration**: `config/config.yaml` → `oracle_db` section

### 4. Embeddings Generation
- **Backend**: `src/ai_congress/integrations/embeddings.py`
  - sentence-transformers integration
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing support
  - L2 normalization for cosine similarity
- **Features**:
  - Singleton pattern for model reuse
  - GPU/CPU auto-detection
  - Similarity calculations

### 5. RAG (Retrieval-Augmented Generation) Engine
- **Backend**: `src/ai_congress/core/rag_engine.py`
  - Orchestrates document processing pipeline
  - Query augmentation with retrieved context
  - Configurable top-K retrieval (default: 10)
- **Integration**:
  - Automatic RAG when documents are uploaded
  - Context injection into swarm prompts
  - Source tracking with similarity scores
- **Configuration**: `config/config.yaml` → `rag` section

### 6. Web Search (DuckDuckGo)
- **Backend**: `src/ai_congress/integrations/web_search.py`
  - Open-source, no API key required
  - Web search and news search
  - Configurable max results
- **API Endpoint**: `POST /api/search/web`
- **Integration**: Toggle in chat interface for automatic web search

### 7. Web Browsing
- **Backend**: `src/ai_congress/integrations/web_browser.py`
  - URL content fetching and parsing
  - Clean text extraction via Trafilatura
  - BeautifulSoup fallback
- **API Endpoint**: `POST /api/browse`
- **Features**:
  - Asynchronous HTTP requests
  - Metadata extraction
  - Multi-URL concurrent fetching

### 8. Image Generation (Stable Diffusion via Ollama)
- **Backend**: `src/ai_congress/integrations/image_gen.py`
  - Stable Diffusion via local Ollama
  - Configurable parameters (steps, size, seed)
  - Negative prompts support
- **API Endpoint**: `POST /api/images/generate`
- **Frontend**: `frontend/src/components/Images/ImageDisplay.svelte`
  - Image preview with metadata
  - Download functionality
- **Setup**: Stable Diffusion model auto-downloads (configured in `startup.sh`)

### 9. Enhanced Chat Interface
- **Frontend**: `frontend/src/components/Chat/ChatInterface.svelte`
- **New Features**:
  - Voice input button next to send
  - RAG toggle (auto-enables on document upload)
  - Web search toggle
  - Documents panel (upload & manage)
  - Image generation panel
  - Context source indicators
- **Smart Context Injection**:
  - Combines RAG and web search results
  - Automatic prompt augmentation
  - Source tracking in responses

## 📁 File Structure

```
src/ai_congress/
├── integrations/
│   ├── __init__.py
│   ├── voice.py                    # Voice transcription
│   ├── documents.py                # Document parsing
│   ├── embeddings.py               # Text embeddings
│   ├── oracle_vector_store.py      # Oracle 26ai integration
│   ├── web_search.py               # DuckDuckGo search
│   ├── web_browser.py              # Web content fetching
│   └── image_gen.py                # Image generation
├── core/
│   └── rag_engine.py               # RAG orchestration
└── api/
    └── main.py                     # Enhanced with new endpoints

frontend/src/components/
├── Voice/
│   └── VoiceInput.svelte           # Audio recording UI
├── Documents/
│   ├── DocumentUpload.svelte       # Drag-drop upload
│   └── DocumentList.svelte         # Document management
├── Images/
│   └── ImageDisplay.svelte         # Generated image display
└── Chat/
    └── ChatInterface.svelte        # Enhanced chat with all features
```

## 🔧 Configuration

### config.yaml
```yaml
# Oracle Database for Vector Store
oracle_db:
  user: "admin"
  password: "your_password_here"
  dsn: "localhost:1521/FREEPDB1"
  use_tls: true
  vector_table: "document_vectors"
  embedding_dimension: 384

# RAG Configuration
rag:
  enabled: true
  auto_on_upload: true
  top_k: 10
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 50

# Voice Input
voice:
  model: "base"  # whisper model size
  language: "en"
  device: "cpu"
  compute_type: "int8"

# Web Search
web_search:
  provider: "duckduckgo"
  max_results: 5
  timeout: 10

# Image Generation
image_gen:
  model: "stable-diffusion"
  default_steps: 30
  width: 512
  height: 512
  output_dir: "static/generated_images"
```

## 📦 Dependencies Added

### requirements.txt
- `faster-whisper==1.0.3` - Voice transcription
- `oracledb==2.4.1` - Oracle Database connector
- `pypdf2==3.0.1`, `python-docx==1.1.2`, etc. - Document parsing
- `sentence-transformers==3.1.1` - Embeddings
- `transformers==4.45.2`, `torch==2.5.1` - ML models
- `duckduckgo-search==6.3.0` - Web search
- `beautifulsoup4==4.12.3`, `trafilatura==1.12.2` - Web parsing
- `lxml==5.3.0`, `requests==2.32.3` - HTTP utilities

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Oracle Database
Update `config/config.yaml` with your Oracle 26ai credentials:
```yaml
oracle_db:
  user: "your_username"
  password: "your_password"
  dsn: "your_host:1521/your_service"
```

### 3. Run Setup Script
```bash
bash startup.sh
```
This will:
- Install dependencies
- Pull Ollama models (including stable-diffusion)
- Download Whisper model
- Setup frontend

### 4. Start the System
```bash
# Backend
uvicorn src.ai_congress.api.main:app --reload

# Frontend (in another terminal)
cd frontend && npm run dev
```

### 5. Use Features in Chat

#### Voice Input:
1. Click the microphone button
2. Speak your query
3. Click again to stop and transcribe

#### Document RAG:
1. Click "📄 Documents" button
2. Drag-drop or click to upload documents
3. RAG toggle auto-enables
4. Ask questions about your documents

#### Web Search:
1. Enable "🔍 Web Search" toggle
2. Your query will automatically search the web
3. Results are added as context to the conversation

#### Image Generation:
1. Click "🎨 Image Gen" button
2. Enter your image prompt
3. Click "Generate Image"
4. View and download the result

## 🔄 Data Flow

### RAG Pipeline:
1. Document Upload → Parse → Chunk → Generate Embeddings
2. Store in Oracle 26ai Vector Store
3. Query → Generate Query Embedding → Similarity Search
4. Top-K Results → Format Context → Inject into Prompt
5. Congress Deliberates with Augmented Context

### Web Search Pipeline:
1. Query → DuckDuckGo Search
2. Format Results → Inject into Prompt
3. Congress Deliberates with Web Context

## 🎯 Key Features

✅ **Hands-free voice input** - Local Whisper transcription
✅ **Document upload & RAG** - All common formats supported  
✅ **Oracle 26ai vector store** - TLS connection with vector similarity
✅ **10 relevant chunks retrieval** - Configurable top-K
✅ **DuckDuckGo web search** - Open-source, no API key needed
✅ **Web browsing capability** - Clean text extraction
✅ **Stable Diffusion image generation** - Via local Ollama
✅ **Auto-enable RAG on upload** - Smart defaults
✅ **Context source tracking** - Know where answers come from

## 📝 API Documentation

All endpoints are documented in the FastAPI Swagger UI:
- Navigate to `http://localhost:8000/docs` after starting the backend
- Interactive API testing available

## 🎨 Frontend Features

- Modern, responsive UI
- Dark mode support
- Real-time status indicators
- Drag-drop document upload
- Voice recording visualization
- Image preview and download
- Context source badges
- Document management panel

## 🔐 Security Notes

- Oracle DB connections use TLS
- File uploads are validated by extension
- Temporary files are cleaned up
- WebSocket connections are secured
- No sensitive data in client-side code

## 🚧 Future Enhancements

Consider adding:
- Document preview before upload
- Multiple document selection for targeted RAG
- Image-to-image generation
- Voice command keywords
- Conversation export with sources
- Advanced RAG filtering options

## ✨ Credits

Based on features from:
- [Open WebUI](https://github.com/open-webui/open-webui)
- [Oracle DevRel Labs - Agentic RAG](https://github.com/oracle-devrel/devrel-labs/tree/main/agentic_rag)

---

**Implementation Complete** ✅ All planned features have been successfully integrated!

