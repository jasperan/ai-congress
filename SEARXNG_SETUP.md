# SearXNG Setup Guide

SearXNG is a free, open-source metasearch engine that aggregates results from multiple search engines without rate limits or API requirements.

## Why SearXNG?

- ✅ **No API required** - Self-hosted, no external dependencies
- ✅ **No rate limits** - Unlike DuckDuckGo's public API
- ✅ **Privacy-focused** - No tracking, no ads
- ✅ **Aggregates multiple engines** - Google, Bing, DuckDuckGo, and 70+ more
- ✅ **Fast and reliable** - Runs locally on your machine

## Quick Start (1 minute)

### 1. Start SearXNG with Docker

```bash
# From the ai-congress directory
docker-compose -f docker-compose.searxng.yml up -d
```

This will:
- Pull the SearXNG Docker image
- Start SearXNG on http://localhost:8888
- Configure it to work with AI Congress

### 2. Verify It's Running

Open your browser and visit: http://localhost:8888

You should see the SearXNG search interface.

### 3. Test a Search

```bash
# Test the search endpoint
curl "http://localhost:8888/search?q=test&format=json"
```

### 4. Configuration Already Done! ✅

The `config/config.yaml` has already been updated to use SearXNG:

```yaml
web_search:
  default_engine: "searxng"  # Uses SearXNG by default
  max_results: 5
  timeout: 10
  searxng_url: "http://localhost:8888"
```

## Usage in AI Congress

Now when you enable web search in your chat requests, it will use SearXNG instead of DuckDuckGo:

```json
{
  "prompt": "What is the latest news about AI?",
  "models": ["mistral:7b", "phi3:3.8b", "llama3.2:3b"],
  "mode": "multi_model",
  "search_web": true
}
```

The response will now include `web_search_results`:

```json
{
  "final_answer": "...",
  "confidence": 0.85,
  "web_search_results": [
    {
      "title": "Latest AI Developments",
      "url": "https://example.com/ai-news",
      "snippet": "Recent breakthroughs in...",
      "source": "searxng"
    }
  ]
}
```

## Advanced Configuration

### Custom SearXNG Settings

To customize SearXNG settings, create a `settings.yml` file:

```bash
mkdir -p searxng
cat > searxng/settings.yml << 'EOF'
use_default_settings: true
server:
  secret_key: "change-this-secret-key"
  limiter: false
  image_proxy: true
search:
  safe_search: 0
  autocomplete: "google"
  formats:
    - html
    - json
engines:
  - name: google
    weight: 2
  - name: bing
    weight: 1.5
  - name: duckduckgo
    weight: 1
EOF
```

Then restart SearXNG:

```bash
docker-compose -f docker-compose.searxng.yml restart
```

### Change Port

Edit `docker-compose.searxng.yml`:

```yaml
ports:
  - "9999:8080"  # Change 8888 to 9999
```

And update `config/config.yaml`:

```yaml
web_search:
  searxng_url: "http://localhost:9999"
```

## Troubleshooting

### SearXNG not responding

```bash
# Check if container is running
docker ps | grep searxng

# Check logs
docker logs searxng

# Restart if needed
docker-compose -f docker-compose.searxng.yml restart
```

### Connection refused error

Make sure SearXNG is running and accessible:

```bash
curl http://localhost:8888/
```

If this fails, check Docker status:

```bash
docker ps -a | grep searxng
```

### Slow search results

SearXNG aggregates from multiple engines, which can take time. You can:

1. Reduce the number of engines in `searxng/settings.yml`
2. Increase timeout in `config/config.yaml`:

```yaml
web_search:
  timeout: 20  # Increase from 10 to 20 seconds
```

## Switching Back to DuckDuckGo

If you prefer to use DuckDuckGo (without rate limits by using it directly), edit `config/config.yaml`:

```yaml
web_search:
  default_engine: "duckduckgo"
```

## Stop SearXNG

```bash
docker-compose -f docker-compose.searxng.yml down
```

## Production Deployment

For production, consider:

1. **Use HTTPS**: Put SearXNG behind a reverse proxy (nginx, Caddy)
2. **Rate limiting**: Enable in SearXNG settings
3. **Authentication**: Add basic auth to prevent abuse
4. **Resource limits**: Set Docker memory/CPU limits

Example with rate limiting enabled in `searxng/settings.yml`:

```yaml
server:
  limiter: true
  rate_limit:
    window_size: 300  # 5 minutes
    max_requests: 100  # Max 100 requests per window
```

## Benefits for AI Congress

With SearXNG integrated:

- ✅ No more rate limit errors
- ✅ See web search results in chat response
- ✅ Multiple search engines aggregated
- ✅ Self-hosted and private
- ✅ Works offline (with cached results)

Enjoy unlimited web search! 🚀

