#!/bin/bash

echo "🔍 Starting SearXNG Search Engine..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start SearXNG
docker-compose -f docker-compose.searxng.yml up -d

# Wait a moment for it to start
sleep 3

# Check if it's running
if docker ps | grep -q searxng; then
    echo "✅ SearXNG started successfully!"
    echo ""
    echo "🌐 Access SearXNG at: http://localhost:8888"
    echo "📡 API endpoint: http://localhost:8888/search?q=test&format=json"
    echo ""
    echo "ℹ️  SearXNG is now the default search engine for AI Congress"
    echo "   No more rate limits! 🚀"
    echo ""
    echo "To stop SearXNG: docker-compose -f docker-compose.searxng.yml down"
else
    echo "❌ Failed to start SearXNG. Check logs with:"
    echo "   docker logs searxng"
fi

