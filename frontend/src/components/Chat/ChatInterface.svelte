<script>
  import ModelResponse from '../Models/ModelResponse.svelte'
  import VoteBreakdown from '../Voting/VoteBreakdown.svelte'
  import VoiceInput from '../Voice/VoiceInput.svelte'
  import DocumentUpload from '../Documents/DocumentUpload.svelte'
  import DocumentList from '../Documents/DocumentList.svelte'
  import ImageDisplay from '../Images/ImageDisplay.svelte'
  
  export let models = []
  export let selectedModels = []

  let prompt = ''
  let mode = 'multi_model'
  let streamResponses = false
  let messages = []
  let isLoading = false
  let showResponses = false
  let currentResult = null
  let websocket = null
  let streamingMessage = null
  
  // New feature toggles
  let useRAG = false
  let searchWeb = false
  let showDocuments = false
  let showImageGen = false
  let documentsRefresh = 0
  
  // Image generation
  let imageGenPrompt = ''
  let generatedImage = null
  let isGeneratingImage = false

  function handleVoiceTranscription(text) {
    prompt = text
  }
  
  function handleDocumentUpload(result) {
    if (result.success) {
      documentsRefresh++
      useRAG = true  // Auto-enable RAG when document is uploaded
    }
  }
  
  function handleDocumentDeleted() {
    documentsRefresh++
  }
  
  async function generateImage() {
    if (!imageGenPrompt.trim()) return
    
    try {
      isGeneratingImage = true
      
      const response = await fetch('/api/images/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: imageGenPrompt
        })
      })
      
      const result = await response.json()
      
      if (result.success) {
        generatedImage = result
      } else {
        alert('Failed to generate image: ' + (result.error || 'Unknown error'))
      }
      
      isGeneratingImage = false
    } catch (error) {
      console.error('Image generation error:', error)
      alert('Error generating image')
      isGeneratingImage = false
    }
  }
  
  async function sendMessage() {
    if (!prompt.trim() || selectedModels.length === 0) {
      return
    }

    isLoading = true
    showResponses = false
    const userMessage = { role: 'user', content: prompt, timestamp: Date.now() }
    messages = [...messages, userMessage]
    const currentPrompt = prompt
    prompt = ''

    // Initialize streaming message
    streamingMessage = {
      role: 'assistant',
      content: '',
      result: null,
      timestamp: Date.now(),
      isStreaming: true
    }
    messages = [...messages, streamingMessage]

    try {
      if (streamResponses) {
        // Use WebSocket for streaming
        websocket = new WebSocket(`ws://localhost:8000/ws/chat`)

        websocket.onopen = () => {
          websocket.send(JSON.stringify({
            prompt: currentPrompt,
            models: selectedModels,
            mode: mode,
            stream: true
          }))
        }

        websocket.onmessage = (event) => {
          const data = JSON.parse(event.data)

          if (data.type === 'start') {
            // Update loading message
            streamingMessage.content = data.message
            messages = [...messages]
          } else if (data.type === 'model_response') {
            // Add individual model response as separate message
            messages = [...messages, {
              role: 'assistant',
              content: data.content,
              entity_name: data.model,
              timestamp: Date.now(),
              isIndividual: true
            }]
          } else if (data.type === 'final_answer') {
            // Final result
            streamingMessage.content = data.content
            streamingMessage.result = {
              confidence: data.confidence,
              semantic_confidence: data.semantic_confidence,
              vote_breakdown: data.vote_breakdown,
              responses: [] // Will be populated if available
            }
            currentResult = streamingMessage.result
            streamingMessage.isStreaming = false
            messages = [...messages]
            websocket.close()
            websocket = null
          } else if (data.type === 'error') {
            streamingMessage.content = `Error: ${data.message}`
            streamingMessage.isStreaming = false
            messages = [...messages]
            websocket.close()
            websocket = null
          }
        }

        websocket.onerror = async (error) => {
          console.error('WebSocket error:', error)
          // Fallback to non-streaming
          try {
            const response = await fetch('/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prompt: currentPrompt,
                models: selectedModels,
                mode
              })
            })
            const result = await response.json()
            currentResult = result
            streamingMessage.content = result.final_answer
            streamingMessage.result = result
            streamingMessage.isStreaming = false
            messages = [...messages]
            isLoading = false
          } catch (fetchError) {
            console.error('Fallback fetch error:', fetchError)
            streamingMessage.content = 'Connection error occurred'
            streamingMessage.isStreaming = false
            messages = [...messages]
            isLoading = false
          }
          if (websocket) {
            websocket.close()
            websocket = null
          }
        }

        websocket.onclose = () => {
          isLoading = false
          websocket = null
        }
      } else {
        // Use HTTP for non-streaming
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: currentPrompt,
            models: selectedModels,
            mode,
            use_rag: useRAG,
            search_web: searchWeb
          })
        })

        const result = await response.json()
        currentResult = result

        // Update the streaming message with final content
        streamingMessage.content = result.final_answer
        streamingMessage.result = result
        streamingMessage.isStreaming = false
        messages = [...messages]
        isLoading = false
      }
    } catch (e) {
      console.error('Error:', e)
      streamingMessage.content = 'Error: ' + e.message
      streamingMessage.isStreaming = false
      messages = [...messages]
      isLoading = false

      if (websocket) {
        websocket.close()
        websocket = null
      }
    }
  }

  function toggleModel(modelName) {
    if (selectedModels.includes(modelName)) {
      selectedModels = selectedModels.filter(m => m !== modelName)
    } else {
      selectedModels = [...selectedModels, modelName]
    }
  }

  function toggleResponsesView() {
    showResponses = !showResponses
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }
</script>

<div class="flex flex-col h-full max-w-7xl mx-auto">
  <!-- Header Controls -->
  <div class="card p-4 mb-4 space-y-4">
    <!-- Model Selector -->
    <div>
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Select Models ({selectedModels.length} selected)
        </h3>
        <span class="text-xs text-gray-500 dark:text-gray-400">
          {models.length} available
        </span>
      </div>
      
      <div class="flex flex-wrap gap-2">
        {#each models as model}
          <button
            on:click={() => toggleModel(model.name)}
            class="group relative px-4 py-2 rounded-lg border-2 transition-all duration-200
                   {selectedModels.includes(model.name)
                     ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                     : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:border-primary-300'
                   }"
          >
            <div class="flex items-center space-x-2">
              {#if selectedModels.includes(model.name)}
                <svg class="w-4 h-4 text-primary-600 dark:text-primary-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                </svg>
              {/if}
              <span class="text-sm font-medium">{model.name}</span>
              <span class="text-xs opacity-75">({(model.weight * 100).toFixed(0)}%)</span>
            </div>
          </button>
        {/each}
      </div>
    </div>

    <!-- Mode Selector & Feature Toggles -->
    <div class="flex flex-wrap items-center gap-4">
      <div class="flex items-center space-x-4">
        <label for="swarm-mode" class="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Swarm Mode:
        </label>
        <select
          id="swarm-mode"
          bind:value={mode}
          class="input-field text-sm py-2 w-auto"
        >
          <option value="multi_model">🔄 Multi-Model (Different Models)</option>
          <option value="multi_request">🌡️ Multi-Request (Temperature Variation)</option>
          <option value="hybrid">⚡ Hybrid (Both)</option>
        </select>
      </div>

      <!-- Feature Toggles -->
      <div class="flex items-center space-x-2">
        <input
          id="stream-toggle"
          type="checkbox"
          bind:checked={streamResponses}
          class="w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500
                 dark:focus:ring-primary-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
        />
        <label for="stream-toggle" class="text-sm font-medium text-gray-700 dark:text-gray-300">
          Stream
        </label>
      </div>
      
      <div class="flex items-center space-x-2">
        <input
          id="rag-toggle"
          type="checkbox"
          bind:checked={useRAG}
          class="w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500
                 dark:focus:ring-primary-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
        />
        <label for="rag-toggle" class="text-sm font-medium text-gray-700 dark:text-gray-300">
          📚 RAG
        </label>
      </div>
      
      <div class="flex items-center space-x-2">
        <input
          id="websearch-toggle"
          type="checkbox"
          bind:checked={searchWeb}
          class="w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500
                 dark:focus:ring-primary-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
        />
        <label for="websearch-toggle" class="text-sm font-medium text-gray-700 dark:text-gray-300">
          🔍 Web Search
        </label>
      </div>
      
      <button
        on:click={() => showDocuments = !showDocuments}
        class="text-sm px-3 py-1 rounded-lg border border-gray-300 dark:border-gray-600 hover:border-primary-500 transition-colors"
      >
        📄 Documents
      </button>
      
      <button
        on:click={() => showImageGen = !showImageGen}
        class="text-sm px-3 py-1 rounded-lg border border-gray-300 dark:border-gray-600 hover:border-primary-500 transition-colors"
      >
        🎨 Image Gen
      </button>
    </div>
  </div>

  <!-- Messages Container -->
  <div class="flex-1 card mb-4 flex flex-col overflow-hidden">
    <div class="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
      {#if messages.length === 0}
        <div class="flex flex-col items-center justify-center h-full text-center py-12">
          <div class="w-16 h-16 mb-4 rounded-full bg-gradient-to-br from-primary-500 to-purple-600 
                      flex items-center justify-center text-white text-2xl shadow-lg">
            🏛️
          </div>
          <h3 class="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            Welcome to AI Congress
          </h3>
          <p class="text-gray-600 dark:text-gray-400 max-w-md">
            Select your models above and start chatting. Watch as multiple LLMs collaborate 
            and vote on the best response using ensemble decision-making.
          </p>
        </div>
      {:else}
        {#each messages as message, i}
          <div class="message-enter flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            {#if message.role === 'user'}
              <!-- User Message -->
              <div class="max-w-2xl">
                <div class="px-4 py-3 rounded-2xl bg-primary-600 text-white shadow-md">
                  <p class="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-1 px-2">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            {:else}
              <!-- Assistant Message -->
              <div class="max-w-4xl w-full space-y-3">
                <div class="flex items-start space-x-3">
                  <div class="flex-shrink-0">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 
                                flex items-center justify-center text-white text-sm shadow-md">
                      🏛️
                    </div>
                  </div>
                  
                  <div class="flex-1">
                    <div class="px-4 py-3 rounded-2xl bg-gray-100 dark:bg-gray-800 border 
                                border-gray-200 dark:border-gray-700 shadow-sm">
                      <p class="text-sm leading-relaxed whitespace-pre-wrap text-gray-900 dark:text-gray-100">
                        {message.content}
                      </p>
                    </div>
                    
                    <div class="flex items-center space-x-4 mt-2 px-2">
                      <span class="text-xs text-gray-500 dark:text-gray-400">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                      
                      {#if message.result}
                        <button
                          on:click={() => {
                            currentResult = message.result
                            showResponses = !showResponses
                          }}
                          class="text-xs text-primary-600 dark:text-primary-400 hover:underline 
                                 flex items-center space-x-1"
                        >
                          <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                            <path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd"/>
                          </svg>
                          <span>View Details</span>
                        </button>
                        
                        {#if message.result.confidence}
                          <span class="text-xs px-2 py-0.5 rounded-full bg-congress-confidence/20 
                                       text-congress-confidence font-medium">
                            {(message.result.confidence * 100).toFixed(1)}% confidence
                          </span>
                        {/if}
                      {/if}
                    </div>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      {/if}

      {#if isLoading}
        <div class="flex justify-start message-enter">
          <div class="flex items-center space-x-3 px-4 py-3 rounded-2xl bg-gray-100 dark:bg-gray-800 
                      border border-gray-200 dark:border-gray-700">
            <div class="flex space-x-1">
              <div class="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
              <div class="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
              <div class="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
            </div>
            <span class="text-sm text-gray-600 dark:text-gray-400">
              Congress is deliberating...
            </span>
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Input Area -->
  <div class="card p-4">
    <div class="flex space-x-3">
      <div class="flex-1 space-y-2">
        <textarea
          bind:value={prompt}
          on:keydown={handleKeydown}
          placeholder="Type your message... (Shift+Enter for new line)"
          class="input-field resize-none h-20"
          disabled={isLoading || selectedModels.length === 0}
        />
        
        {#if selectedModels.length === 0}
          <p class="text-xs text-red-500 dark:text-red-400">
            Please select at least one model to start chatting
          </p>
        {/if}
      </div>
      
      <div class="flex flex-col space-y-2">
        <VoiceInput 
          onTranscription={handleVoiceTranscription}
          disabled={isLoading || selectedModels.length === 0}
        />
        
        <button
          on:click={sendMessage}
          disabled={isLoading || !prompt.trim() || selectedModels.length === 0}
          class="btn-primary px-6 flex items-center justify-center space-x-2 flex-1"
        >
          {#if isLoading}
            <svg class="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            <span>Sending</span>
          {:else}
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/>
            </svg>
            <span>Send</span>
          {/if}
        </button>
      </div>
    </div>
  </div>
</div>

<!-- Response Details Modal/Sidebar -->
{#if showResponses && currentResult}
  <div 
    class="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-fade-in" 
    on:click={toggleResponsesView}
    on:keydown={(e) => e.key === 'Escape' && toggleResponsesView()}
    role="button"
    tabindex="0"
    aria-label="Close details panel"
  ></div>
  <div class="fixed right-0 top-0 bottom-0 w-full md:w-2/3 lg:w-1/2 bg-white dark:bg-gray-900 
              shadow-2xl z-50 overflow-y-auto custom-scrollbar animate-slide-up">
    <div class="p-6 space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Response Details
        </h2>
        <button 
          on:click={toggleResponsesView}
          class="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        </button>
      </div>

      <!-- Vote Breakdown -->
      {#if currentResult.vote_breakdown || currentResult.confidence}
        <VoteBreakdown
          voteBreakdown={currentResult.vote_breakdown || {}}
          confidence={currentResult.confidence || 0}
          semanticConfidence={currentResult.semantic_confidence}
        />
      {/if}

      <!-- Individual Model Responses -->
      {#if currentResult.responses && currentResult.responses.length > 0}
        <div class="space-y-4">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Individual Model Responses
          </h3>
          
          {#each currentResult.responses as response, i}
            <ModelResponse {response} index={i} />
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<!-- Documents Panel -->
{#if showDocuments}
  <div 
    class="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-fade-in" 
    on:click={() => showDocuments = false}
    on:keydown={(e) => e.key === 'Escape' && (showDocuments = false)}
    role="button"
    tabindex="0"
    aria-label="Close documents panel"
  ></div>
  <div class="fixed right-0 top-0 bottom-0 w-full md:w-2/3 lg:w-1/2 bg-white dark:bg-gray-900 
              shadow-2xl z-50 overflow-y-auto custom-scrollbar animate-slide-up">
    <div class="p-6 space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
          📄 Document Management
        </h2>
        <button 
          on:click={() => showDocuments = false}
          class="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        </button>
      </div>

      <!-- Upload Section -->
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
          Upload New Document
        </h3>
        <DocumentUpload onUploadComplete={handleDocumentUpload} />
      </div>

      <!-- Documents List -->
      <div>
        <DocumentList 
          onDocumentDeleted={handleDocumentDeleted}
          refreshTrigger={documentsRefresh}
        />
      </div>
    </div>
  </div>
{/if}

<!-- Image Generation Panel -->
{#if showImageGen}
  <div 
    class="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-fade-in" 
    on:click={() => showImageGen = false}
    on:keydown={(e) => e.key === 'Escape' && (showImageGen = false)}
    role="button"
    tabindex="0"
    aria-label="Close image generation panel"
  ></div>
  <div class="fixed right-0 top-0 bottom-0 w-full md:w-2/3 lg:w-1/2 bg-white dark:bg-gray-900 
              shadow-2xl z-50 overflow-y-auto custom-scrollbar animate-slide-up">
    <div class="p-6 space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
          🎨 Image Generation
        </h2>
        <button 
          on:click={() => showImageGen = false}
          class="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        </button>
      </div>

      <!-- Generation Form -->
      <div class="space-y-4">
        <div>
          <label for="image-prompt" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Image Prompt
          </label>
          <textarea
            id="image-prompt"
            bind:value={imageGenPrompt}
            placeholder="Describe the image you want to generate..."
            class="input-field resize-none h-32"
            disabled={isGeneratingImage}
          />
        </div>

        <button
          on:click={generateImage}
          disabled={isGeneratingImage || !imageGenPrompt.trim()}
          class="btn-primary w-full"
        >
          {#if isGeneratingImage}
            <svg class="animate-spin h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            Generating...
          {:else}
            Generate Image
          {/if}
        </button>
      </div>

      <!-- Generated Image Display -->
      {#if generatedImage}
        <div class="mt-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Generated Image
          </h3>
          <ImageDisplay
            imageUrl={generatedImage.url}
            prompt={generatedImage.prompt}
            metadata={generatedImage.metadata}
          />
        </div>
      {/if}
    </div>
  </div>
{/if}
