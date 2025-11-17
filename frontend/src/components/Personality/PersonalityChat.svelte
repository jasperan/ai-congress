<script>
  import ModelResponse from '../Models/ModelResponse.svelte'
  import VoteBreakdown from '../Voting/VoteBreakdown.svelte'

  export const models = []

  let personalities = []
  let personalityLists = []
  let selectedPersonalities = []
  let prompt = ''
  let messages = []
  let isLoading = false
  let showResponses = false
  let currentResult = null
  let showCreateForm = false
  let newPersonality = {
    name: '',
    description: '',
    system_prompt: ''
  }

  // Load personalities on mount
  async function loadPersonalities() {
    try {
      const response = await fetch('/api/personalities')
      if (response.ok) {
        personalities = await response.json()
      }
    } catch (e) {
      console.error('Error loading personalities:', e)
    }
  }

  // Load available personality lists
  async function loadPersonalityLists() {
    try {
      const response = await fetch('/api/personality-lists')
      if (response.ok) {
        personalityLists = await response.json()
      }
    } catch (e) {
      console.error('Error loading personality lists:', e)
    }
  }

  // Load personalities when component mounts
  import { onMount } from 'svelte'
  onMount(async () => {
    await loadPersonalities()
    await loadPersonalityLists()
  })

  async function sendMessage() {
    if (!prompt.trim() || selectedPersonalities.length === 0) {
      return
    }

    isLoading = true
    showResponses = false
    const userMessage = { role: 'user', content: prompt, timestamp: Date.now() }
    messages = [...messages, userMessage]
    const currentPrompt = prompt
    prompt = ''

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: currentPrompt,
          personalities: selectedPersonalities,
          models: [],
          mode: 'personality'
        })
      })

      const result = await response.json()
      currentResult = result

      messages = [...messages, {
        role: 'assistant',
        content: result.final_answer,
        result: result,
        timestamp: Date.now()
      }]
    } catch (e) {
      console.error('Error:', e)
      messages = [...messages, {
        role: 'assistant',
        content: 'Error: ' + e.message,
        timestamp: Date.now()
      }]
    }

    isLoading = false
  }

  function togglePersonality(personality) {
    if (selectedPersonalities.find(p => p.name === personality.name)) {
      selectedPersonalities = selectedPersonalities.filter(p => p.name !== personality.name)
    } else {
      selectedPersonalities = [...selectedPersonalities, personality]
    }
  }

  function toggleResponsesView() {
    showResponses = !showResponses
  }

  function toggleCreateForm() {
    showCreateForm = !showCreateForm
    if (!showCreateForm) {
      newPersonality = { name: '', description: '', system_prompt: '' }
    }
  }

  async function createPersonality() {
    if (!newPersonality.name.trim() || !newPersonality.system_prompt.trim()) {
      return
    }

    try {
      const response = await fetch('/api/personalities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPersonality)
      })

      if (response.ok) {
        await loadPersonalities() // Reload personalities
        toggleCreateForm()
      } else {
        alert('Failed to create personality')
      }
    } catch (e) {
      console.error('Error creating personality:', e)
      alert('Error creating personality')
    }
  }

  async function loadGroup(listName) {
    console.log(`[UI][${new Date().toISOString()}][GROUP_LOAD] Loading ${listName} personalities...`)
    try {
      const response = await fetch(`/api/personality-list/${listName}`)
      if (response.ok) {
        const groupPersonalities = await response.json()
        // Add all personalities from the list that aren't already selected
        const newPersonalities = groupPersonalities.filter(p =>
          !selectedPersonalities.find(sp => sp.name === p.name)
        )
        selectedPersonalities = [...selectedPersonalities, ...newPersonalities]
        // Also add to personalities array for individual selection buttons
        personalities = [...personalities, ...newPersonalities]
        console.log(`[UI][${new Date().toISOString()}][GROUP_LOADED] Added ${newPersonalities.length} personalities from ${listName} (total: ${selectedPersonalities.length})`)
      } else {
        console.error(`[UI][${new Date().toISOString()}][GROUP_ERROR] Failed to load ${listName} list: ${response.status} ${response.statusText}`)
      }
    } catch (e) {
      console.error(`[UI][${new Date().toISOString()}][GROUP_ERROR] Error loading ${listName}:`, e)
    }
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
    <!-- Personality Group Selector -->
    <div>
      <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
        Load Personality Groups
      </h3>
      <div class="flex flex-wrap gap-2">
        {#each personalityLists as list}
          <button
            on:click={() => loadGroup(list)}
            class="btn-secondary text-sm px-3 py-2"
          >
            Load {list.charAt(0).toUpperCase() + list.slice(1)} Group
          </button>
        {/each}
      </div>
    </div>

    <!-- Personality Selector -->
    <div>
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Select Personalities ({selectedPersonalities.length} selected)
        </h3>
        <div class="flex items-center space-x-2">
          <span class="text-xs text-gray-500 dark:text-gray-400">
            {personalities.length} available
          </span>
          <button
            on:click={toggleCreateForm}
            class="btn-secondary text-xs px-3 py-1"
          >
            + Create New
          </button>
        </div>
      </div>

      <div class="flex flex-wrap gap-2">
        {#each personalities as personality}
          <button
            on:click={() => togglePersonality(personality)}
            class="group relative px-4 py-2 rounded-lg border-2 transition-all duration-200
                   {selectedPersonalities.find(p => p.name === personality.name)
                     ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                     : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:border-primary-300'
                   }"
          >
            <div class="flex items-center space-x-2">
              {#if selectedPersonalities.find(p => p.name === personality.name)}
                <svg class="w-4 h-4 text-primary-600 dark:text-primary-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                </svg>
              {/if}
              <span class="text-sm font-medium">{personality.name}</span>
            </div>
          </button>
        {/each}
      </div>
    </div>
  </div>

  <!-- Create Personality Form -->
  {#if showCreateForm}
    <div class="card p-4 mb-4 space-y-4">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">Create New Personality</h3>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label for="personality-name" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Name *
          </label>
          <input
            id="personality-name"
            type="text"
            bind:value={newPersonality.name}
            placeholder="e.g., Albert Einstein"
            class="input-field"
          />
        </div>

        <div>
          <label for="personality-description" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Description
          </label>
          <input
            id="personality-description"
            type="text"
            bind:value={newPersonality.description}
            placeholder="Brief description of the personality"
            class="input-field"
          />
        </div>
      </div>

      <div>
        <label for="personality-prompt" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          System Prompt *
        </label>
        <textarea
          id="personality-prompt"
          bind:value={newPersonality.system_prompt}
          placeholder="Describe how this personality should respond and behave..."
          class="input-field h-32 resize-none"
        />
      </div>

      <div class="flex justify-end space-x-3">
        <button on:click={toggleCreateForm} class="btn-secondary">
          Cancel
        </button>
        <button
          on:click={createPersonality}
          disabled={!newPersonality.name.trim() || !newPersonality.system_prompt.trim()}
          class="btn-primary"
        >
          Create Personality
        </button>
      </div>
    </div>
  {/if}

  <!-- Messages Container -->
  <div class="flex-1 card mb-4 flex flex-col overflow-hidden">
    <div class="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
      {#if messages.length === 0}
        <div class="flex flex-col items-center justify-center h-full text-center py-12">
          <div class="w-16 h-16 mb-4 rounded-full bg-gradient-to-br from-purple-500 to-pink-600
                      flex items-center justify-center text-white text-2xl shadow-lg">
            🎭
          </div>
          <h3 class="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            Personality Swarm Chat
          </h3>
          <p class="text-gray-600 dark:text-gray-400 max-w-md">
            Select personalities above and start chatting. Watch as different AI personalities
            debate and collaborate using their unique perspectives and characteristics.
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
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-600
                                flex items-center justify-center text-white text-sm shadow-md">
                      🎭
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
              Personalities are debating...
            </span>
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Input Area -->
  <div class="card p-4">
    <div class="flex space-x-3">
      <textarea
        bind:value={prompt}
        on:keydown={handleKeydown}
        placeholder="Type your message... (Shift+Enter for new line)"
        class="input-field resize-none h-20"
        disabled={isLoading || selectedPersonalities.length === 0}
      />
      <button
        on:click={sendMessage}
        disabled={isLoading || !prompt.trim() || selectedPersonalities.length === 0}
        class="btn-primary px-6 flex items-center space-x-2 self-end"
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

    {#if selectedPersonalities.length === 0}
      <p class="text-xs text-red-500 dark:text-red-400 mt-2">
        Please select at least one personality to start chatting
      </p>
    {/if}
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
          Personality Response Details
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

      <!-- Individual Personality Responses -->
      {#if currentResult.responses && currentResult.responses.length > 0}
        <div class="space-y-4">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Individual Personality Responses
          </h3>

          {#each currentResult.responses as response, i}
            <ModelResponse {response} index={i} />
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}
