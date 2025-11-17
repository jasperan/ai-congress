<script>
  import { onMount } from 'svelte'
  import ChatInterface from './components/Chat/ChatInterface.svelte'
  import PersonalityChat from './components/Personality/PersonalityChat.svelte'

  let models = []
  let selectedModels = []
  let isLoading = true
  let error = null
  let darkMode = false
  let activeTab = 'models' // 'models' or 'personalities'

  // Load models on mount
  onMount(async () => {
    // Check for dark mode preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      darkMode = true
      document.documentElement.classList.add('dark')
    }

    try {
      const response = await fetch('/api/models')
      if (!response.ok) throw new Error('Failed to load models')
      
      models = await response.json()
      
      // Auto-select preferred models if available
      const preferredModels = ['phi3:3.8b', 'mistral:7b', 'llama3.2:3b']
      selectedModels = models
        .filter(m => preferredModels.includes(m.name))
        .map(m => m.name)
      
      // If no preferred models found, select first 2
      if (selectedModels.length === 0 && models.length > 0) {
        selectedModels = models.slice(0, Math.min(2, models.length)).map(m => m.name)
      }
    } catch (e) {
      console.error('Error loading models:', e)
      error = e.message
    } finally {
      isLoading = false
    }
  })

  function toggleDarkMode() {
    darkMode = !darkMode
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
  <!-- Navigation Bar -->
  <nav class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo and Title -->
        <div class="flex items-center space-x-3">
          <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-purple-600 
                      flex items-center justify-center text-white text-xl shadow-lg">
            🏛️
          </div>
          <div>
            <h1 class="text-xl font-bold text-gray-900 dark:text-gray-100">
              AI Congress
            </h1>
            <p class="text-xs text-gray-600 dark:text-gray-400">
              LLM Swarm Decision Making
            </p>
          </div>
        </div>

        <!-- Controls -->
        <div class="flex items-center space-x-4">
          <!-- Stats Badge -->
          {#if models.length > 0}
            <div class="hidden sm:flex items-center space-x-2 text-sm">
              <span class="badge badge-info">
                {models.length} Models Available
              </span>
              <span class="badge badge-success">
                {selectedModels.length} Selected
              </span>
            </div>
          {/if}

          <!-- Dark Mode Toggle -->
          <button
            on:click={toggleDarkMode}
            class="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 
                   dark:hover:bg-gray-600 transition-colors"
            title="Toggle dark mode"
          >
            {#if darkMode}
              <svg class="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd"/>
              </svg>
            {:else}
              <svg class="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 20 20">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/>
              </svg>
            {/if}
          </button>

          <!-- GitHub Link -->
          <a
            href="https://github.com/jasperan/ai-congress"
            target="_blank"
            rel="noopener noreferrer"
            class="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 
                   dark:hover:bg-gray-600 transition-colors"
            title="View on GitHub"
          >
            <svg class="w-5 h-5 text-gray-700 dark:text-gray-300" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clip-rule="evenodd"/>
            </svg>
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="container mx-auto px-4 py-6 h-[calc(100vh-4rem)]">
    {#if isLoading}
      <!-- Loading State -->
      <div class="flex items-center justify-center h-full">
        <div class="text-center space-y-4">
          <div class="inline-flex items-center justify-center w-16 h-16 rounded-full 
                      bg-primary-100 dark:bg-primary-900/30">
            <svg class="animate-spin h-8 w-8 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Loading AI Congress...
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">
              Fetching available models from Ollama
            </p>
          </div>
        </div>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="flex items-center justify-center h-full">
        <div class="card p-8 max-w-md text-center space-y-4">
          <div class="w-16 h-16 mx-auto rounded-full bg-red-100 dark:bg-red-900/30 
                      flex items-center justify-center">
            <svg class="w-8 h-8 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
            </svg>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Failed to Connect
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {error}
            </p>
            <button 
              on:click={() => window.location.reload()}
              class="btn-primary"
            >
              Retry Connection
            </button>
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400 pt-4 border-t border-gray-200 dark:border-gray-700">
            <p class="font-medium mb-1">Troubleshooting:</p>
            <ul class="text-left space-y-1">
              <li>• Make sure Ollama is running</li>
              <li>• Check that the API server is started</li>
              <li>• Verify your network connection</li>
            </ul>
          </div>
        </div>
      </div>
    {:else if models.length === 0}
      <!-- No Models State -->
      <div class="flex items-center justify-center h-full">
        <div class="card p-8 max-w-md text-center space-y-4">
          <div class="w-16 h-16 mx-auto rounded-full bg-yellow-100 dark:bg-yellow-900/30 
                      flex items-center justify-center">
            <svg class="w-8 h-8 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
            </svg>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              No Models Available
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
              You need to pull some Ollama models before you can start chatting.
            </p>
            <div class="text-left bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-xs font-mono">
              <p class="text-gray-700 dark:text-gray-300 mb-2">Run these commands:</p>
              <code class="block text-primary-600 dark:text-primary-400">
                ollama pull phi3:3.8b<br/>
                ollama pull mistral:7b<br/>
                ollama pull llama3.2:3b
              </code>
            </div>
          </div>
        </div>
      </div>
    {:else}
      <!-- Tab Navigation -->
      <div class="mb-6">
        <div class="border-b border-gray-200 dark:border-gray-700">
          <nav class="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              on:click={() => activeTab = 'models'}
              class="border-b-2 py-2 px-1 text-sm font-medium {activeTab === 'models'
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'}"
            >
              🤖 Model Swarm
            </button>
            <button
              on:click={() => activeTab = 'personalities'}
              class="border-b-2 py-2 px-1 text-sm font-medium {activeTab === 'personalities'
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'}"
            >
              🎭 Personality Swarm
            </button>
          </nav>
        </div>
      </div>

      <!-- Tab Content -->
      <div class="h-full">
        {#if activeTab === 'models'}
          <ChatInterface {models} bind:selectedModels />
        {:else if activeTab === 'personalities'}
          <PersonalityChat {models} />
        {/if}
      </div>
    {/if}
  </main>

  <!-- Footer -->
  <footer class="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t 
                 border-gray-200 dark:border-gray-700 py-2 px-4 text-center text-xs 
                 text-gray-600 dark:text-gray-400">
    <p>
      Powered by 
      <a href="https://ollama.com" target="_blank" class="text-primary-600 dark:text-primary-400 hover:underline">
        Ollama
      </a>
      • Built with 
      <span class="text-red-500">❤️</span>
      for the open-source community
    </p>
  </footer>
</div>
