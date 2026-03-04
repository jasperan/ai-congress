<script>
  import { onMount } from 'svelte'
  import ChatInterface from './components/Chat/ChatInterface.svelte'
  import PersonalityChat from './components/Personality/PersonalityChat.svelte'

  let models = []
  let selectedModels = []
  let isLoading = true
  let error = null
  let darkMode = false
  let activeTab = 'models'
  let mounted = false

  onMount(async () => {
    mounted = true
    
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      darkMode = true
      document.documentElement.classList.add('dark')
    }

    try {
      const response = await fetch('/api/models')
      if (!response.ok) throw new Error('Failed to load models')
      models = await response.json()

      const preferredModels = ['phi3:3.8b', 'mistral:7b', 'llama3.2:3b']
      selectedModels = models
        .filter(m => preferredModels.includes(m.name))
        .map(m => m.name)

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

<!-- 
  DESIGN SYSTEM: Capitol Modern
  Aesthetic: Government architecture meets futuristic civic tech
  Inspiration: US Capitol building geometry + modern democratic transparency
  Colors: Deep navy (#0f172a), marble white (#f8fafc), gold accents (#d4af37), democratic blue (#1e40af)
  Typography: Playfair Display (authority) + Inter (clarity)
-->

<svelte:head>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</svelte:head>

<div class="min-h-screen bg-capitol-50 dark:bg-capitol-950 transition-colors duration-500 font-sans">
  <!-- Navigation Bar -->
  <nav class="sticky top-0 z-50 bg-white/90 dark:bg-capitol-900/90 backdrop-blur-xl border-b border-capitol-200 dark:border-capitol-800 shadow-sm transition-all duration-300">
    <div class="max-w-7xl mx-auto px-6 lg:px-8">
      <div class="flex items-center justify-between h-20">
        <!-- Logo and Title -->
        <div class="flex items-center gap-4">
          <div class="relative w-12 h-12 rounded-xl bg-gradient-to-br from-capitol-700 via-capitol-600 to-gold-500 flex items-center justify-center shadow-lg shadow-capitol-500/20 overflow-hidden group">
            <div class="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width=\'20\' height=\'20\' viewBox=\'0 0 20 20\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'%23ffffff\' fill-opacity=\'0.05\' fill-rule=\'evenodd\'%3E%3Ccircle cx=\'3\' cy=\'3\' r=\'3\'/%3E%3Ccircle cx=\'13\' cy=\'13\' r=\'3\'/%3E%3C/g%3E%3C/svg%3E')] opacity-50"></div>
            <svg class="w-7 h-7 text-white relative z-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div class="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
          </div>
          <div>
            <h1 class="font-display text-2xl font-bold text-capitol-900 dark:text-white tracking-tight">
              AI Congress
            </h1>
            <p class="text-xs font-medium text-capitol-500 dark:text-capitol-400 uppercase tracking-widest">
              Democratic Intelligence
            </p>
          </div>
        </div>

        <!-- Center Stats -->
        {#if models.length > 0}
          <div class="hidden md:flex items-center gap-6">
            <div class="flex items-center gap-2 px-4 py-2 rounded-full bg-capitol-100 dark:bg-capitol-800/50 border border-capitol-200 dark:border-capitol-700">
              <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              <span class="text-sm font-medium text-capitol-700 dark:text-capitol-300">
                {models.length} Models Available
              </span>
            </div>
            <div class="flex items-center gap-2 px-4 py-2 rounded-full bg-gold-100 dark:bg-gold-900/30 border border-gold-200 dark:border-gold-800">
              <svg class="w-4 h-4 text-gold-600 dark:text-gold-400" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
              </svg>
              <span class="text-sm font-medium text-gold-700 dark:text-gold-300">
                {selectedModels.length} Active
              </span>
            </div>
          </div>
        {/if}

        <!-- Controls -->
        <div class="flex items-center gap-3">
          <button
            on:click={toggleDarkMode}
            class="p-2.5 rounded-xl bg-capitol-100 dark:bg-capitol-800 text-capitol-600 dark:text-capitol-400 hover:bg-capitol-200 dark:hover:bg-capitol-700 transition-all duration-200"
            aria-label="Toggle dark mode"
          >
            {#if darkMode}
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd"/>
              </svg>
            {:else}
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/>
              </svg>
            {/if}
          </button>

          <a
            href="https://github.com/jasperan/ai-congress"
            target="_blank"
            rel="noopener noreferrer"
            class="p-2.5 rounded-xl bg-capitol-100 dark:bg-capitol-800 text-capitol-600 dark:text-capitol-400 hover:bg-capitol-200 dark:hover:bg-capitol-700 transition-all duration-200"
            aria-label="View on GitHub"
          >
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clip-rule="evenodd"/>
            </svg>
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-6 lg:px-8 py-8 min-h-[calc(100vh-5rem)]">
    {#if isLoading}
      <!-- Loading State -->
      <div class="flex items-center justify-center h-[60vh]">
        <div class="text-center space-y-6">
          <div class="relative w-24 h-24 mx-auto">
            <div class="absolute inset-0 rounded-full border-4 border-capitol-200 dark:border-capitol-800"></div>
            <div class="absolute inset-0 rounded-full border-4 border-gold-500 border-t-transparent animate-spin"></div>
            <div class="absolute inset-4 rounded-full bg-gradient-to-br from-capitol-100 to-capitol-200 dark:from-capitol-800 dark:to-capitol-900 flex items-center justify-center">
              <svg class="w-8 h-8 text-capitol-600 dark:text-capitol-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
          </div>
          <div>
            <h3 class="font-display text-2xl font-semibold text-capitol-900 dark:text-white mb-2">
              Assembling Congress...
            </h3>
            <p class="text-capitol-500 dark:text-capitol-400">
              Gathering representatives from Ollama
            </p>
          </div>
        </div>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="flex items-center justify-center h-[60vh]">
        <div class="max-w-md w-full bg-white dark:bg-capitol-900 rounded-2xl shadow-xl border border-red-200 dark:border-red-900/50 p-8 text-center">
          <div class="w-20 h-20 mx-auto rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mb-6">
            <svg class="w-10 h-10 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
            </svg>
          </div>
          <h3 class="font-display text-xl font-semibold text-capitol-900 dark:text-white mb-2">
            Session Adjourned
          </h3>
          <p class="text-capitol-500 dark:text-capitol-400 mb-6">{error}</p>
          <button
            on:click={() => window.location.reload()}
            class="px-6 py-3 bg-capitol-600 hover:bg-capitol-700 text-white font-medium rounded-xl transition-colors shadow-lg shadow-capitol-600/25"
          >
            Reconvene
          </button>
          <div class="mt-6 pt-6 border-t border-capitol-200 dark:border-capitol-800 text-left">
            <p class="text-sm font-medium text-capitol-700 dark:text-capitol-300 mb-3">Troubleshooting:</p>
            <ul class="text-sm text-capitol-500 dark:text-capitol-400 space-y-2">
              <li class="flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-capitol-400"></span>
                Ensure Ollama is running
              </li>
              <li class="flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-capitol-400"></span>
                Check API server status
              </li>
              <li class="flex items-center gap-2">
                <span class="w-1.5 h-1.5 rounded-full bg-capitol-400"></span>
                Verify network connection
              </li>
            </ul>
          </div>
        </div>
      </div>
    {:else if models.length === 0}
      <!-- No Models State -->
      <div class="flex items-center justify-center h-[60vh]">
        <div class="max-w-md w-full bg-white dark:bg-capitol-900 rounded-2xl shadow-xl border border-amber-200 dark:border-amber-900/50 p-8 text-center">
          <div class="w-20 h-20 mx-auto rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center mb-6">
            <svg class="w-10 h-10 text-amber-600 dark:text-amber-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
            </svg>
          </div>
          <h3 class="font-display text-xl font-semibold text-capitol-900 dark:text-white mb-2">
            Empty Chamber
          </h3>
          <p class="text-capitol-500 dark:text-capitol-400 mb-6">
            No representatives found. Install Ollama models to begin.
          </p>
          <div class="bg-capitol-100 dark:bg-capitol-800 rounded-xl p-4 text-left font-mono text-sm">
            <p class="text-capitol-600 dark:text-capitol-400 mb-2"># Pull representatives:</p>
            <code class="block text-capitol-800 dark:text-capitol-200 space-y-1">
              <div>ollama pull phi3:3.8b</div>
              <div>ollama pull mistral:7b</div>
              <div>ollama pull llama3.2:3b</div>
            </code>
          </div>
        </div>
      </div>
    {:else}
      <!-- Tab Navigation -->
      <div class="mb-8">
        <div class="inline-flex p-1.5 bg-capitol-100 dark:bg-capitol-800/50 rounded-2xl border border-capitol-200 dark:border-capitol-700">
          <button
            on:click={() => activeTab = 'models'}
            class="px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-300 flex items-center gap-2 {activeTab === 'models' ? 'bg-white dark:bg-capitol-700 text-capitol-900 dark:text-white shadow-md' : 'text-capitol-600 dark:text-capitol-400 hover:text-capitol-900 dark:hover:text-white'}"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/>
            </svg>
            Model Assembly
          </button>
          <button
            on:click={() => activeTab = 'personalities'}
            class="px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-300 flex items-center gap-2 {activeTab === 'personalities' ? 'bg-white dark:bg-capitol-700 text-capitol-900 dark:text-white shadow-md' : 'text-capitol-600 dark:text-capitol-400 hover:text-capitol-900 dark:hover:text-white'}"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/>
            </svg>
            Character Caucus
          </button>
        </div>
      </div>

      <!-- Tab Content -->
      <div class="animate-fade-in">
        {#if activeTab === 'models'}
          <ChatInterface {models} bind:selectedModels />
        {:else if activeTab === 'personalities'}
          <PersonalityChat {models} />
        {/if}
      </div>
    {/if}
  </main>

  <!-- Footer -->
  <footer class="border-t border-capitol-200 dark:border-capitol-800 bg-white dark:bg-capitol-900">
    <div class="max-w-7xl mx-auto px-6 lg:px-8 py-6">
      <div class="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-capitol-500 dark:text-capitol-400">
        <div class="flex items-center gap-2">
          <span class="font-display font-semibold text-capitol-700 dark:text-capitol-300">AI Congress</span>
          <span class="text-capitol-300 dark:text-capitol-600">•</span>
          <span>Open Source Democracy</span>
        </div>
        <div class="flex items-center gap-6">
          <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" class="hover:text-capitol-700 dark:hover:text-capitol-300 transition-colors">
            Powered by Ollama
          </a>
          <span class="text-gold-500">✦</span>
          <span>Built for the People</span>
        </div>
      </div>
    </div>
  </footer>
</div>

<style>
  :global(:root) {
    /* Capitol Design System */
    --color-capitol-50: #f8fafc;
    --color-capitol-100: #f1f5f9;
    --color-capitol-200: #e2e8f0;
    --color-capitol-300: #cbd5e1;
    --color-capitol-400: #94a3b8;
    --color-capitol-500: #64748b;
    --color-capitol-600: #475569;
    --color-capitol-700: #334155;
    --color-capitol-800: #1e293b;
    --color-capitol-900: #0f172a;
    --color-capitol-950: #020617;
    
    --color-gold-100: #fef3c7;
    --color-gold-200: #fde68a;
    --color-gold-300: #fcd34d;
    --color-gold-400: #fbbf24;
    --color-gold-500: #d4af37;
    --color-gold-600: #b45309;
    --color-gold-700: #92400e;
    --color-gold-800: #78350f;
    --color-gold-900: #451a03;
  }

  :global(.font-display) {
    font-family: 'Playfair Display', serif;
  }

  :global(.font-sans) {
    font-family: 'Inter', system-ui, sans-serif;
  }

  :global(.bg-capitol-50) { background-color: var(--color-capitol-50); }
  :global(.bg-capitol-100) { background-color: var(--color-capitol-100); }
  :global(.bg-capitol-200) { background-color: var(--color-capitol-200); }
  :global(.bg-capitol-600) { background-color: var(--color-capitol-600); }
  :global(.bg-capitol-700) { background-color: var(--color-capitol-700); }
  :global(.bg-capitol-800) { background-color: var(--color-capitol-800); }
  :global(.bg-capitol-900) { background-color: var(--color-capitol-900); }
  :global(.bg-capitol-950) { background-color: var(--color-capitol-950); }
  
  :global(.bg-gold-100) { background-color: var(--color-gold-100); }
  :global(.bg-gold-500) { background-color: var(--color-gold-500); }
  :global(.bg-gold-900\/30) { background-color: rgba(69, 26, 3, 0.3); }
  
  :global(.text-capitol-200) { color: var(--color-capitol-200); }
  :global(.text-capitol-300) { color: var(--color-capitol-300); }
  :global(.text-capitol-400) { color: var(--color-capitol-400); }
  :global(.text-capitol-500) { color: var(--color-capitol-500); }
  :global(.text-capitol-600) { color: var(--color-capitol-600); }
  :global(.text-capitol-700) { color: var(--color-capitol-700); }
  :global(.text-capitol-800) { color: var(--color-capitol-800); }
  :global(.text-capitol-900) { color: var(--color-capitol-900); }
  
  :global(.text-gold-300) { color: var(--color-gold-300); }
  :global(.text-gold-400) { color: var(--color-gold-400); }
  :global(.text-gold-500) { color: var(--color-gold-500); }
  :global(.text-gold-600) { color: var(--color-gold-600); }
  :global(.text-gold-700) { color: var(--color-gold-700); }
  
  :global(.border-capitol-200) { border-color: var(--color-capitol-200); }
  :global(.border-capitol-300) { border-color: var(--color-capitol-300); }
  :global(.border-capitol-700) { border-color: var(--color-capitol-700); }
  :global(.border-capitol-800) { border-color: var(--color-capitol-800); }
  :global(.border-gold-200) { border-color: var(--color-gold-200); }
  :global(.border-gold-800) { border-color: var(--color-gold-800); }
  
  :global(.shadow-capitol-500\/20) { box-shadow: 0 10px 15px -3px rgba(100, 116, 139, 0.2); }
  :global(.shadow-capitol-600\/25) { box-shadow: 0 10px 15px -3px rgba(71, 85, 105, 0.25); }

  @keyframes fade-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  :global(.animate-fade-in) {
    animation: fade-in 0.4s ease-out forwards;
  }
</style>
