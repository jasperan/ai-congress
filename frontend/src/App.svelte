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
  DESIGN SYSTEM: Capitol Modern (v2, taste-skill audited)
  Aesthetic: Editorial civic meets premium dashboard.
  Colors: Deep navy (#0f172a), marble (#f8fafc), gold (#d4af37). One accent, saturation <80%.
  Typography: Cinzel (display, editorial) + Geist (UI body) + Geist Mono (data). No Inter, no Playfair.
  Fonts load in index.html only (single source of truth).
-->

<div class="min-h-[100dvh] bg-capitol-50 dark:bg-capitol-950 transition-colors duration-500 font-sans">
  <!-- Skip to content -->
  <a href="#main-content" class="skip-link">Skip to content</a>

  <!-- Navigation Bar: liquid-glass refraction (1px inner border + inset hairline) -->
  <nav
    class="sticky top-0 z-50 bg-white/75 dark:bg-capitol-900/75 backdrop-blur-xl border-b border-capitol-200/80 dark:border-capitol-800/80 transition-all duration-300"
    style="box-shadow: inset 0 1px 0 rgba(255,255,255,0.35), 0 1px 0 rgba(15,23,42,0.04);"
  >
    <div class="max-w-7xl mx-auto px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo and Title -->
        <div class="flex items-center gap-3.5">
          <div class="relative w-10 h-10 rounded-lg bg-gradient-to-br from-capitol-700 via-capitol-600 to-gold-500 flex items-center justify-center shadow-md overflow-hidden">
            <svg class="w-5.5 h-5.5 text-white relative z-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div>
            <h1 class="font-display text-xl font-bold text-capitol-900 dark:text-white tracking-tight leading-none">
              AI Congress
            </h1>
            <p class="text-[0.65rem] font-medium text-capitol-500 dark:text-capitol-400 uppercase tracking-[0.15em] mt-0.5">
              Democratic Intelligence
            </p>
          </div>
        </div>

        <!-- Center Stats -->
        {#if models.length > 0}
          <div class="hidden md:flex items-center gap-4">
            <div class="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-capitol-100 dark:bg-capitol-800/50 border border-capitol-200 dark:border-capitol-700">
              <div class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
              <span class="text-xs font-medium text-capitol-700 dark:text-capitol-300 tabular-nums">
                {models.length} models
              </span>
            </div>
            <div class="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-gold-100 dark:bg-gold-900/30 border border-gold-200 dark:border-gold-800">
              <span class="text-xs font-medium text-gold-700 dark:text-gold-300 tabular-nums">
                {selectedModels.length} active
              </span>
            </div>
          </div>
        {/if}

        <!-- Controls -->
        <div class="flex items-center gap-2">
          <button
            on:click={toggleDarkMode}
            class="p-2 rounded-lg bg-capitol-100 dark:bg-capitol-800 text-capitol-600 dark:text-capitol-400 hover:bg-capitol-200 dark:hover:bg-capitol-700 transition-all duration-200 active:scale-95"
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
            class="p-2 rounded-lg bg-capitol-100 dark:bg-capitol-800 text-capitol-600 dark:text-capitol-400 hover:bg-capitol-200 dark:hover:bg-capitol-700 transition-all duration-200 active:scale-95"
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
  <main id="main-content" class="max-w-7xl mx-auto px-6 lg:px-8 py-8 min-h-[calc(100dvh-4rem)]">
    {#if isLoading}
      <!-- Loading State: asymmetric split, skeletal shimmer instead of centered spinner -->
      <section class="grid md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-10 md:gap-16 items-center min-h-[calc(100dvh-10rem)] py-10">
        <div class="space-y-4">
          <p class="font-mono text-[0.7rem] uppercase tracking-[0.22em] text-capitol-500 dark:text-capitol-400">
            Session opening
          </p>
          <h2 class="font-display text-4xl md:text-5xl font-semibold text-capitol-900 dark:text-white leading-[1.05] tracking-tight">
            Assembling the<br/>chamber.
          </h2>
          <p class="text-sm text-capitol-500 dark:text-capitol-400 max-w-[40ch] leading-relaxed">
            Gathering representatives from Ollama. First response lands in a moment.
          </p>
          <div class="flex items-center gap-2 pt-2">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span class="font-mono text-xs text-capitol-600 dark:text-capitol-400 tabular-nums">
              ollama · localhost:11434
            </span>
          </div>
        </div>
        <!-- Skeletal cards matching the real layout so the loading state doesn't feel like a spinner -->
        <div class="space-y-3" aria-hidden="true">
          {#each Array(4) as _, i}
            <div
              class="h-16 rounded-2xl bg-gradient-to-r from-capitol-100 via-capitol-200/70 to-capitol-100 dark:from-capitol-800 dark:via-capitol-700/70 dark:to-capitol-800 bg-[length:200%_100%] animate-skeleton-shimmer"
              style="animation-delay: {i * 120}ms"
            ></div>
          {/each}
        </div>
      </section>
    {:else if error}
      <!-- Error State: asymmetric, no centered card. Two columns: status on left, remedy on right. -->
      <section class="grid md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-10 md:gap-16 items-start min-h-[calc(100dvh-10rem)] py-10">
        <div class="space-y-5">
          <div class="flex items-center gap-2.5">
            <span class="relative flex h-2 w-2">
              <span class="absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-60 animate-ping"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            <p class="font-mono text-[0.7rem] uppercase tracking-[0.22em] text-red-600 dark:text-red-400">
              Session adjourned
            </p>
          </div>
          <h2 class="font-display text-4xl md:text-5xl font-semibold text-capitol-900 dark:text-white leading-[1.05] tracking-tight">
            The chamber<br/>isn't responding.
          </h2>
          <p class="font-mono text-sm text-capitol-600 dark:text-capitol-400 bg-capitol-100/60 dark:bg-capitol-800/60 border-l-2 border-red-400 pl-3 py-2 max-w-[50ch]">
            {error}
          </p>
          <button
            on:click={() => window.location.reload()}
            class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-capitol-900 dark:bg-white text-white dark:text-capitol-900 font-medium text-sm transition-all duration-200 hover:-translate-y-[1px] active:scale-[0.98]"
            style="box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 8px 24px -10px rgba(15,23,42,0.35);"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.75">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v6h6M20 20v-6h-6M4 10a8 8 0 0114-5l2 3M20 14a8 8 0 01-14 5l-2-3"/>
            </svg>
            Reconvene
          </button>
        </div>
        <!-- Remedy column: stacked rows, no card, divide-y logic grouping -->
        <div class="pt-2">
          <p class="font-mono text-[0.7rem] uppercase tracking-[0.22em] text-capitol-500 dark:text-capitol-400 mb-4">
            Remedy
          </p>
          <ol class="divide-y divide-capitol-200 dark:divide-capitol-800 border-y border-capitol-200 dark:border-capitol-800">
            <li class="flex items-baseline gap-4 py-4">
              <span class="font-mono text-xs text-capitol-400 tabular-nums w-8 shrink-0">01</span>
              <div class="min-w-0">
                <p class="text-sm text-capitol-900 dark:text-white">Ensure Ollama is running</p>
                <code class="font-mono text-xs text-capitol-500 dark:text-capitol-400">ollama serve</code>
              </div>
            </li>
            <li class="flex items-baseline gap-4 py-4">
              <span class="font-mono text-xs text-capitol-400 tabular-nums w-8 shrink-0">02</span>
              <div class="min-w-0">
                <p class="text-sm text-capitol-900 dark:text-white">Check the API server</p>
                <code class="font-mono text-xs text-capitol-500 dark:text-capitol-400">python run_server.py</code>
              </div>
            </li>
            <li class="flex items-baseline gap-4 py-4">
              <span class="font-mono text-xs text-capitol-400 tabular-nums w-8 shrink-0">03</span>
              <div class="min-w-0">
                <p class="text-sm text-capitol-900 dark:text-white">Verify the network</p>
                <code class="font-mono text-xs text-capitol-500 dark:text-capitol-400">curl :8100/api/models</code>
              </div>
            </li>
          </ol>
        </div>
      </section>
    {:else if models.length === 0}
      <!-- Empty State: asymmetric with terminal-style install guide -->
      <section class="grid md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-10 md:gap-16 items-start min-h-[calc(100dvh-10rem)] py-10">
        <div class="space-y-5">
          <p class="font-mono text-[0.7rem] uppercase tracking-[0.22em] text-amber-600 dark:text-amber-400">
            Empty chamber
          </p>
          <h2 class="font-display text-4xl md:text-5xl font-semibold text-capitol-900 dark:text-white leading-[1.05] tracking-tight">
            No representatives<br/>have been seated.
          </h2>
          <p class="text-sm text-capitol-500 dark:text-capitol-400 max-w-[40ch] leading-relaxed">
            Pull at least two models and AI Congress will call them to the floor.
          </p>
        </div>
        <div class="rounded-2xl border border-capitol-200 dark:border-capitol-800 bg-capitol-900 dark:bg-capitol-950 overflow-hidden"
             style="box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 20px 40px -15px rgba(0,0,0,0.2);">
          <div class="flex items-center gap-1.5 px-4 py-2.5 border-b border-capitol-800 bg-capitol-950/60">
            <span class="w-2.5 h-2.5 rounded-full bg-red-500/70"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-amber-500/70"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500/70"></span>
            <span class="font-mono text-[0.7rem] text-capitol-400 ml-3">terminal · ollama</span>
          </div>
          <pre class="font-mono text-[13px] leading-6 p-5 text-capitol-200"><span class="text-capitol-500"># Pull three representatives to seat the chamber.</span>
<span class="text-gold-400">$</span> ollama pull phi3:3.8b
<span class="text-gold-400">$</span> ollama pull mistral:7b
<span class="text-gold-400">$</span> ollama pull llama3.2:3b</pre>
        </div>
      </section>
    {:else}
      <!-- Tab Navigation -->
      <div class="mb-8">
        <div class="inline-flex p-1 bg-capitol-100 dark:bg-capitol-800/50 rounded-xl border border-capitol-200 dark:border-capitol-700">
          <button
            on:click={() => activeTab = 'models'}
            class="px-5 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center gap-2 active:scale-[0.98] {activeTab === 'models' ? 'bg-white dark:bg-capitol-700 text-capitol-900 dark:text-white shadow-sm' : 'text-capitol-600 dark:text-capitol-400 hover:text-capitol-900 dark:hover:text-white'}"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/>
            </svg>
            Model Assembly
          </button>
          <button
            on:click={() => activeTab = 'personalities'}
            class="px-5 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center gap-2 active:scale-[0.98] {activeTab === 'personalities' ? 'bg-white dark:bg-capitol-700 text-capitol-900 dark:text-white shadow-sm' : 'text-capitol-600 dark:text-capitol-400 hover:text-capitol-900 dark:hover:text-white'}"
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
  <footer class="border-t border-capitol-200 dark:border-capitol-800 bg-white/50 dark:bg-capitol-900/50">
    <div class="max-w-7xl mx-auto px-6 lg:px-8 py-5">
      <div class="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-capitol-500 dark:text-capitol-400">
        <div class="flex items-center gap-2">
          <span class="font-display font-semibold text-capitol-700 dark:text-capitol-300">AI Congress</span>
          <span class="opacity-30">|</span>
          <span>Open-source multi-agent system</span>
        </div>
        <div class="flex items-center gap-4">
          <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" class="hover:text-capitol-700 dark:hover:text-capitol-300 transition-colors">
            Powered by Ollama
          </a>
          <a href="https://github.com/jasperan/ai-congress" target="_blank" rel="noopener noreferrer" class="hover:text-capitol-700 dark:hover:text-capitol-300 transition-colors">
            GitHub
          </a>
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
    font-family: 'Cinzel', 'Iowan Old Style', Georgia, serif;
    letter-spacing: 0.01em;
  }

  :global(.font-sans) {
    font-family: 'Geist', 'Geist Fallback', ui-sans-serif, system-ui, -apple-system, sans-serif;
    font-feature-settings: "ss01", "cv11";
  }

  :global(.font-mono) {
    font-family: 'Geist Mono', ui-monospace, 'JetBrains Mono', monospace;
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
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  :global(.animate-fade-in) {
    animation: fade-in 0.35s cubic-bezier(0.4, 0, 0.2, 1) forwards;
  }

  /* Skeletal shimmer for loading state - moves a highlight band across pill-shaped rows. */
  @keyframes skeleton-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  :global(.animate-skeleton-shimmer) {
    animation: skeleton-shimmer 1.6s linear infinite;
    will-change: background-position;
  }

  :global(.tabular-nums) {
    font-variant-numeric: tabular-nums;
  }

  :global(.active\:scale-95:active) {
    transform: scale(0.95);
  }

  :global(.active\:scale-\[0\.98\]:active) {
    transform: scale(0.98);
  }

  @media (prefers-reduced-motion: reduce) {
    :global(*) {
      animation-duration: 0.01ms !important;
      transition-duration: 0.01ms !important;
    }
  }
</style>
