<script>
  // SourcesPanel — renders RAG chunk attribution (3.7.1 / 3.6.3):
  // which documents grounded the answer, with similarity + snippet.
  export let sources = []
  export let webSearchResults = []

  $: total = sources.length + webSearchResults.length
</script>

{#if total > 0}
  <div class="card p-4 space-y-3">
    <div class="flex items-center justify-between">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">
        📚 Sources
      </h4>
      <span class="text-xs text-text-secondary dark:text-text-tertiary">{total} cited</span>
    </div>

    {#if sources.length > 0}
      <div class="space-y-2">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">RAG Documents</p>
        {#each sources as source, i}
          <div class="rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700 p-2.5">
            <div class="flex items-center justify-between gap-2">
              <p class="text-xs font-medium text-primary-700 dark:text-primary-300 break-all">
                📄 {source.document_id || `chunk ${i + 1}`}
              </p>
              <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 border border-primary-200 dark:border-primary-800 font-medium whitespace-nowrap">
                {Math.round((source.similarity || 0) * 100)}% match
              </span>
            </div>
            {#if source.snippet}
              <p class="text-xs text-text-secondary dark:text-text-tertiary mt-1 line-clamp-3">{source.snippet}</p>
            {/if}
          </div>
        {/each}
      </div>
    {/if}

    {#if webSearchResults.length > 0}
      <div class="space-y-2">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">Web Search</p>
        {#each webSearchResults as item}
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            class="block rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700 p-2.5 hover:border-primary-400 transition-colors"
          >
            <p class="text-xs font-medium text-primary-700 dark:text-primary-300">{item.title}</p>
            <p class="text-xs text-text-secondary dark:text-text-tertiary mt-0.5 line-clamp-2">{item.description}</p>
          </a>
        {/each}
      </div>
    {/if}
  </div>
{/if}
