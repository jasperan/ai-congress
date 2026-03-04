<script>
  export let response
  export let index = 0

  $: isSuccess = response.success
  $: modelName = response.personality_name || response.model || 'Unknown'
  $: content = response.response || response.error || 'No response'
  $: temperature = response.temperature || null
</script>

<div class="card p-4 space-y-3 animate-fade-in-up" style="animation-delay: {index * 0.15}s">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="flex-shrink-0">
        <div class="w-8 h-8 rounded-full bg-gradient-to-br from-secondary-500 to-primary-500 dark:from-secondary-400 dark:to-primary-400 flex items-center justify-center text-white text-sm font-bold shadow-md no-select" role="img" aria-label={`${modelName} icon`}>
          {modelName.charAt(0).toUpperCase()}
        </div>
      </div>

      <div>
        <h4 class="text-sm font-semibold text-text-primary dark:text-text-primary">
          {modelName}
        </h4>
        {#if temperature !== null}
          <span class="text-xs text-text-secondary dark:text-text-tertiary">
            Temp: {temperature}
          </span>
        {/if}
      </div>
    </div>

    <!-- Status Badge -->
    <span class="badge {isSuccess ? 'badge-success' : 'badge-danger'}">
      {isSuccess ? '✓ Success' : '✗ Error'}
    </span>
  </div>

  <!-- Response Content -->
  <div class="mt-3">
    {#if isSuccess}
      <div class="text-sm text-text-primary dark:text-text-primary leading-relaxed p-4 bg-surface-50 dark:bg-surface-700/30 rounded-lg border border-surface-200 dark:border-surface-600">
        <p class="whitespace-pre-wrap">{content}</p>
      </div>
    {:else}
      <div class="text-sm text-danger-600 dark:text-danger-400 p-4 bg-danger-50 dark:bg-danger-900/20 rounded-lg border border-danger-200 dark:border-danger-800">
        <p class="font-medium">Error:</p>
        <p class="mt-1">{content}</p>
      </div>
    {/if}
  </div>

  <!-- Metadata Footer -->
  {#if isSuccess}
    <div class="flex items-center justify-between text-xs text-text-secondary dark:text-text-tertiary pt-2 border-t border-surface-200 dark:border-surface-700">
      <span>Response length: {content.length} chars</span>
      {#if response.weight}
        <span class="font-medium text-success-600 dark:text-success-400">
          Weight: {(response.weight * 100).toFixed(1)}%
        </span>
      {/if}
    </div>
  {/if}
</div>
