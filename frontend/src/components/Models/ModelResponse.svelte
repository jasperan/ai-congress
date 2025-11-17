<script>
  export let response
  export let index = 0
  
  $: isSuccess = response.success
  $: modelName = response.model || 'Unknown'
  $: content = response.response || response.error || 'No response'
  $: temperature = response.temperature || null
</script>

<div class="card p-4 space-y-3 animate-slide-up" style="animation-delay: {index * 0.15}s">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="flex-shrink-0">
        <div class="w-8 h-8 rounded-full bg-gradient-to-br from-congress-model to-purple-400 
                    flex items-center justify-center text-white text-sm font-bold shadow-md">
          {modelName.charAt(0).toUpperCase()}
        </div>
      </div>
      
      <div>
        <h4 class="text-sm font-semibold text-gray-900 dark:text-gray-100">
          {modelName}
        </h4>
        {#if temperature !== null}
          <span class="text-xs text-gray-500 dark:text-gray-400">
            Temp: {temperature}
          </span>
        {/if}
      </div>
    </div>

    <!-- Status Badge -->
    <span class="badge {isSuccess ? 'badge-success' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}">
      {isSuccess ? '✓ Success' : '✗ Error'}
    </span>
  </div>

  <!-- Response Content -->
  <div class="mt-3">
    {#if isSuccess}
      <div class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed p-4 
                  bg-gray-50 dark:bg-gray-700/30 rounded-lg border border-gray-200 dark:border-gray-600">
        <p class="whitespace-pre-wrap">{content}</p>
      </div>
    {:else}
      <div class="text-sm text-red-600 dark:text-red-400 p-4 
                  bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
        <p class="font-medium">Error:</p>
        <p class="mt-1">{content}</p>
      </div>
    {/if}
  </div>

  <!-- Metadata Footer -->
  {#if isSuccess}
    <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 pt-2 
                border-t border-gray-200 dark:border-gray-700">
      <span>Response length: {content.length} chars</span>
      {#if response.weight}
        <span class="font-medium text-congress-vote">
          Weight: {(response.weight * 100).toFixed(1)}%
        </span>
      {/if}
    </div>
  {/if}
</div>

