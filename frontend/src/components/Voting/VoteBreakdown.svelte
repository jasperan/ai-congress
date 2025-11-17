<script>
  export let voteBreakdown = {}
  export let confidence = 0
  export let semanticConfidence = null

  $: votes = Object.entries(voteBreakdown)
  $: confidenceColor = confidence > 0.8 ? 'bg-green-500' : confidence > 0.6 ? 'bg-yellow-500' : 'bg-orange-500'
  $: confidenceText = confidence > 0.8 ? 'High' : confidence > 0.6 ? 'Medium' : 'Low'
  $: semanticConfidenceColor = semanticConfidence !== null
      ? (semanticConfidence > 0.8 ? 'bg-green-500' : semanticConfidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500')
      : null
  $: semanticConfidenceText = semanticConfidence !== null
      ? (semanticConfidence > 0.8 ? 'High' : semanticConfidence > 0.6 ? 'Medium' : 'Low')
      : null
  $: showSemanticWarning = semanticConfidence !== null && semanticConfidence < 0.6
</script>

<div class="card p-4 space-y-4 animate-fade-in">
  <div class="flex items-center justify-between">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
      🗳️ Vote Breakdown
    </h3>
    <div class="flex items-center space-x-2">
      <span class="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
      <span class="badge {confidenceColor} text-white px-3 py-1">
        {confidenceText} ({(confidence * 100).toFixed(1)}%)
      </span>
    </div>
  </div>

  <!-- Confidence Bar -->
  <div class="space-y-2">
    <div class="confidence-bar">
      <div
        class="confidence-fill {confidenceColor}"
        style="width: {confidence * 100}%"
      ></div>
    </div>
  </div>

  <!-- Semantic Confidence (if available) -->
  {#if semanticConfidence !== null}
    <div class="space-y-2">
      <div class="flex items-center justify-between">
        <span class="text-sm text-gray-600 dark:text-gray-400">Semantic Confidence:</span>
        <span class="badge {semanticConfidenceColor} text-white px-3 py-1">
          {semanticConfidenceText} ({(semanticConfidence * 100).toFixed(1)}%)
        </span>
      </div>
      <div class="confidence-bar">
        <div
          class="confidence-fill {semanticConfidenceColor}"
          style="width: {semanticConfidence * 100}%"
        ></div>
      </div>

      <!-- Semantic Confidence Warning -->
      {#if showSemanticWarning}
        <div class="flex items-start space-x-2 p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20
                    border border-yellow-200 dark:border-yellow-700">
          <svg class="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
          <div class="text-sm">
            <p class="font-medium text-yellow-800 dark:text-yellow-200">
              Low Semantic Agreement
            </p>
            <p class="text-yellow-700 dark:text-yellow-300">
              Responses show low semantic similarity despite voting consensus. Consider reviewing individual responses for nuanced differences.
            </p>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Vote Details -->
  {#if votes.length > 0}
    <div class="space-y-3">
      {#each votes as [model, data], i}
        <div class="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 
                    border border-gray-200 dark:border-gray-600 animate-slide-up"
             style="animation-delay: {i * 0.1}s">
          <div class="flex-shrink-0">
            <div class="w-10 h-10 rounded-full bg-congress-model flex items-center justify-center text-white font-bold">
              {model.charAt(0).toUpperCase()}
            </div>
          </div>
          
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between mb-1">
              <p class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {model}
              </p>
              <span class="text-sm font-semibold text-congress-vote">
                {(data.weight * 100).toFixed(1)}%
              </span>
            </div>
            
            <!-- Vote weight bar -->
            <div class="h-1.5 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
              <div 
                class="h-full bg-gradient-to-r from-congress-vote to-green-400 transition-all duration-500"
                style="width: {(data.weight * 100)}%"
              ></div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="text-center py-8 text-gray-500 dark:text-gray-400">
      <p>No vote data available yet</p>
      <p class="text-sm mt-1">Submit a query to see voting breakdown</p>
    </div>
  {/if}

  <!-- Algorithm Info -->
  <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
    <div class="flex items-start space-x-2 text-xs text-gray-600 dark:text-gray-400">
      <svg class="w-4 h-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
      </svg>
      <span>
        Votes are weighted by model accuracy scores from MMLU benchmarks. 
        Higher-performing models have more influence on the final decision.
      </span>
    </div>
  </div>
</div>
