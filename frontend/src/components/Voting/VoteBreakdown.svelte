<script>
  export let voteBreakdown = {}
  export let confidence = 0
  export let semanticConfidence = null
  export let semanticVote = null

  $: votes = Object.entries(voteBreakdown)
  $: confidenceColor = confidence > 0.8 ? 'bg-success-500' : confidence > 0.6 ? 'bg-warning-500' : 'bg-danger-500'
  $: confidenceText = confidence > 0.8 ? 'High' : confidence > 0.6 ? 'Medium' : 'Low'
  $: semanticConfidenceColor = semanticConfidence !== null
      ? (semanticConfidence > 0.8 ? 'bg-success-500' : semanticConfidence > 0.6 ? 'bg-warning-500' : 'bg-danger-500')
      : null
  $: semanticConfidenceText = semanticConfidence !== null
      ? (semanticConfidence > 0.8 ? 'High' : semanticConfidence > 0.6 ? 'Medium' : 'Low')
      : null
  $: showSemanticWarning = semanticConfidence !== null && semanticConfidence < 0.6

  // Semantic vote data
  $: clusters = semanticVote?.clusters || []
  $: debateTriggered = semanticVote?.debate_triggered || false
  $: debateRounds = semanticVote?.debate_rounds || 0
  $: debateTranscript = semanticVote?.debate_transcript || []
  $: convictionScores = semanticVote?.conviction_scores || {}
  $: dissentingSummary = semanticVote?.dissenting_summary || ''
  $: winningModel = semanticVote?.winning_model || ''

  let showDebateTranscript = false
</script>

<div class="card p-4 space-y-4 animate-fade-in">
  <div class="flex items-center justify-between">
    <h3 class="text-lg font-semibold text-text-primary dark:text-text-primary">
      Vote Breakdown
    </h3>
    <div class="flex items-center space-x-2">
      <span class="text-sm text-text-secondary dark:text-text-tertiary">Confidence:</span>
      <span class="badge {confidenceColor} text-white px-3 py-1">
        {confidenceText} ({(confidence * 100).toFixed(1)}%)
      </span>
    </div>
  </div>

  <!-- Confidence Bar -->
  <div class="space-y-2">
    <div class="progress-bar">
      <div class="progress-fill {confidenceColor}" style="width: {confidence * 100}%"></div>
    </div>
  </div>

  <!-- Semantic Confidence (if available) -->
  {#if semanticConfidence !== null}
    <div class="space-y-2">
      <div class="flex items-center justify-between">
        <span class="text-sm text-text-secondary dark:text-text-tertiary">Semantic Confidence:</span>
        <span class="badge {semanticConfidenceColor} text-white px-3 py-1">
          {semanticConfidenceText} ({(semanticConfidence * 100).toFixed(1)}%)
        </span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill {semanticConfidenceColor}" style="width: {semanticConfidence * 100}%"></div>
      </div>

      {#if showSemanticWarning}
        <div class="flex items-start space-x-2 p-3 rounded-lg bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-700">
          <svg class="w-5 h-5 text-warning-600 dark:text-warning-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
          <div class="text-sm">
            <p class="font-medium text-warning-800 dark:text-warning-200">
              Low Semantic Agreement
            </p>
            <p class="text-warning-700 dark:text-warning-300">
              Responses show low semantic similarity despite voting consensus. Consider reviewing individual responses for nuanced differences.
            </p>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Semantic Clusters -->
  {#if clusters.length > 0}
    <div class="space-y-3">
      <h4 class="text-sm font-semibold text-text-primary dark:text-text-primary flex items-center space-x-2">
        <svg class="w-4 h-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
        </svg>
        <span>Semantic Clusters ({clusters.length})</span>
      </h4>

      {#each clusters as cluster, i}
        <div
          class="p-3 rounded-lg border animate-fade-in-up {cluster.models?.includes(winningModel) ? 'bg-success-50 dark:bg-success-900/20 border-success-300 dark:border-success-700' : 'bg-surface-50 dark:bg-surface-700/50 border-surface-200 dark:border-surface-600'}"
          style="animation-delay: {i * 0.08}s"
        >
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center space-x-2">
              {#if cluster.models?.includes(winningModel)}
                <span class="text-xs px-2 py-0.5 rounded-full bg-success-500 text-white font-medium">Winner</span>
              {/if}
              <span class="text-sm font-medium text-text-primary dark:text-text-primary">
                {cluster.label || `Cluster ${cluster.id}`}
              </span>
            </div>
            <span class="text-xs text-text-secondary dark:text-text-tertiary">
              {cluster.models?.length || 0} model{(cluster.models?.length || 0) !== 1 ? 's' : ''}
            </span>
          </div>

          <!-- Models in cluster -->
          <div class="flex flex-wrap gap-1.5 mb-2">
            {#each (cluster.models || []) as model}
              <span class="text-xs px-2 py-0.5 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 font-medium {model === winningModel ? 'ring-1 ring-success-400' : ''}">
                {model}
                {#if convictionScores[model]}
                  <span class="opacity-70 ml-0.5" title="Conviction score">({convictionScores[model].toFixed(2)}x)</span>
                {/if}
              </span>
            {/each}
          </div>

          <!-- Key claims -->
          {#if cluster.key_claims && cluster.key_claims.length > 0}
            <div class="text-xs text-text-secondary dark:text-text-tertiary space-y-0.5">
              {#each cluster.key_claims as claim}
                <p class="flex items-start space-x-1">
                  <span class="text-primary-400 mt-0.5 flex-shrink-0">-</span>
                  <span>{claim}</span>
                </p>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <!-- Debate Info -->
  {#if debateTriggered}
    <div class="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/15 border border-amber-200 dark:border-amber-800 space-y-2">
      <div class="flex items-center justify-between">
        <h4 class="text-sm font-semibold text-amber-800 dark:text-amber-200 flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
          </svg>
          <span>Debate Escalation</span>
        </h4>
        <span class="text-xs px-2 py-0.5 rounded-full bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 font-medium">
          {debateRounds} round{debateRounds !== 1 ? 's' : ''}
        </span>
      </div>

      <p class="text-xs text-amber-700 dark:text-amber-300">
        Consensus was below threshold. Models debated for {debateRounds} round{debateRounds !== 1 ? 's' : ''} with escalating commitment pressure.
      </p>

      <!-- Dissenting Summary -->
      {#if dissentingSummary}
        <div class="text-xs p-2 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200">
          <span class="font-medium">Dissenting view:</span> {dissentingSummary}
        </div>
      {/if}

      <!-- Debate Transcript Toggle -->
      {#if debateTranscript.length > 0}
        <button
          on:click={() => showDebateTranscript = !showDebateTranscript}
          class="text-xs text-amber-700 dark:text-amber-300 hover:underline flex items-center space-x-1 focus:outline-none"
          aria-expanded={showDebateTranscript}
        >
          <svg class="w-3 h-3 transition-transform {showDebateTranscript ? 'rotate-90' : ''}" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"/>
          </svg>
          <span>{showDebateTranscript ? 'Hide' : 'Show'} debate transcript</span>
        </button>

        {#if showDebateTranscript}
          <div class="space-y-2 mt-2">
            {#each debateTranscript as entry, i}
              <div class="text-xs p-2 rounded bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700">
                <div class="flex items-center justify-between mb-1">
                  <span class="font-medium text-text-primary dark:text-text-primary">
                    Round {entry.round || i + 1} - {entry.model || 'Unknown'}
                  </span>
                  {#if entry.indecisive}
                    <span class="px-1.5 py-0.5 rounded bg-warning-100 dark:bg-warning-900/30 text-warning-700 dark:text-warning-300 text-[10px] font-medium">
                      Indecisive
                    </span>
                  {/if}
                </div>
                <p class="text-text-secondary dark:text-text-tertiary leading-relaxed">
                  {entry.response || entry.content || ''}
                </p>
              </div>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  {/if}

  <!-- Conviction Scores -->
  {#if Object.keys(convictionScores).length > 0}
    <div class="space-y-2">
      <h4 class="text-sm font-semibold text-text-primary dark:text-text-primary">
        Conviction Scores
      </h4>
      <div class="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {#each Object.entries(convictionScores) as [model, score]}
          <div class="flex items-center justify-between p-2 rounded-lg bg-surface-50 dark:bg-surface-700/50 border border-surface-200 dark:border-surface-600 text-xs">
            <span class="font-medium text-text-primary dark:text-text-primary truncate mr-2">{model}</span>
            <span class="font-semibold whitespace-nowrap {score > 1.0 ? 'text-success-600 dark:text-success-400' : 'text-text-secondary dark:text-text-tertiary'}">
              {score.toFixed(2)}x
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Vote Details -->
  {#if votes.length > 0}
    <div class="space-y-3">
      {#each votes as [model, data], i}
        <div class="flex items-center space-x-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50 border border-surface-200 dark:border-surface-600 animate-fade-in-up" style="animation-delay: {i * 0.1}s">
          <div class="flex-shrink-0">
            <div class="w-10 h-10 rounded-full bg-secondary-500 dark:bg-secondary-400 flex items-center justify-center text-white font-bold no-select" aria-hidden="true">
              {model.charAt(0).toUpperCase()}
            </div>
          </div>

          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between mb-1">
              <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate">
                {model}
              </p>
              <span class="text-sm font-semibold text-success-600 dark:text-success-400">
                {(data.weight * 100).toFixed(1)}%
              </span>
            </div>

            <!-- Vote weight bar -->
            <div class="h-1.5 bg-surface-200 dark:bg-surface-600 rounded-full overflow-hidden">
              <div
                class="h-full bg-gradient-to-r from-success-500 to-success-400 transition-all duration-500"
                style="width: {(data.weight * 100)}%"
              ></div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {:else if clusters.length === 0}
    <div class="text-center py-8 text-text-secondary dark:text-text-tertiary">
      <p>No vote data available yet</p>
      <p class="text-sm mt-1">Submit a query to see voting breakdown</p>
    </div>
  {/if}

  <!-- Algorithm Info -->
  <div class="mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
    <div class="flex items-start space-x-2 text-xs text-text-secondary dark:text-text-tertiary">
      <svg class="w-4 h-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
      </svg>
      <span>
        {#if clusters.length > 0}
          Semantic voting groups responses by meaning using an LLM judge. Conviction bonuses reward models that maintain consistent positions across debate rounds.
        {:else}
          Votes are weighted by model accuracy scores from MMLU benchmarks.
          Higher-performing models have more influence on the final decision.
        {/if}
      </span>
    </div>
  </div>
</div>
