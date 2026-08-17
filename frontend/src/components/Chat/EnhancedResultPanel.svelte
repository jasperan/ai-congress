<script>
  // EnhancedResultPanel — surfaces the enhanced pipeline's observability
  // artifacts (3.7.1): minority report, decision explanation, performance
  // profile waterfall, and the run's event log.
  export let result = null

  $: data = result?.data || result || {}
  $: minority = data.minority_report || {}
  $: explanation = data.decision_explanation || ''
  $: profile = data.performance_profile || {}
  $: eventLog = data.event_log || []
  $: reasoningMode = data.reasoning_mode || ''
  $: duration = data.duration_seconds || 0
  $: precedent = data.precedent || null

  $: stages = profile.stages || []
  $: slowest = profile.slowest_stage || ''
  $: totalMs = profile.total_ms || 0

  $: minorityText = minority.summary || minority.text || ''
  $: minorityModels = minority.minority_models || []
  $: maxStageMs = Math.max(1, ...stages.map(s => s.duration_ms || 0))
</script>

<div class="space-y-4">
  {#if reasoningMode || duration}
    <div class="flex flex-wrap items-center gap-2">
      {#if reasoningMode}
        <span class="text-xs px-2 py-0.5 rounded-full bg-capitol-100 dark:bg-capitol-800 text-capitol-700 dark:text-capitol-300 border border-capitol-200 dark:border-capitol-700 font-medium">
          reasoning: {reasoningMode}
        </span>
      {/if}
      <span class="text-xs px-2 py-0.5 rounded-full bg-surface-100 dark:bg-surface-800 text-text-secondary dark:text-text-tertiary border border-surface-200 dark:border-surface-700 font-medium">
        ⏱ {duration.toFixed(1)}s
      </span>
    </div>
  {/if}

  <!-- Minority Report -->
  {#if minorityText}
    <div class="rounded-xl border border-warning-300 dark:border-warning-700 bg-warning-50 dark:bg-warning-900/20 p-4">
      <h4 class="text-sm font-bold text-warning-800 dark:text-warning-300">🗣️ Minority Report</h4>
      {#if minorityModels.length > 0}
        <p class="text-xs text-warning-700 dark:text-warning-400 mt-1">
          Raised by: {minorityModels.join(', ')}
        </p>
      {/if}
      <p class="text-sm text-warning-900 dark:text-warning-100 mt-2 whitespace-pre-wrap">{minorityText}</p>
    </div>
  {/if}

  <!-- Decision Explanation -->
  {#if explanation}
    <div class="card p-4">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary mb-1">🧠 Decision Explanation</h4>
      <p class="text-sm text-text-secondary dark:text-text-tertiary whitespace-pre-wrap">{explanation}</p>
    </div>
  {/if}

  <!-- Precedent -->
  {#if precedent && precedent.cited}
    <div class="card p-4">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary mb-1">⚖️ Precedent Cited (stare decisis)</h4>
      <p class="text-sm text-text-secondary dark:text-text-tertiary">
        {precedent.question || 'Prior ruling'} — action: {precedent.action || 'cited'}
      </p>
    </div>
  {/if}

  <!-- Performance Waterfall -->
  {#if stages.length > 0}
    <div class="card p-4 space-y-2">
      <div class="flex items-center justify-between">
        <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">📊 Pipeline Profile</h4>
        <span class="text-xs text-text-secondary dark:text-text-tertiary">
          total {(totalMs / 1000).toFixed(1)}s{slowest ? ` · bottleneck: ${slowest}` : ''}
        </span>
      </div>
      <div class="space-y-1.5">
        {#each stages as stage}
          <div class="flex items-center gap-2">
            <span class="w-40 text-[11px] text-text-secondary dark:text-text-tertiary truncate text-right shrink-0">{stage.name}</span>
            <div class="flex-1 h-2.5 rounded bg-surface-100 dark:bg-surface-800 overflow-hidden">
              <div
                class="h-full rounded {stage.name === slowest ? 'bg-danger-400' : 'bg-primary-500'}"
                style="width: {Math.max(2, ((stage.duration_ms || 0) / maxStageMs) * 100)}%"
              ></div>
            </div>
            <span class="w-14 text-[11px] text-text-secondary dark:text-text-tertiary shrink-0">
              {(stage.duration_ms || 0) / 1000}s
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Event Log -->
  {#if eventLog.length > 0}
    <div class="card p-4">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary mb-2">📜 Event Log ({eventLog.length})</h4>
      <div class="space-y-1 max-h-56 overflow-y-auto custom-scrollbar">
        {#each eventLog as event, i}
          <div class="flex items-start gap-2 text-xs">
            <span class="text-text-tertiary font-mono whitespace-nowrap shrink-0">{i + 1}</span>
            <span class="font-mono font-semibold text-primary-600 dark:text-primary-400 shrink-0">{event.event_type}</span>
            <span class="text-text-secondary dark:text-text-tertiary break-words">{event.detail || event.message || ''}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>
