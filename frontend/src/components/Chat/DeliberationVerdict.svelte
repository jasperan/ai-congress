<script>
  // DeliberationVerdict — renders the 3-round council verdict structure (4.3.2):
  // Question-Reframing Warning, Unresolved Questions, Recommended Next Steps,
  // Steelmanned Dissent, Final Positions (with evidence alignment), and the
  // Weighted Majority winner, plus Round-2 engagement compliance stats.
  export let result = null

  $: verdict = result?.verdict || ''
  $: restate = result?.restate || null
  $: dissentReport = result?.dissent_report || null
  $: steelman = result?.steelman || []
  $: finalPositions = result?.responses || []
  $: rounds = result?.rounds || []
  $: engagement = result?.engagement_compliance || null
  $: metadata = result?.metadata || {}
  $: finalAnswer = result?.final_answer || ''
  $: confidence = result?.confidence || 0

  $: restateWarning = restate?.warning || ''
  $: restates = restate?.restates || []
  $: alternativeFramings = restates.filter(r => r.alt_framing)

  // Engagement compliance summary
  $: compliancePct = engagement && engagement.total
      ? Math.round((engagement.compliant / engagement.total) * 100)
      : null

  // Evidence alignment badge color
  function alignmentColor(score) {
    if (score == null) return ''
    if (score >= 0.7) return 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300 border-success-300 dark:border-success-800'
    if (score >= 0.4) return 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300 border-warning-300 dark:border-warning-800'
    return 'bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300 border-danger-300 dark:border-danger-800'
  }

  function roundLabel(name) {
    return { round1: 'Round 1 — Independent Analysis', round2: 'Round 2 — Cross-Examination', round3: 'Round 3 — Final Position', steelman: 'Steelmanned Dissent' }[name] || name
  }
</script>

<div class="space-y-6">
  <!-- Verdict header -->
  <div class="flex items-center justify-between flex-wrap gap-2">
    <h3 class="text-lg font-bold text-text-primary dark:text-text-primary">
      🏛️ Council Verdict
    </h3>
    <div class="flex items-center space-x-2">
      {#if metadata.evidence_grounded}
        <span class="text-xs px-2 py-0.5 rounded-full bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 border border-primary-200 dark:border-primary-800 font-medium">
          🌐 evidence-grounded
        </span>
      {/if}
      {#if compliancePct !== null}
        <span class="text-xs px-2 py-0.5 rounded-full {compliancePct === 100 ? 'bg-success-50 dark:bg-success-900/30 text-success-700 dark:text-success-300 border border-success-200 dark:border-success-800' : 'bg-warning-50 dark:bg-warning-900/30 text-warning-700 dark:text-warning-300 border border-warning-200 dark:border-warning-800'} font-medium" title="Round-2 protocol compliance: each member must engage ≥ {engagement?.min_peers} peers by name">
          {compliancePct}% engaged{engagement?.re_prompted ? ` · ${engagement.re_prompted} re-prompted` : ''}
        </span>
      {/if}
    </div>
  </div>

  <!-- Question-Reframing Warning -->
  {#if restateWarning}
    <div class="rounded-xl border border-warning-300 dark:border-warning-700 bg-warning-50 dark:bg-warning-900/20 p-4 space-y-2">
      <h4 class="text-sm font-semibold text-warning-800 dark:text-warning-300">⚠️ Question-Reframing Warning</h4>
      <p class="text-sm text-warning-800 dark:text-warning-200">{restateWarning}</p>
      {#if alternativeFramings.length > 0}
        <div class="pt-1 space-y-1">
          <p class="text-xs font-medium text-warning-700 dark:text-warning-400">Alternative framings proposed:</p>
          {#each alternativeFramings as r}
            <p class="text-sm text-warning-800 dark:text-warning-200">
              <span class="font-semibold">{r.agent}:</span> {r.alt_framing}
            </p>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Unresolved Questions -->
  {#if verdict}
    <div class="card p-4 space-y-2">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">Unresolved Questions</h4>
      {#if verdict.includes('## Unresolved Questions')}
        <p class="text-sm text-text-secondary dark:text-text-tertiary whitespace-pre-wrap">{verdict}</p>
      {:else}
        <p class="text-sm text-text-secondary dark:text-text-tertiary whitespace-pre-wrap">{finalAnswer}</p>
      {/if}
    </div>
  {:else}
    <p class="text-sm text-text-secondary dark:text-text-tertiary">(no verdict returned)</p>
  {/if}

  <!-- Steelmanned Dissent -->
  {#if steelman && steelman.length > 0}
    <div class="card p-4 space-y-2">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">💥 Steelmanned Dissent</h4>
      <p class="text-xs text-text-secondary dark:text-text-tertiary">
        Premature consensus detected after Round 1 — these members were forced to argue the strongest opposing view.
      </p>
      {#each steelman as item}
        <div class="rounded-lg bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 p-3">
          <p class="text-xs font-semibold text-primary-600 dark:text-primary-400 mb-1">
            {item.agent || item.role || item.model}
          </p>
          <p class="text-sm text-text-primary dark:text-text-primary whitespace-pre-wrap">{item.response}</p>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Dissent report -->
  {#if dissentReport}
    <div class="card p-4">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary mb-2">Round-1 Agreement</h4>
      <p class="text-sm text-text-secondary dark:text-text-tertiary">
        {Math.round((dissentReport.agreement_ratio || 0) * 100)}% agreement
        {dissentReport.method ? ` (method: ${dissentReport.method})` : ''}
        {dissentReport.premature ? ' · ⚠️ premature consensus' : ''}
      </p>
    </div>
  {/if}

  <!-- Final Positions -->
  {#if finalPositions.length > 0}
    <div class="card p-4 space-y-3">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">Final Positions (Round 3)</h4>
      {#each finalPositions as pos, i}
        {#if pos.success}
          <div class="rounded-lg bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 p-3">
            <div class="flex items-center justify-between flex-wrap gap-2 mb-1">
              <p class="text-xs font-semibold text-primary-600 dark:text-primary-400">
                {pos.agent || pos.role || pos.model}
                {#if pos.model && pos.model !== (pos.agent || '').split('@')[1]}
                  <span class="text-text-tertiary font-normal">({pos.model})</span>
                {/if}
              </p>
              {#if pos.evidence_alignment != null}
                <span class="text-[10px] px-2 py-0.5 rounded-full border font-medium {alignmentColor(pos.evidence_alignment)}">
                  evidence {Math.round(pos.evidence_alignment * 100)}%
                </span>
              {/if}
            </div>
            <p class="text-sm text-text-primary dark:text-text-primary whitespace-pre-wrap">{pos.response}</p>
          </div>
        {/if}
      {/each}
    </div>
  {/if}

  <!-- Weighted Majority -->
  {#if finalAnswer}
    <div class="rounded-xl border border-success-300 dark:border-success-800 bg-success-50 dark:bg-success-900/20 p-4">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <h4 class="text-sm font-bold text-success-800 dark:text-success-300">🏛️ Weighted Majority</h4>
        <span class="badge bg-success-500 text-white px-3 py-1">
          {Math.round(confidence * 100)}% confidence
        </span>
      </div>
      <p class="text-sm text-success-900 dark:text-success-100 mt-2 whitespace-pre-wrap">{finalAnswer}</p>
      {#if dissentReport}
        <p class="text-xs text-success-700 dark:text-success-300 mt-2 italic">
          Round-1 agreement {Math.round((dissentReport.agreement_ratio || 0) * 100)}% — consensus ranking placed last because the council's disagreements matter more than where it agrees.
        </p>
      {/if}
    </div>
  {/if}

  <!-- Rounds transcript -->
  {#if rounds && rounds.length > 0}
    <div class="card p-4 space-y-4">
      <h4 class="text-sm font-bold text-text-primary dark:text-text-primary">Debate Transcript</h4>
      {#each rounds as round}
        <div class="space-y-2">
          <h5 class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">{roundLabel(round.name)}</h5>
          {#each round.outputs as o}
            {#if o.success}
              <div class="rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700 p-2.5">
                <div class="flex items-center justify-between flex-wrap gap-1">
                  <p class="text-[11px] font-semibold text-text-secondary dark:text-text-tertiary">{o.agent}</p>
                  {#if o.engagement}
                    <span class="text-[10px] px-1.5 py-0.5 rounded-full border font-medium {o.engagement.compliant ? 'bg-success-50 dark:bg-success-900/30 text-success-700 dark:text-success-300 border-success-200 dark:border-success-800' : 'bg-danger-50 dark:bg-danger-900/30 text-danger-700 dark:text-danger-300 border-danger-200 dark:border-danger-800'}">
                      {o.engagement.compliant ? '✓ engaged' : '✗ not engaged'}{o.engagement.re_prompted ? ' · re-prompted' : ''}
                    </span>
                  {/if}
                </div>
                <p class="text-sm text-text-primary dark:text-text-primary mt-1 whitespace-pre-wrap">{o.response}</p>
              </div>
            {/if}
          {/each}
        </div>
      {/each}
    </div>
  {/if}
</div>
