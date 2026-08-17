<script>
  // ObservabilityDashboard — the congress "control room" (3.7.5):
  // leaderboard, circuit breakers, calibration, recent runs, event logger.
  import { onMount } from 'svelte'

  let summary = null
  let loading = true
  let error = null
  let autoRefresh = true
  let refreshTimer = null

  $: leaderboard = summary?.leaderboard || []
  $: breakers = summary?.circuit_breakers || {}
  $: breakerOpen = summary?.breaker_open_count || 0
  $: calibration = summary?.calibration || {}
  $: runs = summary?.recent_runs || []
  $: eventLogger = summary?.event_logger || {}
  $: calibrationCurves = summary?.calibration_curves || []
  $: domainWinRates = summary?.domain_win_rates || []
  $: moeRouting = summary?.moe_routing || {}
  $: moeEntries = typeof moeRouting === 'object' && moeRouting !== null ? Object.entries(moeRouting) : []

  $: maxWeight = Math.max(0.01, ...leaderboard.map(r => r.weight || 0))
  $: maxParticipation = Math.max(1, ...leaderboard.map(r => r.participations || 0))
  $: maxDomainFeedback = Math.max(1, ...domainWinRates.map(d => d.feedback_count || 0))

  // SVG calibration curve geometry — clip accuracy dots into the visible band
  function curvePoints(points, w = 180, h = 44) {
    if (!points || points.length === 0) return ''
    const maxX = points.length > 1 ? points.length - 1 : 1
    return points
      .map((p, i) => {
        const x = 4 + (i / maxX) * (w - 8)
        const y = h - 4 - (Math.min(1, Math.max(0, p.accuracy || 0)) * (h - 8))
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(' ')
  }

  // Per-model calibration digest for the simple list (bin rows → obs + accuracy)
  function calibrationSummary(model) {
    const bins = Object.entries((calibration || {})[model] || {})
    const obs = bins.reduce((acc, [, b]) => acc + Number(b?.total || 0), 0)
    const correct = bins.reduce((acc, [, b]) => acc + Number(b?.correct || 0), 0)
    return { obs, acc: obs > 0 ? correct / obs : 0 }
  }

  function statusColor(state) {
    return {
      CLOSED: 'bg-success-50 dark:bg-success-900/30 text-success-700 dark:text-success-300 border-success-200 dark:border-success-800',
      OPEN: 'bg-danger-50 dark:bg-danger-900/30 text-danger-700 dark:text-danger-300 border-danger-200 dark:border-danger-800',
      HALF_OPEN: 'bg-warning-50 dark:bg-warning-900/30 text-warning-700 dark:text-warning-300 border-warning-200 dark:border-warning-800',
    }[state] || 'bg-surface-100 dark:bg-surface-800 text-text-secondary dark:text-text-tertiary border-surface-200 dark:border-surface-700'
  }

  async function load() {
    try {
      const response = await fetch('/api/observability/summary')
      if (!response.ok) throw new Error('Failed to load observability summary')
      summary = await response.json()
      error = null
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  onMount(() => {
    load()
    if (autoRefresh) {
      refreshTimer = setInterval(load, 15000)
    }
    return () => {
      if (refreshTimer) clearInterval(refreshTimer)
    }
  })
</script>

<div class="space-y-6">
  <div class="flex items-center justify-between flex-wrap gap-3">
    <div>
      <h2 class="text-xl font-bold text-text-primary dark:text-text-primary">🛰️ Observability</h2>
      <p class="text-sm text-text-secondary dark:text-text-tertiary">The congress control room — what the models have learned.</p>
    </div>
    <div class="flex items-center space-x-3">
      <label class="flex items-center space-x-2 text-sm text-text-secondary dark:text-text-tertiary cursor-pointer">
        <input
          type="checkbox"
          bind:checked={autoRefresh}
          on:change={() => {
            if (autoRefresh) refreshTimer = setInterval(load, 15000)
            else if (refreshTimer) clearInterval(refreshTimer)
          }}
          class="toggle-switch {autoRefresh ? 'checked' : ''}"
        />
        <span>Auto-refresh</span>
      </label>
      <button on:click={load} class="btn-primary text-sm px-4 py-2" disabled={loading}>
        {loading ? 'Refreshing…' : '↻ Refresh'}
      </button>
    </div>
  </div>

  {#if error}
    <div class="card p-4 bg-danger-50 dark:bg-danger-900/20 border-danger-200 dark:border-danger-800">
      <p class="text-sm text-danger-700 dark:text-danger-300">⚠️ {error}</p>
    </div>
  {/if}

  {#if loading && !summary}
    <div class="flex justify-center py-16">
      <div class="spinner h-8 w-8 text-primary-500" aria-label="Loading"></div>
    </div>
  {:else if summary}
    <!-- KPI row -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="card p-4">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">Models tracked</p>
        <p class="text-3xl font-bold text-text-primary dark:text-text-primary mt-1">{leaderboard.length}</p>
      </div>
      <div class="card p-4">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">Open breakers</p>
        <p class="text-3xl font-bold {breakerOpen > 0 ? 'text-danger-600 dark:text-danger-400' : 'text-success-600 dark:text-success-400'} mt-1">{breakerOpen}</p>
      </div>
      <div class="card p-4">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">Recent runs</p>
        <p class="text-3xl font-bold text-text-primary dark:text-text-primary mt-1">{runs.length}</p>
      </div>
      <div class="card p-4">
        <p class="text-xs font-semibold text-text-secondary dark:text-text-tertiary uppercase tracking-wide">Fallback events</p>
        <p class="text-3xl font-bold text-text-primary dark:text-text-primary mt-1">{eventLogger.fallback_hits || 0}</p>
        {#if eventLogger.fallback_hits > 0}
          <p class="text-[11px] text-warning-600 dark:text-warning-400">Oracle down → JSONL</p>
        {/if}
      </div>
    </div>

    <div class="grid lg:grid-cols-2 gap-6">
      <!-- Leaderboard -->
      <div class="card p-4 space-y-3">
        <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">🏆 Model Leaderboard</h3>
        {#if leaderboard.length === 0}
          <p class="text-sm text-text-secondary dark:text-text-tertiary">No learning data yet — run a session and the weights will adapt.</p>
        {:else}
          {#each leaderboard as row, i}
            <div class="flex items-center gap-3">
              <span class="w-5 text-sm font-bold text-text-tertiary text-right shrink-0">{i + 1}</span>
              <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between gap-2">
                  <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate">{row.model}</p>
                  <p class="text-xs text-text-secondary dark:text-text-tertiary whitespace-nowrap">
                    {row.win_rate > 0 ? `${Math.round(row.win_rate * 100)}% win · ` : ''}{row.participations} runs
                  </p>
                </div>
                <div class="h-2 rounded bg-surface-100 dark:bg-surface-800 overflow-hidden mt-1">
                  <div class="h-full rounded bg-primary-500" style="width: {Math.max(2, (row.weight / maxWeight) * 100)}%"></div>
                </div>
              </div>
              <span class="w-14 text-sm font-bold text-primary-600 dark:text-primary-400 text-right shrink-0">{(row.weight * 100).toFixed(0)}</span>
            </div>
          {/each}
        {/if}
      </div>

      <!-- Circuit breakers -->
      <div class="card p-4 space-y-3">
        <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">🛡️ Circuit Breakers</h3>
        {#if Object.keys(breakers).length === 0}
          <p class="text-sm text-text-secondary dark:text-text-tertiary">All breakers closed — no model has been failing.</p>
        {:else}
          {#each Object.entries(breakers) as [model, info]}
            <div class="flex items-center justify-between gap-3">
              <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate">{model}</p>
              <div class="flex items-center gap-2">
                {#if info.last_failure_age_s != null}
                  <span class="text-[11px] text-text-tertiary">{Math.round(info.last_failure_age_s)}s ago</span>
                {/if}
                <span class="text-[10px] px-2 py-0.5 rounded-full border font-medium {statusColor(info.state)}">{info.state}</span>
              </div>
            </div>
          {/each}
        {/if}

        <h3 class="text-sm font-bold text-text-primary dark:text-text-primary pt-3 border-t border-surface-200 dark:border-surface-700">📈 Calibration</h3>
        {#if Object.keys(calibration).length === 0}
          <p class="text-sm text-text-secondary dark:text-text-tertiary">No calibration observations yet.</p>
        {:else}
          {#each Object.entries(calibration) as [model]}
            {@const s = calibrationSummary(model)}
            <div class="flex items-center justify-between gap-3">
              <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate">{model}</p>
              <div class="flex items-center gap-2">
                {#if s.obs > 0}<span class="text-[11px] text-text-tertiary">{Math.round(s.acc * 100)}% acc</span>{/if}
                <span class="text-xs text-text-secondary dark:text-text-tertiary whitespace-nowrap">{s.obs} obs</span>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    </div>

    <!-- L3: Calibration curves + MoE routing -->
    <div class="grid lg:grid-cols-2 gap-6">
      <!-- Calibration curves -->
      <div class="card p-4 space-y-3">
        <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">📐 Calibration Curves</h3>
        <p class="text-xs text-text-secondary dark:text-text-tertiary">Observed accuracy per reported-confidence bin (points = bins, height = accuracy).</p>
        {#if calibrationCurves.length === 0}
          <p class="text-sm text-text-secondary dark:text-text-tertiary">No calibrated observations yet — confidence never calibrated if a bin is thin.</p>
        {:else}
          {#each calibrationCurves as curve}
            <div class="space-y-1">
              <div class="flex items-center justify-between gap-2">
                <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate">{curve.model}</p>
                <p class="text-[11px] text-text-tertiary">{curve.points.length} bins</p>
              </div>
              <svg viewBox="0 0 180 44" class="w-full h-11" preserveAspectRatio="none">
                <line x1="4" y1="40" x2="176" y2="40" stroke="currentColor" class="text-surface-300 dark:text-surface-600" stroke-width="1" />
                <polyline points={curvePoints(curve.points)} fill="none" stroke="#3b82f6" stroke-width="1.5" />
                {#each curve.points as p, i}
                  {@const last = i === curve.points.length - 1}
                  <circle cx={String(4 + (i / Math.max(1, curve.points.length - 1)) * 172)} cy={String(40 - Math.min(1, Math.max(0, p.accuracy || 0)) * 36)} r="2" fill={last ? '#22c55e' : '#3b82f6'}>
                    <title>{(p.bin || '')}: {Math.round((p.accuracy || 0) * 100)}% (n={p.n})</title>
                  </circle>
                {/each}
              </svg>
            </div>
          {/each}
        {/if}
      </div>

      <!-- MoE routing -->
      <div class="card p-4 space-y-3">
        <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">🧭 MoE Routing</h3>
        {#if moeEntries.length === 0}
          <p class="text-sm text-text-secondary dark:text-text-tertiary">No routing statistics yet.</p>
        {:else}
          {#each moeEntries as [route, count]}
            <div class="flex items-center justify-between gap-3">
              <p class="text-sm font-medium text-text-primary dark:text-text-primary truncate font-mono text-xs">{route}</p>
              <span class="text-xs text-text-secondary dark:text-text-tertiary">{count}</span>
            </div>
          {/each}
        {/if}
      </div>
    </div>

    <!-- L3: Domain win rates -->
    <div class="card p-4 space-y-3">
      <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">🎯 Domain Win Rates</h3>
      {#if domainWinRates.length === 0}
        <p class="text-sm text-text-secondary dark:text-text-tertiary">No domain-tagged feedback yet — send feedback with a domain to populate this.</p>
      {:else}
        {#each domainWinRates as d}
          <div class="space-y-1">
            <div class="flex items-center justify-between gap-2">
              <p class="text-sm font-medium text-text-primary dark:text-text-primary">{d.domain}</p>
              <p class="text-xs text-text-secondary dark:text-text-tertiary">
                {d.feedback_count} ratings · {Math.round(d.win_rate * 100)}% positive
              </p>
            </div>
            <div class="h-2 rounded bg-surface-100 dark:bg-surface-800 overflow-hidden">
              <div class="h-full rounded bg-gradient-to-r from-success-500 to-primary-500" style="width: {Math.max(0, d.win_rate * 100)}%"></div>
            </div>
            {#if (d.top_models || []).length > 0}
              <div class="flex flex-wrap gap-1.5 pt-1">
                {#each d.top_models as m}
                  <span class="text-[10px] px-2 py-0.5 rounded-full border border-surface-200 dark:border-surface-700 text-text-secondary dark:text-text-tertiary">
                    {m.model} · {Math.round(m.win_rate * 100)}% ({m.positive}/{m.positive + m.negative})
                  </span>
                {/each}
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

    <!-- Recent runs -->
    <div class="card p-4 space-y-2">
      <h3 class="text-sm font-bold text-text-primary dark:text-text-primary">🕐 Recent Runs</h3>
      {#if runs.length === 0}
        <p class="text-sm text-text-secondary dark:text-text-tertiary">No enhanced-pipeline runs yet.</p>
      {:else}
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-xs text-text-tertiary uppercase tracking-wide">
                <th class="py-2 pr-4">Run</th>
                <th class="py-2 pr-4">Query</th>
                <th class="py-2 pr-4">Status</th>
                <th class="py-2 pr-4">Duration</th>
                <th class="py-2">Events</th>
              </tr>
            </thead>
            <tbody>
              {#each runs as run}
                <tr class="border-t border-surface-100 dark:border-surface-800">
                  <td class="py-2 pr-4 font-mono text-xs text-text-secondary dark:text-text-tertiary">{run.run_id.slice(0, 8)}</td>
                  <td class="py-2 pr-4 text-text-primary dark:text-text-primary max-w-xs truncate">{run.query || '(no query)'}</td>
                  <td class="py-2 pr-4"><span class="text-xs text-primary-600 dark:text-primary-400">{run.status}</span></td>
                  <td class="py-2 pr-4 text-xs text-text-secondary dark:text-text-tertiary">{run.duration_seconds != null ? `${run.duration_seconds.toFixed(1)}s` : '—'}</td>
                  <td class="py-2 text-xs text-text-secondary dark:text-text-tertiary">{run.event_count}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>
  {/if}
</div>
