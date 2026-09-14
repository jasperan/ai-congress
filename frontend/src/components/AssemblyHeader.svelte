<script>
  export let models = []
  export let selectedModels = []
</script>

<section class="assembly-masthead" aria-labelledby="assembly-title">
  <div><p class="assembly-eyebrow">THE DELIBERATION WORKSPACE</p><h2 id="assembly-title">More perspectives.<br /><em>A considered answer.</em></h2><p>Convene a council of models. Compare their reasoning,<br class="assembly-break" /> weigh the evidence, and keep the dissent in view.</p></div>
  <aside class="assembly-seats" aria-label="Current model selection">
    <div class="assembly-seat-heading"><span>THE CHAMBER</span><span>OLLAMA / LOCAL</span></div>
    <div class="assembly-hemicycle" aria-hidden="true">{#each Array(27) as _, i}<span style={`--angle:${-180 + i * (180 / 26)}deg`} class:seated={selectedModels.length > 0 && i % 3 !== 0}></span>{/each}<b>{selectedModels.length}<small>selected</small></b></div>
    <div class="assembly-models">{#each selectedModels as model}<span>{model}</span>{:else}<span>Select models below to begin</span>{/each}</div>
    <p>{models.length} available models · Your selection defines the council</p>
  </aside>
</section>
