<script>
  export let imageUrl = ''
  export let prompt = ''
  export let metadata = {}
  export let onClose = () => {}
  export let onDownload = () => {}
  
  let isLoading = true
  
  function handleImageLoad() {
    isLoading = false
  }
  
  function handleImageError() {
    isLoading = false
  }
  
  function downloadImage() {
    const link = document.createElement('a')
    link.href = imageUrl
    link.download = imageUrl.split('/').pop() || 'generated_image.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    onDownload()
  }
</script>

<div class="image-display">
  <div class="image-container">
    {#if isLoading}
      <div class="loading-skeleton">
        <svg class="animate-spin w-8 h-8 text-primary-600" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
        </svg>
        <p class="text-sm text-gray-500 mt-2">Loading image...</p>
      </div>
    {/if}
    
    <img
      src={imageUrl}
      alt={prompt || 'Generated image'}
      on:load={handleImageLoad}
      on:error={handleImageError}
      class:hidden={isLoading}
      class="generated-image"
    />
    
    <div class="image-actions">
      <button
        on:click={downloadImage}
        class="action-button"
        title="Download image"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>
      </button>
    </div>
  </div>
  
  {#if prompt}
    <div class="image-info">
      <div class="info-section">
        <h4 class="info-label">Prompt:</h4>
        <p class="info-value">{prompt}</p>
      </div>
      
      {#if metadata}
        <div class="info-grid">
          {#if metadata.steps}
            <div class="info-item">
              <span class="info-label">Steps:</span>
              <span class="info-value">{metadata.steps}</span>
            </div>
          {/if}
          
          {#if metadata.width && metadata.height}
            <div class="info-item">
              <span class="info-label">Size:</span>
              <span class="info-value">{metadata.width}×{metadata.height}</span>
            </div>
          {/if}
          
          {#if metadata.seed !== null && metadata.seed !== undefined}
            <div class="info-item">
              <span class="info-label">Seed:</span>
              <span class="info-value">{metadata.seed}</span>
            </div>
          {/if}
          
          {#if metadata.negative_prompt}
            <div class="info-section">
              <h4 class="info-label">Negative Prompt:</h4>
              <p class="info-value text-sm">{metadata.negative_prompt}</p>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .image-display {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .image-container {
    position: relative;
    border-radius: 0.75rem;
    overflow: hidden;
    background: var(--color-bg-secondary);
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .loading-skeleton {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
  }
  
  .generated-image {
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;
    max-height: 600px;
  }
  
  .generated-image.hidden {
    display: none;
  }
  
  .image-actions {
    position: absolute;
    top: 1rem;
    right: 1rem;
    display: flex;
    gap: 0.5rem;
    opacity: 0;
    transition: opacity 0.2s;
  }
  
  .image-container:hover .image-actions {
    opacity: 1;
  }
  
  .action-button {
    padding: 0.5rem;
    border-radius: 0.5rem;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    backdrop-filter: blur(8px);
  }
  
  .action-button:hover {
    background: rgba(0, 0, 0, 0.9);
    transform: scale(1.05);
  }
  
  .image-info {
    padding: 1rem;
    background: var(--color-bg);
    border-radius: 0.75rem;
    border: 1px solid var(--color-border);
  }
  
  .info-section {
    margin-bottom: 1rem;
  }
  
  .info-section:last-child {
    margin-bottom: 0;
  }
  
  .info-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--color-text-secondary);
    margin-bottom: 0.25rem;
  }
  
  .info-value {
    font-size: 0.875rem;
    color: var(--color-text);
  }
  
  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  
  .info-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
</style>

