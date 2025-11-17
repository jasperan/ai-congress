<script>
  export let onUploadComplete = (result) => {}
  
  let isDragging = false
  let isUploading = false
  let uploadProgress = 0
  let errorMessage = ''
  let fileInput
  
  function handleDragOver(event) {
    event.preventDefault()
    isDragging = true
  }
  
  function handleDragLeave() {
    isDragging = false
  }
  
  function handleDrop(event) {
    event.preventDefault()
    isDragging = false
    
    const files = event.dataTransfer.files
    if (files.length > 0) {
      uploadFile(files[0])
    }
  }
  
  function handleFileSelect(event) {
    const files = event.target.files
    if (files.length > 0) {
      uploadFile(files[0])
    }
  }
  
  async function uploadFile(file) {
    try {
      isUploading = true
      errorMessage = ''
      uploadProgress = 0
      
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      })
      
      uploadProgress = 100
      
      if (!response.ok) {
        throw new Error('Upload failed')
      }
      
      const result = await response.json()
      
      if (result.success) {
        onUploadComplete(result)
      } else {
        errorMessage = result.error || 'Upload failed'
      }
      
      isUploading = false
      
      // Reset file input
      if (fileInput) {
        fileInput.value = ''
      }
      
    } catch (error) {
      console.error('Upload error:', error)
      errorMessage = error.message || 'Failed to upload document'
      isUploading = false
    }
  }
  
  function triggerFileSelect() {
    fileInput?.click()
  }
</script>

<div class="document-upload">
  <div
    class="upload-area"
    class:dragging={isDragging}
    class:uploading={isUploading}
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
    on:click={triggerFileSelect}
    role="button"
    tabindex="0"
  >
    <input
      bind:this={fileInput}
      type="file"
      on:change={handleFileSelect}
      accept=".pdf,.docx,.txt,.pptx,.csv,.json,.md,.xlsx"
      style="display: none"
    />
    
    {#if isUploading}
      <div class="upload-status">
        <svg class="animate-spin w-8 h-8 text-primary-600" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
        </svg>
        <p class="text-sm mt-2">Processing document...</p>
        {#if uploadProgress > 0}
          <div class="progress-bar">
            <div class="progress-fill" style="width: {uploadProgress}%"></div>
          </div>
        {/if}
      </div>
    {:else}
      <svg class="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
      </svg>
      
      <p class="text-sm font-medium text-gray-700 dark:text-gray-300">
        Drop a document here or click to browse
      </p>
      <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
        PDF, DOCX, TXT, PPTX, CSV, JSON, MD, XLSX
      </p>
    {/if}
  </div>
  
  {#if errorMessage}
    <div class="error-message">
      <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
      </svg>
      {errorMessage}
    </div>
  {/if}
</div>

<style>
  .document-upload {
    width: 100%;
  }
  
  .upload-area {
    border: 2px dashed #d1d5db;
    border-radius: 0.75rem;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    background: var(--color-bg);
  }
  
  .upload-area:hover:not(.uploading) {
    border-color: var(--color-primary);
    background: var(--color-bg-hover);
  }
  
  .upload-area.dragging {
    border-color: var(--color-primary);
    background: var(--color-primary-light);
    transform: scale(1.02);
  }
  
  .upload-area.uploading {
    cursor: not-allowed;
  }
  
  .upload-status {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  
  .progress-bar {
    width: 100%;
    height: 0.5rem;
    background: #e5e7eb;
    border-radius: 0.25rem;
    overflow: hidden;
    margin-top: 1rem;
  }
  
  .progress-fill {
    height: 100%;
    background: var(--color-primary);
    transition: width 0.3s;
  }
  
  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    padding: 0.5rem;
    background: #fee2e2;
    color: #991b1b;
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }
</style>

