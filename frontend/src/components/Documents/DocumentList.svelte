<script>
  import { onMount } from 'svelte'
  
  export let onDocumentDeleted = () => {}
  export let refreshTrigger = 0
  
  let documents = []
  let isLoading = false
  let errorMessage = ''
  
  $: if (refreshTrigger) {
    loadDocuments()
  }
  
  onMount(() => {
    loadDocuments()
  })
  
  async function loadDocuments() {
    try {
      isLoading = true
      errorMessage = ''
      
      const response = await fetch('/api/documents/list')
      
      if (!response.ok) {
        throw new Error('Failed to load documents')
      }
      
      const result = await response.json()
      documents = result.documents || []
      
      isLoading = false
    } catch (error) {
      console.error('Load documents error:', error)
      errorMessage = error.message
      isLoading = false
    }
  }
  
  async function deleteDocument(documentId) {
    if (!confirm('Are you sure you want to delete this document?')) {
      return
    }
    
    try {
      const response = await fetch(`/api/documents/${documentId}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) {
        throw new Error('Failed to delete document')
      }
      
      // Refresh list
      await loadDocuments()
      onDocumentDeleted()
      
    } catch (error) {
      console.error('Delete error:', error)
      alert('Failed to delete document')
    }
  }
  
  function formatDate(dateString) {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
  }
</script>

<div class="document-list">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
      Uploaded Documents ({documents.length})
    </h3>
    <button
      on:click={loadDocuments}
      disabled={isLoading}
      class="btn-secondary"
      title="Refresh list"
    >
      <svg class="w-4 h-4" class:animate-spin={isLoading} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
    </button>
  </div>
  
  {#if isLoading && documents.length === 0}
    <div class="loading-state">
      <svg class="animate-spin w-6 h-6 text-primary-600" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
      </svg>
      <p class="text-sm text-gray-500 mt-2">Loading documents...</p>
    </div>
  {:else if errorMessage}
    <div class="error-state">
      <p class="text-red-500">{errorMessage}</p>
    </div>
  {:else if documents.length === 0}
    <div class="empty-state">
      <svg class="w-16 h-16 text-gray-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <p class="text-gray-500">No documents uploaded yet</p>
    </div>
  {:else}
    <div class="documents-grid">
      {#each documents as doc}
        <div class="document-card">
          <div class="document-header">
            <svg class="w-5 h-5 text-primary-600" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"/>
            </svg>
            <h4 class="font-medium text-gray-900 dark:text-gray-100 truncate flex-1">
              {doc.metadata?.filename || doc.document_id}
            </h4>
            <button
              on:click={() => deleteDocument(doc.document_id)}
              class="delete-button"
              title="Delete document"
            >
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
              </svg>
            </button>
          </div>
          
          <div class="document-meta">
            <span class="meta-item">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
              </svg>
              {doc.chunk_count} chunks
            </span>
            
            {#if doc.created_at}
              <span class="meta-item text-xs">
                {formatDate(doc.created_at)}
              </span>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .document-list {
    width: 100%;
  }
  
  .loading-state,
  .empty-state,
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
  }
  
  .documents-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  }
  
  .document-card {
    padding: 1rem;
    border: 1px solid var(--color-border);
    border-radius: 0.75rem;
    background: var(--color-bg);
    transition: all 0.2s;
  }
  
  .document-card:hover {
    border-color: var(--color-primary);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }
  
  .document-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }
  
  .delete-button {
    padding: 0.25rem;
    border-radius: 0.25rem;
    color: #6b7280;
    transition: all 0.2s;
  }
  
  .delete-button:hover {
    color: #ef4444;
    background: #fee2e2;
  }
  
  .document-meta {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--color-text-secondary);
  }
  
  .meta-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }
  
  .btn-secondary {
    padding: 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid var(--color-border);
    background: var(--color-bg);
    color: var(--color-text);
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .btn-secondary:hover:not(:disabled) {
    background: var(--color-bg-hover);
    border-color: var(--color-primary);
  }
  
  .btn-secondary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>

