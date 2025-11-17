<script>
  import { onMount, onDestroy } from 'svelte'
  
  export let onTranscription = (text) => {}
  export let disabled = false
  
  let isRecording = false
  let mediaRecorder = null
  let audioChunks = []
  let errorMessage = ''
  let isProcessing = false
  
  async function startRecording() {
    try {
      errorMessage = ''
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      mediaRecorder = new MediaRecorder(stream)
      audioChunks = []
      
      mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data)
      })
      
      mediaRecorder.addEventListener('stop', async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
        await transcribeAudio(audioBlob)
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      })
      
      mediaRecorder.start()
      isRecording = true
    } catch (error) {
      console.error('Error accessing microphone:', error)
      errorMessage = 'Could not access microphone. Please check permissions.'
    }
  }
  
  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop()
      isRecording = false
    }
  }
  
  async function transcribeAudio(audioBlob) {
    try {
      isProcessing = true
      errorMessage = ''
      
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.wav')
      
      const response = await fetch('/api/audio/transcribe', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Transcription failed')
      }
      
      const result = await response.json()
      
      if (result.success && result.text) {
        onTranscription(result.text)
      } else {
        errorMessage = 'No transcription received'
      }
      
      isProcessing = false
    } catch (error) {
      console.error('Transcription error:', error)
      errorMessage = 'Failed to transcribe audio'
      isProcessing = false
    }
  }
  
  function toggleRecording() {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }
  
  onDestroy(() => {
    if (isRecording) {
      stopRecording()
    }
  })
</script>

<div class="voice-input">
  <button
    on:click={toggleRecording}
    disabled={disabled || isProcessing}
    class="voice-button"
    class:recording={isRecording}
    class:processing={isProcessing}
    title={isRecording ? 'Stop recording' : 'Start voice input'}
  >
    {#if isProcessing}
      <svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
      </svg>
    {:else if isRecording}
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
        <rect x="6" y="6" width="8" height="8" rx="1"/>
      </svg>
    {:else}
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
        <path d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4z"/>
        <path d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z"/>
      </svg>
    {/if}
  </button>
  
  {#if isRecording}
    <div class="recording-indicator">
      <span class="pulse"></span>
      Recording...
    </div>
  {/if}
  
  {#if isProcessing}
    <div class="text-xs text-gray-500 dark:text-gray-400 ml-2">
      Transcribing...
    </div>
  {/if}
  
  {#if errorMessage}
    <div class="text-xs text-red-500 mt-1">
      {errorMessage}
    </div>
  {/if}
</div>

<style>
  .voice-input {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .voice-button {
    padding: 0.5rem;
    border-radius: 0.5rem;
    border: 2px solid transparent;
    background: var(--color-bg-secondary);
    color: var(--color-text);
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .voice-button:hover:not(:disabled) {
    background: var(--color-bg-hover);
    border-color: var(--color-primary);
  }
  
  .voice-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .voice-button.recording {
    background: #ef4444;
    color: white;
    animation: pulse 1.5s ease-in-out infinite;
  }
  
  .voice-button.processing {
    background: var(--color-primary);
    color: white;
  }
  
  .recording-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #ef4444;
    font-weight: 500;
  }
  
  .pulse {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse 1.5s ease-in-out infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.1);
    }
  }
</style>

