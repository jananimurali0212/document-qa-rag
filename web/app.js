/**
 * RAG STUDIO PRO • Single-Page Application Client Logic
 * Vanilla ES6 JavaScript for fast, reactive document Q&A and benchmarking.
 */

(function () {
  'use strict';

  // ===================================================================
  // GLOBAL STATE
  // ===================================================================
  const state = {
    topK: 5,
    enableReranker: true,
    chunkSize: 500,
    chunkOverlap: 100,
    documents: [],
    stagedFiles: [],
    isQuerying: false,
    isBenchmarking: false,
    latestBenchmark: null,
    latestSweep: null,
    chatMessages: []
  };

  // ===================================================================
  // DOM ELEMENT REFERENCES
  // ===================================================================
  const elements = {
    // Header telemetry
    llmPulse: document.getElementById('llmPulse'),
    llmStatusText: document.getElementById('llmStatusText'),
    statDocCount: document.getElementById('statDocCount'),
    statChunkCount: document.getElementById('statChunkCount'),
    statVectorBackend: document.getElementById('statVectorBackend'),
    tabBadgeDocs: document.getElementById('tabBadgeDocs'),
    btnSettingsToggle: document.getElementById('btnSettingsToggle'),

    // Navigation tabs
    navTabs: document.querySelectorAll('.nav-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // Settings drawer
    settingsDrawer: document.getElementById('settingsDrawer'),
    drawerBackdrop: document.getElementById('drawerBackdrop'),
    btnCloseDrawer: document.getElementById('btnCloseDrawer'),
    settingTopK: document.getElementById('settingTopK'),
    valTopK: document.getElementById('valTopK'),
    settingReranker: document.getElementById('settingReranker'),
    settingChunkSize: document.getElementById('settingChunkSize'),
    valChunkSize: document.getElementById('valChunkSize'),
    settingChunkOverlap: document.getElementById('settingChunkOverlap'),
    valChunkOverlap: document.getElementById('valChunkOverlap'),
    btnClearKB: document.getElementById('btnClearKB'),
    badgeTopK: document.getElementById('badgeTopK'),
    badgeReranker: document.getElementById('badgeReranker'),

    // Chat view
    chatStream: document.getElementById('chatStream'),
    chatWelcomeCard: document.getElementById('chatWelcomeCard'),
    chatForm: document.getElementById('chatForm'),
    queryInput: document.getElementById('queryInput'),
    btnSendQuery: document.getElementById('btnSendQuery'),
    quickChips: document.querySelectorAll('.quick-chip'),

    // Knowledge Base view
    uploadDropzone: document.getElementById('uploadDropzone'),
    fileInput: document.getElementById('fileInput'),
    fileQueue: document.getElementById('fileQueue'),
    queueCount: document.getElementById('queueCount'),
    queueList: document.getElementById('queueList'),
    btnClearQueue: document.getElementById('btnClearQueue'),
    btnProcessUpload: document.getElementById('btnProcessUpload'),
    uploadProgressBox: document.getElementById('uploadProgressBox'),
    uploadProgressBar: document.getElementById('uploadProgressBar'),
    uploadProgressText: document.getElementById('uploadProgressText'),
    uploadProgressPercent: document.getElementById('uploadProgressPercent'),
    docsTableBody: document.getElementById('docsTableBody'),
    tableDocCount: document.getElementById('tableDocCount'),
    docSearchInput: document.getElementById('docSearchInput'),

    // Benchmark view
    btnRunBenchmark: document.getElementById('btnRunBenchmark'),
    benchmarkLoader: document.getElementById('benchmarkLoader'),
    btnExportMenu: document.getElementById('btnExportMenu'),
    exportDropdown: document.getElementById('exportDropdown'),
    btnExportCSV: document.getElementById('btnExportCSV'),
    btnExportJSON: document.getElementById('btnExportJSON'),
    metricHitRate: document.getElementById('metricHitRate'),
    metricHitRateSub: document.getElementById('metricHitRateSub'),
    metricRecall: document.getElementById('metricRecall'),
    metricMRR: document.getElementById('metricMRR'),
    metricPrecision: document.getElementById('metricPrecision'),
    metricFaithfulness: document.getElementById('metricFaithfulness'),
    metricLatency: document.getElementById('metricLatency'),
    sweepChartSvg: document.getElementById('sweepChartSvg'),
    evalTableBody: document.getElementById('evalTableBody'),
    evalSearchInput: document.getElementById('evalSearchInput'),

    // Modals
    chunkModal: document.getElementById('chunkModal'),
    modalBackdrop: document.getElementById('modalBackdrop'),
    modalDocTitle: document.getElementById('modalDocTitle'),
    modalDocMeta: document.getElementById('modalDocMeta'),
    modalChunkList: document.getElementById('modalChunkList'),
    btnCloseModal: document.getElementById('btnCloseModal'),

    questionModal: document.getElementById('questionModal'),
    qModalBackdrop: document.getElementById('qModalBackdrop'),
    qModalTitle: document.getElementById('qModalTitle'),
    qModalId: document.getElementById('qModalId'),
    qModalBody: document.getElementById('qModalBody'),
    btnCloseQModal: document.getElementById('btnCloseQModal'),

    // Toast
    toastContainer: document.getElementById('toastContainer')
  };

  // ===================================================================
  // TOAST NOTIFICATIONS
  // ===================================================================
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(40px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatBytes(bytes, decimals = 1) {
    if (!+bytes) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  }

  // ===================================================================
  // SYSTEM STATUS & INITIAL DATA LOAD
  // ===================================================================
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status check failed');
      const data = await res.json();

      // LLM status badge
      if (data.llm_connected) {
        elements.llmPulse.className = 'pulse-indicator pulse-green';
        elements.llmStatusText.textContent = `${data.model_name} Online`;
      } else {
        elements.llmPulse.className = 'pulse-indicator pulse-amber';
        elements.llmStatusText.textContent = 'Local Extractive Grounding';
      }

      // Telemetry counts
      elements.statDocCount.textContent = data.total_documents;
      elements.statChunkCount.textContent = data.total_chunks;
      elements.statVectorBackend.textContent = data.vector_backend;
      elements.tabBadgeDocs.textContent = data.total_documents;
      elements.tableDocCount.textContent = data.total_documents;

    } catch (err) {
      elements.llmPulse.className = 'pulse-indicator pulse-amber';
      elements.llmStatusText.textContent = 'Server Offline';
      console.warn('Status error:', err);
    }
  }

  async function loadDocuments() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) throw new Error('Failed to load documents');
      const data = await res.json();
      state.documents = data.documents || [];
      renderDocumentsTable(state.documents);
    } catch (err) {
      console.error('Failed to load documents:', err);
      elements.docsTableBody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-muted">Error loading documents.</td></tr>`;
    }
  }

  // ===================================================================
  // NAVIGATION & TAB SWITCHING
  // ===================================================================
  function initNavigation() {
    elements.navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetId = tab.dataset.tab;
        
        elements.navTabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        elements.tabPanes.forEach(pane => pane.classList.remove('active'));

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');

        // Contextual refresh
        if (targetId === 'tabDocs') {
          loadDocuments();
        } else if (targetId === 'tabChat') {
          elements.queryInput.focus();
        }
      });
    });
  }

  // ===================================================================
  // PIPELINE SETTINGS DRAWER
  // ===================================================================
  function initSettingsDrawer() {
    elements.btnSettingsToggle.addEventListener('click', () => {
      elements.settingsDrawer.classList.add('open');
      elements.drawerBackdrop.classList.add('open');
    });

    const closeDrawer = () => {
      elements.settingsDrawer.classList.remove('open');
      elements.drawerBackdrop.classList.remove('open');
    };

    elements.btnCloseDrawer.addEventListener('click', closeDrawer);
    elements.drawerBackdrop.addEventListener('click', closeDrawer);

    // Top-K Slider
    elements.settingTopK.addEventListener('input', (e) => {
      state.topK = parseInt(e.target.value, 10);
      elements.valTopK.textContent = state.topK;
      elements.badgeTopK.textContent = `Top-K: ${state.topK}`;
    });

    // Reranker toggle
    elements.settingReranker.addEventListener('change', (e) => {
      state.enableReranker = e.target.checked;
      elements.badgeReranker.textContent = `Reranker: ${state.enableReranker ? 'ON' : 'OFF'}`;
    });

    // Chunk size & overlap
    elements.settingChunkSize.addEventListener('input', (e) => {
      state.chunkSize = parseInt(e.target.value, 10);
      elements.valChunkSize.textContent = state.chunkSize;
      elements.settingChunkOverlap.max = Math.min(250, state.chunkSize - 50);
    });

    elements.settingChunkOverlap.addEventListener('input', (e) => {
      state.chunkOverlap = parseInt(e.target.value, 10);
      elements.valChunkOverlap.textContent = state.chunkOverlap;
    });

    // Clear KB
    elements.btnClearKB.addEventListener('click', async () => {
      if (!confirm('⚠️ Are you sure you want to reset the entire knowledge base? All vector embeddings and chunks will be purged.')) {
        return;
      }
      try {
        const res = await fetch('/api/documents/clear', { method: 'POST' });
        if (!res.ok) throw new Error('Reset failed');
        showToast('Knowledge base completely cleared', 'success');
        closeDrawer();
        await fetchSystemStatus();
        await loadDocuments();
      } catch (err) {
        showToast(err.message, 'error');
      }
    });
  }

  // ===================================================================
  // TAB 1: CHAT & Q&A
  // ===================================================================
  function initChat() {
    // Quick prompt chips
    elements.quickChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.dataset.prompt;
        elements.queryInput.value = prompt;
        autoResizeTextarea(elements.queryInput);
        elements.chatForm.requestSubmit();
      });
    });

    // Auto-resize input
    elements.queryInput.addEventListener('input', function() {
      autoResizeTextarea(this);
    });

    // Enter key submission (Shift+Enter for newline)
    elements.queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.chatForm.requestSubmit();
      }
    });

    // Form submit
    elements.chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = elements.queryInput.value.trim();
      if (!question || state.isQuerying) return;

      // Hide welcome card if present
      if (elements.chatWelcomeCard) {
        elements.chatWelcomeCard.style.display = 'none';
      }

      // Add user message
      appendMessage('user', question);
      elements.queryInput.value = '';
      autoResizeTextarea(elements.queryInput);

      // Loading assistant bubble
      const loaderId = appendThinkingBubble();
      state.isQuerying = true;
      elements.btnSendQuery.disabled = true;

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            top_k: state.topK,
            enable_reranker: state.enableReranker
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Query execution failed');
        }

        const data = await res.json();
        removeThinkingBubble(loaderId);
        appendAssistantMessage(data);

      } catch (err) {
        removeThinkingBubble(loaderId);
        appendErrorMessage(err.message);
      } finally {
        state.isQuerying = false;
        elements.btnSendQuery.disabled = false;
        elements.queryInput.focus();
      }
    });
  }

  function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  function appendMessage(role, text) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : '⚡';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    content.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(content);

    elements.chatStream.appendChild(row);
    scrollChatToBottom();
  }

  function appendThinkingBubble() {
    const id = 'loader_' + Date.now();
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '⚡';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: var(--text-secondary);">
        <div class="loader-spinner" style="width: 14px; height: 14px;"></div>
        <span>Searching candidate passages & generating strict factual answer...</span>
      </div>
    `;

    content.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(content);

    elements.chatStream.appendChild(row);
    scrollChatToBottom();
    return id;
  }

  function removeThinkingBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '⚡';

    const content = document.createElement('div');
    content.className = 'message-content';

    // Bubble
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMarkdown(data.answer);

    // Citations Container
    if (data.citations && data.citations.length > 0) {
      const citationsBox = document.createElement('div');
      citationsBox.className = 'citations-box';

      const citationsHeader = document.createElement('div');
      citationsHeader.className = 'citations-header';
      citationsHeader.innerHTML = `
        <div class="citations-title">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
          <span>Audited Grounded Citations (${data.citations.length})</span>
        </div>
        <span style="font-size: 0.72rem; color: var(--text-muted);">Click to toggle</span>
      `;

      const citationsList = document.createElement('div');
      citationsList.className = 'citations-list';

      data.citations.forEach((c) => {
        const item = document.createElement('div');
        item.className = 'citation-item';
        item.innerHTML = `
          <div class="citation-top">
            <span class="citation-badge">${escapeHtml(c.document_name)} • Page ${c.page_number}</span>
            <span class="citation-score">Score: ${c.score}</span>
          </div>
          <div class="citation-text">"${escapeHtml(c.snippet)}"</div>
        `;
        citationsList.appendChild(item);
      });

      citationsHeader.addEventListener('click', () => {
        citationsList.style.display = citationsList.style.display === 'none' ? 'flex' : 'none';
      });

      citationsBox.appendChild(citationsHeader);
      citationsBox.appendChild(citationsList);
      bubble.appendChild(citationsBox);
    }

    // Metadata & copy action
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.innerHTML = `
      <span>⚡ ${data.model_name || 'Gemini 3.8 Flash'}</span>
      <span>⏱️ ${data.latency_ms} ms</span>
      <span>📑 ${data.retrieved_chunks_count} retrieved</span>
      <button class="btn-action-small" style="margin-left: auto;" title="Copy answer">
        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        Copy
      </button>
    `;

    const copyBtn = meta.querySelector('button');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(data.answer);
      copyBtn.innerHTML = `<span>Copied!</span>`;
      setTimeout(() => {
        copyBtn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`;
      }, 1800);
    });

    content.appendChild(bubble);
    content.appendChild(meta);
    row.appendChild(avatar);
    row.appendChild(content);

    elements.chatStream.appendChild(row);
    scrollChatToBottom();
  }

  function appendErrorMessage(errorMsg) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.style.background = 'var(--accent-rose)';
    avatar.textContent = '!';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.style.borderLeft = '3px solid var(--accent-rose)';
    bubble.innerHTML = `<strong>Error:</strong> ${escapeHtml(errorMsg)}`;

    content.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(content);

    elements.chatStream.appendChild(row);
    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Bullets
    html = html.replace(/^\s*[\-\*]\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul style="margin: 8px 0; padding-left: 20px;">$1</ul>');

    // Paragraph breaks
    html = html.replace(/\n\n+/g, '<br><br>');
    html = html.replace(/\n/g, '<br>');

    return html;
  }

  // ===================================================================
  // TAB 2: KNOWLEDGE BASE & UPLOADS
  // ===================================================================
  function initKnowledgeBase() {
    const dropzone = elements.uploadDropzone;
    const fileInput = elements.fileInput;

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files);
      }
    });

    elements.btnClearQueue.addEventListener('click', () => {
      state.stagedFiles = [];
      fileInput.value = '';
      updateQueueUI();
    });

    elements.btnProcessUpload.addEventListener('click', processUploads);

    // Document search
    elements.docSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        renderDocumentsTable(state.documents);
        return;
      }
      const filtered = state.documents.filter(d => 
        (d.document_name && d.document_name.toLowerCase().includes(q)) ||
        (d.document_id && d.document_id.toLowerCase().includes(q)) ||
        (d.file_type && d.file_type.toLowerCase().includes(q))
      );
      renderDocumentsTable(filtered);
    });

    // Chunk Inspector Modal close events
    elements.btnCloseModal.addEventListener('click', closeModal);
    elements.modalBackdrop.addEventListener('click', closeModal);

    // Question Modal close events
    elements.btnCloseQModal.addEventListener('click', closeQModal);
    elements.qModalBackdrop.addEventListener('click', closeQModal);
  }

  function handleFileSelection(fileList) {
    const validExtensions = ['.pdf', '.docx', '.txt'];
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (validExtensions.includes(ext)) {
        if (!state.stagedFiles.some(f => f.name === file.name && f.size === file.size)) {
          state.stagedFiles.push(file);
        }
      } else {
        showToast(`Skipped '${file.name}': Unsupported format`, 'error');
      }
    }
    updateQueueUI();
  }

  function updateQueueUI() {
    if (state.stagedFiles.length === 0) {
      elements.fileQueue.style.display = 'none';
      return;
    }

    elements.fileQueue.style.display = 'block';
    elements.queueCount.textContent = `${state.stagedFiles.length} file(s) ready to ingest`;
    elements.queueList.innerHTML = '';

    state.stagedFiles.forEach((f, idx) => {
      const item = document.createElement('div');
      item.className = 'queue-item';
      item.innerHTML = `
        <div class="queue-item-info">
          <span class="badge-format">${f.name.split('.').pop().toUpperCase()}</span>
          <span>${escapeHtml(f.name)}</span>
          <span class="text-muted" style="font-size: 0.74rem;">(${formatBytes(f.size)})</span>
        </div>
        <button class="btn-text" data-idx="${idx}">Remove</button>
      `;

      item.querySelector('button').addEventListener('click', (e) => {
        const removeIdx = parseInt(e.target.dataset.idx, 10);
        state.stagedFiles.splice(removeIdx, 1);
        updateQueueUI();
      });

      elements.queueList.appendChild(item);
    });
  }

  async function processUploads() {
    if (state.stagedFiles.length === 0) return;

    elements.btnProcessUpload.disabled = true;
    elements.uploadProgressBox.style.display = 'block';
    elements.uploadProgressBar.style.width = '15%';
    elements.uploadProgressText.textContent = 'Uploading files to server...';
    elements.uploadProgressPercent.textContent = '15%';

    const formData = new FormData();
    state.stagedFiles.forEach(f => formData.append('files', f));
    formData.append('chunk_size', state.chunkSize);
    formData.append('chunk_overlap', state.chunkOverlap);

    try {
      elements.uploadProgressBar.style.width = '45%';
      elements.uploadProgressText.textContent = 'Parsing text & generating chunks...';
      elements.uploadProgressPercent.textContent = '45%';

      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      });

      elements.uploadProgressBar.style.width = '80%';
      elements.uploadProgressText.textContent = 'Calculating embeddings & indexing...';
      elements.uploadProgressPercent.textContent = '80%';

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Ingestion request failed');
      }

      const result = await res.json();
      elements.uploadProgressBar.style.width = '100%';
      elements.uploadProgressPercent.textContent = '100%';
      elements.uploadProgressText.textContent = 'Ingestion complete!';

      showToast(result.message, 'success');
      state.stagedFiles = [];
      updateQueueUI();

      setTimeout(() => {
        elements.uploadProgressBox.style.display = 'none';
        elements.uploadProgressBar.style.width = '0%';
      }, 1500);

      await fetchSystemStatus();
      await loadDocuments();

    } catch (err) {
      showToast(err.message, 'error');
      elements.uploadProgressBox.style.display = 'none';
    } finally {
      elements.btnProcessUpload.disabled = false;
    }
  }

  function renderDocumentsTable(docs) {
    if (!docs || docs.length === 0) {
      elements.docsTableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center py-4 text-muted">
            No documents uploaded yet. Upload PDF, DOCX, or TXT files above to populate your knowledge base.
          </td>
        </tr>
      `;
      elements.tableDocCount.textContent = '0';
      return;
    }

    elements.tableDocCount.textContent = docs.length;
    elements.docsTableBody.innerHTML = '';

    docs.forEach(doc => {
      const tr = document.createElement('tr');
      const createdAt = doc.created_at ? doc.created_at.split('T')[0] : 'N/A';

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 0.78rem; color: #93C5FD;">${escapeHtml(doc.document_id)}</td>
        <td style="font-weight: 600;">${escapeHtml(doc.document_name)}</td>
        <td><span class="badge-format">${escapeHtml(doc.file_type.toUpperCase())}</span></td>
        <td>${doc.num_pages}</td>
        <td><strong>${doc.num_chunks}</strong></td>
        <td>${formatBytes(doc.file_size_bytes)}</td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${createdAt}</td>
        <td class="text-right">
          <button class="btn-action-small btn-inspect" data-id="${doc.document_id}" data-name="${escapeHtml(doc.document_name)}" data-chunks="${doc.num_chunks}" title="Inspect chunks">
            🔍 Chunks
          </button>
          <button class="btn-action-small btn-delete-small" data-id="${doc.document_id}" title="Delete document">
            🗑️
          </button>
        </td>
      `;

      // Inspect chunks button
      tr.querySelector('.btn-inspect').addEventListener('click', (e) => {
        const docId = e.currentTarget.dataset.id;
        const docName = e.currentTarget.dataset.name;
        const chunksCount = e.currentTarget.dataset.chunks;
        openChunkInspector(docId, docName, chunksCount);
      });

      // Delete document button
      tr.querySelector('.btn-delete-small').addEventListener('click', async (e) => {
        const docId = e.currentTarget.dataset.id;
        if (!confirm(`Delete document '${docId}' and all its vector embeddings?`)) return;

        try {
          const delRes = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
          if (!delRes.ok) throw new Error('Deletion failed');
          showToast(`Document ${docId} deleted`, 'success');
          await fetchSystemStatus();
          await loadDocuments();
        } catch (err) {
          showToast(err.message, 'error');
        }
      });

      elements.docsTableBody.appendChild(tr);
    });
  }

  async function openChunkInspector(docId, docName, chunksCount) {
    elements.modalDocTitle.textContent = docName;
    elements.modalDocMeta.textContent = `ID: ${docId} • Total Chunks: ${chunksCount}`;
    elements.modalChunkList.innerHTML = `<div class="text-center py-4 text-muted">Loading chunks...</div>`;

    elements.chunkModal.classList.add('open');
    elements.modalBackdrop.classList.add('open');

    try {
      const res = await fetch(`/api/documents/${docId}/chunks`);
      if (!res.ok) throw new Error('Failed to load chunks');
      const data = await res.json();
      const chunks = data.chunks || [];

      if (chunks.length === 0) {
        elements.modalChunkList.innerHTML = `<div class="text-center py-4 text-muted">No chunks found for this document.</div>`;
        return;
      }

      elements.modalChunkList.innerHTML = '';
      chunks.forEach((c) => {
        const card = document.createElement('div');
        card.className = 'chunk-card';
        card.innerHTML = `
          <div class="chunk-card-meta">
            <span><strong>Chunk ID:</strong> <code>${c.chunk_id}</code></span>
            <span>Page ${c.page_number} • Index ${c.chunk_index}</span>
          </div>
          <div class="chunk-card-text">${escapeHtml(c.text)}</div>
        `;
        elements.modalChunkList.appendChild(card);
      });

    } catch (err) {
      elements.modalChunkList.innerHTML = `<div class="text-center py-4 text-muted">Error: ${escapeHtml(err.message)}</div>`;
    }
  }

  function closeModal() {
    elements.chunkModal.classList.remove('open');
    elements.modalBackdrop.classList.remove('open');
  }

  function closeQModal() {
    elements.questionModal.classList.remove('open');
    elements.qModalBackdrop.classList.remove('open');
  }

  // ===================================================================
  // TAB 3: EVALUATION BENCHMARK DASHBOARD
  // ===================================================================
  function initBenchmark() {
    // Run benchmark button
    elements.btnRunBenchmark.addEventListener('click', runBenchmarkSuite);

    // Export dropdown toggle
    elements.btnExportMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.exportDropdown.classList.toggle('show');
    });

    document.addEventListener('click', () => {
      elements.exportDropdown.classList.remove('show');
    });

    elements.btnExportCSV.addEventListener('click', () => {
      window.location.href = '/api/benchmark/export/csv';
    });

    elements.btnExportJSON.addEventListener('click', () => {
      window.location.href = '/api/benchmark/export/json';
    });

    // Eval table search
    elements.evalSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!state.latestBenchmark || !state.latestBenchmark.per_question_results) return;

      const rows = state.latestBenchmark.per_question_results;
      if (!q) {
        renderPerQuestionTable(rows);
        return;
      }
      const filtered = rows.filter(r => 
        (r.Question && r.Question.toLowerCase().includes(q)) ||
        (r.Query && r.Query.toLowerCase().includes(q)) ||
        (r.Generated_Answer && r.Generated_Answer.toLowerCase().includes(q))
      );
      renderPerQuestionTable(filtered);
    });
  }

  async function runBenchmarkSuite() {
    if (state.isBenchmarking) return;

    state.isBenchmarking = true;
    elements.btnRunBenchmark.disabled = true;
    elements.benchmarkLoader.style.display = 'flex';

    try {
      const res = await fetch('/api/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          k: 5,
          dataset_path: 'evaluation_data/evaluation_dataset.json',
          include_sweep: true
        })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Benchmark evaluation failed');
      }

      const data = await res.json();
      state.latestBenchmark = {
        ...data.summary,
        per_question_results: data.per_question
      };
      state.latestSweep = data.k_sweep;

      renderBenchmarkSummary(data.summary);
      renderSweepChart(data.k_sweep);
      renderPerQuestionTable(data.per_question);

      showToast('Benchmark run completed successfully!', 'success');

    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      state.isBenchmarking = false;
      elements.btnRunBenchmark.disabled = false;
      elements.benchmarkLoader.style.display = 'none';
    }
  }

  function renderBenchmarkSummary(summary) {
    if (!summary) return;

    const hitPct = (summary.hit_rate_at_k * 100).toFixed(1);
    const recallPct = (summary.recall_at_k * 100).toFixed(1);
    const mrrVal = summary.mrr.toFixed(3);
    const precPct = (summary.precision_at_k * 100).toFixed(1);
    const faithPct = summary.mean_faithfulness !== 'N/A' && summary.mean_faithfulness != null
      ? (parseFloat(summary.mean_faithfulness) * 100).toFixed(1) + '%'
      : 'N/A';
    const latencyVal = Math.round(summary.avg_latency_ms);

    elements.metricHitRate.textContent = `${hitPct}%`;
    elements.metricRecall.textContent = `${recallPct}%`;
    elements.metricMRR.textContent = mrrVal;
    elements.metricPrecision.textContent = `${precPct}%`;
    elements.metricFaithfulness.textContent = faithPct;
    elements.metricLatency.textContent = `${latencyVal} ms`;

    // Dynamic color coding
    elements.metricHitRate.style.color = hitPct >= 85 ? 'var(--accent-emerald)' : (hitPct >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)');
    elements.metricRecall.style.color = recallPct >= 75 ? 'var(--accent-emerald)' : (recallPct >= 60 ? 'var(--accent-amber)' : 'var(--accent-rose)');
    elements.metricMRR.style.color = mrrVal >= 0.70 ? 'var(--accent-emerald)' : 'var(--accent-amber)';
  }

  function renderSweepChart(kSweep) {
    const svg = elements.sweepChartSvg;
    svg.innerHTML = '';

    if (!kSweep || Object.keys(kSweep).length === 0) {
      svg.innerHTML = `<text x="400" y="130" text-anchor="middle" fill="#64748B" font-size="14">No sweep data available.</text>`;
      return;
    }

    const kValues = Object.keys(kSweep).map(k => parseInt(k, 10)).sort((a, b) => a - b);
    const margin = { top: 30, right: 40, bottom: 40, left: 60 };
    const width = 800;
    const height = 260;
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    // Draw horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    for (let pct = 0; pct <= 100; pct += 25) {
      const y = margin.top + chartH - (pct / 100) * chartH;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left);
      line.setAttribute('y1', y);
      line.setAttribute('x2', width - margin.right);
      line.setAttribute('y2', y);
      line.setAttribute('stroke', 'rgba(255, 255, 255, 0.08)');
      line.setAttribute('stroke-dasharray', '3,3');
      svg.appendChild(line);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 12);
      label.setAttribute('y', y + 4);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('fill', '#64748B');
      label.setAttribute('font-size', '11');
      label.textContent = `${pct}%`;
      svg.appendChild(label);
    }

    // Group width calculations
    const groupW = chartW / kValues.length;
    const barW = 28;
    const barGap = 6;

    kValues.forEach((kVal, idx) => {
      const data = kSweep[kVal];
      const groupCenterX = margin.left + idx * groupW + groupW / 2;

      // Group X-axis label
      const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xLabel.setAttribute('x', groupCenterX);
      xLabel.setAttribute('y', height - 12);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('fill', '#CBD5E1');
      xLabel.setAttribute('font-size', '12');
      xLabel.setAttribute('font-weight', '700');
      xLabel.textContent = `K = ${kVal}`;
      svg.appendChild(xLabel);

      const metrics = [
        { key: 'hit_rate', val: data.hit_rate, color: '#3B82F6' },
        { key: 'recall', val: data.recall, color: '#10B981' },
        { key: 'mrr', val: data.mrr, color: '#8B5CF6' }
      ];

      const startX = groupCenterX - (metrics.length * barW + (metrics.length - 1) * barGap) / 2;

      metrics.forEach((m, mIdx) => {
        const barX = startX + mIdx * (barW + barGap);
        const barHeight = Math.max(2, m.val * chartH);
        const barY = margin.top + chartH - barHeight;

        // Bar rect
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', barX);
        rect.setAttribute('y', barY);
        rect.setAttribute('width', barW);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('fill', m.color);
        rect.setAttribute('rx', 4);
        rect.setAttribute('opacity', '0.9');
        svg.appendChild(rect);

        // Value text above bar
        const valText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        valText.setAttribute('x', barX + barW / 2);
        valText.setAttribute('y', barY - 6);
        valText.setAttribute('text-anchor', 'middle');
        valText.setAttribute('fill', '#FFFFFF');
        valText.setAttribute('font-size', '10');
        valText.setAttribute('font-weight', '600');
        valText.textContent = (m.val * 100).toFixed(0) + '%';
        svg.appendChild(valText);
      });
    });
  }

  function renderPerQuestionTable(rows) {
    if (!rows || rows.length === 0) {
      elements.evalTableBody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-muted">No questions found matching criteria.</td></tr>`;
      return;
    }

    elements.evalTableBody.innerHTML = '';
    rows.forEach(r => {
      const tr = document.createElement('tr');
      const hitBadge = r['Hit@K'] === 1
        ? `<span style="color: var(--accent-emerald); font-weight: 700;">✓ 1</span>`
        : `<span style="color: var(--accent-rose); font-weight: 700;">✗ 0</span>`;

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 0.76rem; color: #93C5FD;">${escapeHtml(r.Question)}</td>
        <td style="max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(r.Query)}">${escapeHtml(r.Query)}</td>
        <td>${hitBadge}</td>
        <td>${(r['Recall@K'] * 100).toFixed(0)}%</td>
        <td>${r.MRR}</td>
        <td>${(r['Precision@K'] * 100).toFixed(0)}%</td>
        <td>${r.Faithfulness}</td>
        <td style="font-size: 0.78rem;">${r.Latency_ms} ms</td>
        <td class="text-right">
          <button class="btn-action-small btn-view-q" title="Inspect answer & citations">
            View
          </button>
        </td>
      `;

      tr.querySelector('.btn-view-q').addEventListener('click', () => {
        openQuestionModal(r);
      });

      elements.evalTableBody.appendChild(tr);
    });
  }

  function openQuestionModal(r) {
    elements.qModalTitle.textContent = r.Query;
    elements.qModalId.textContent = `Question ID: ${r.Question}`;

    elements.qModalBody.innerHTML = `
      <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
        <span class="badge-accent">Hit@K: ${r['Hit@K']}</span>
        <span class="badge-accent">Recall@K: ${(r['Recall@K'] * 100).toFixed(1)}%</span>
        <span class="badge-accent">MRR: ${r.MRR}</span>
        <span class="badge-accent">Precision@K: ${(r['Precision@K'] * 100).toFixed(1)}%</span>
        <span class="badge-accent">Faithfulness: ${r.Faithfulness}</span>
        <span class="badge-accent">Latency: ${r.Latency_ms} ms</span>
      </div>

      <div style="margin-bottom: 16px;">
        <h4 style="font-size: 0.88rem; font-weight: 700; color: #60A5FA; margin-bottom: 6px;">Generated Answer:</h4>
        <div class="message-bubble" style="background: rgba(0, 0, 0, 0.3); border: 1px solid var(--border-subtle); padding: 12px 14px; border-radius: var(--radius-md);">
          ${formatMarkdown(r.Generated_Answer)}
        </div>
      </div>

      <div style="margin-bottom: 16px;">
        <h4 style="font-size: 0.88rem; font-weight: 700; color: #34D399; margin-bottom: 6px;">Ground Truth Reference Answer:</h4>
        <div style="background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border-subtle); padding: 10px 14px; border-radius: var(--radius-md); font-size: 0.85rem; color: #CBD5E1;">
          ${escapeHtml(r.Reference)}
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
        <div>
          <h5 style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px;">Ground Truth Chunks:</h5>
          <code style="font-size: 0.76rem; color: #60A5FA; display: block; background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: var(--radius-sm);">
            ${Array.isArray(r.Ground_Truth_IDs) ? r.Ground_Truth_IDs.join(', ') : r.Ground_Truth_IDs}
          </code>
        </div>
        <div>
          <h5 style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px;">Retrieved Candidates:</h5>
          <code style="font-size: 0.76rem; color: #34D399; display: block; background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: var(--radius-sm);">
            ${Array.isArray(r.Retrieved_IDs) ? r.Retrieved_IDs.join(', ') : r.Retrieved_IDs}
          </code>
        </div>
      </div>
    `;

    elements.questionModal.classList.add('open');
    elements.qModalBackdrop.classList.add('open');
  }

  // ===================================================================
  // BOOTSTRAP APPLICATION
  // ===================================================================
  async function init() {
    initNavigation();
    initSettingsDrawer();
    initChat();
    initKnowledgeBase();
    initBenchmark();

    // Initial data fetch
    await fetchSystemStatus();
    await loadDocuments();

    // Periodic telemetry ping (every 15s)
    setInterval(fetchSystemStatus, 15000);
  }

  // Run on DOM load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
