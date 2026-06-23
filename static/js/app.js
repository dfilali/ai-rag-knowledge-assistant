// AG Knowledge Assistant - Frontend Logic
document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const providerSelect = document.getElementById("provider-select");
    const openaiKeyGroup = document.getElementById("openai-key-group");
    const mistralKeyGroup = document.getElementById("mistral-key-group");
    const openaiKeyInput = document.getElementById("openai-key-input");
    const mistralKeyInput = document.getElementById("mistral-key-input");
    const openaiStatus = document.getElementById("openai-status");
    const mistralStatus = document.getElementById("mistral-status");
    
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgressContainer = document.getElementById("upload-progress-container");
    const uploadProgressFill = document.getElementById("upload-progress-fill");
    const uploadProgressText = document.getElementById("upload-progress-text");
    
    const docCount = document.getElementById("doc-count");
    const docList = document.getElementById("doc-list");
    
    const activeModelBadge = document.getElementById("active-model-badge");
    const clearBtn = document.getElementById("clear-btn");
    const messagesContainer = document.getElementById("messages-container");
    const emptyState = document.getElementById("empty-state");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    // --- State Variables ---
    let activeProvider = localStorage.getItem("ag_provider") || "openai";
    let sessionId = localStorage.getItem("ag_session_id") || generateUUID();
    localStorage.setItem("ag_session_id", sessionId);

    // --- Init ---
    initSettings();
    loadDocuments();
    checkBackendStatus();

    // --- Functions ---
    
    // Generate simple UUID
    function generateUUID() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // Initialize UI settings from localStorage
    function initSettings() {
        providerSelect.value = activeProvider;
        openaiKeyInput.value = localStorage.getItem("ag_openai_key") || "";
        mistralKeyInput.value = localStorage.getItem("ag_mistral_key") || "";
        
        toggleKeyInputs();
        updateSendButtonState();
        updateActiveBadge();
    }

    // Hide/Show key input fields depending on active provider
    function toggleKeyInputs() {
        if (activeProvider === "openai") {
            openaiKeyGroup.classList.remove("hidden");
            mistralKeyGroup.classList.add("hidden");
        } else {
            openaiKeyGroup.classList.add("hidden");
            mistralKeyGroup.classList.remove("hidden");
        }
    }

    function updateActiveBadge() {
        if (activeProvider === "openai") {
            activeModelBadge.textContent = "OpenAI GPT-4o-mini RAG";
            activeModelBadge.style.color = "#6366f1";
            activeModelBadge.style.borderColor = "rgba(99, 102, 241, 0.3)";
            activeModelBadge.style.backgroundColor = "rgba(99, 102, 241, 0.1)";
        } else {
            activeModelBadge.textContent = "Mistral Large RAG";
            activeModelBadge.style.color = "#a855f7";
            activeModelBadge.style.borderColor = "rgba(168, 85, 247, 0.3)";
            activeModelBadge.style.backgroundColor = "rgba(168, 85, 247, 0.1)";
        }
    }

    // Get active keys and configurations
    function getCredentials() {
        const openaiKey = openaiKeyInput.value.trim();
        const mistralKey = mistralKeyInput.value.trim();
        
        return {
            provider: activeProvider,
            openaiKey,
            mistralKey,
            headers: {
                "X-Provider": activeProvider,
                "X-OpenAI-Key": openaiKey,
                "X-Mistral-Key": mistralKey
            }
        };
    }

    // Check if the current provider is ready to send queries
    function isConfiguredState() {
        const creds = getCredentials();
        if (creds.provider === "openai") {
            return creds.openaiKey.length > 0 || openaiStatus.classList.contains("configured");
        } else if (creds.provider === "mistral") {
            return creds.mistralKey.length > 0 || mistralStatus.classList.contains("configured");
        }
        return false;
    }

    function updateSendButtonState() {
        const hasText = chatInput.value.trim().length > 0;
        sendBtn.disabled = !(hasText && isConfiguredState());
    }

    // Fetch API configurations on backend to see if env keys are present
    async function checkBackendStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const status = await res.json();
                if (status.env_openai_configured) {
                    openaiStatus.classList.add("configured");
                    openaiStatus.title = "Clé configurée dans l'environnement backend (.env)";
                } else {
                    openaiStatus.classList.remove("configured");
                }
                if (status.env_mistral_configured) {
                    mistralStatus.classList.add("configured");
                    mistralStatus.title = "Clé configurée dans l'environnement backend (.env)";
                } else {
                    mistralStatus.classList.remove("configured");
                }
            }
        } catch (e) {
            console.error("Failed to fetch API status", e);
        }
        updateSendButtonState();
    }

    // Fetch list of documents
    async function loadDocuments() {
        try {
            const res = await fetch("/api/documents");
            if (!res.ok) throw new Error("Erreur de chargement");
            
            const docs = await res.json();
            docCount.textContent = docs.length;
            
            if (docs.length === 0) {
                docList.innerHTML = '<div class="doc-list-empty">Aucun document chargé dans la base de connaissances.</div>';
                return;
            }
            
            docList.innerHTML = "";
            docs.forEach(doc => {
                const sizeKb = (doc.size_bytes / 1024).toFixed(1);
                
                const docItem = document.createElement("div");
                docItem.className = "doc-item";
                docItem.innerHTML = `
                    <div class="doc-details">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        <div class="doc-info">
                            <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                            <span class="doc-size">${sizeKb} KB</span>
                        </div>
                    </div>
                    <button class="btn-delete" title="Supprimer ce document">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                `;
                
                docItem.querySelector(".btn-delete").addEventListener("click", () => {
                    deleteDocument(doc.filename);
                });
                
                docList.appendChild(docItem);
            });
        } catch (e) {
            console.error("Failed to load documents", e);
            docList.innerHTML = '<div class="doc-list-empty text-danger">Erreur de chargement des documents.</div>';
        }
    }

    // Delete document from vector base
    async function deleteDocument(filename) {
        if (!confirm(`Voulez-vous vraiment supprimer et désindexer "${filename}" ?`)) return;
        
        const creds = getCredentials();
        
        // Disable delete buttons while working
        document.querySelectorAll(".btn-delete").forEach(btn => btn.disabled = true);
        
        // Show status loading overlay in sidebar
        docList.innerHTML = '<div class="doc-list-empty">Mise à jour de l\'index vectoriel...</div>';
        
        try {
            const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE",
                headers: creds.headers
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Échec de la suppression");
            }
            
            alert(`"${filename}" supprimé avec succès.`);
        } catch (e) {
            alert(`Erreur: ${e.message}`);
        } finally {
            loadDocuments();
        }
    }

    // Upload PDF file
    async function uploadFile(file) {
        if (!file || !file.name.toLowerCase().endswith(".pdf")) {
            alert("Veuillez sélectionner un fichier PDF valide.");
            return;
        }

        const creds = getCredentials();
        if (!isConfiguredState()) {
            alert("Veuillez configurer votre clé API avant de charger des documents.");
            return;
        }

        // Show progress UI
        uploadProgressContainer.classList.remove("hidden");
        uploadProgressFill.style.width = "40%";
        uploadProgressText.textContent = "Lecture et découpage du PDF...";
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                headers: {
                    "X-Provider": creds.headers["X-Provider"],
                    "X-OpenAI-Key": creds.headers["X-OpenAI-Key"],
                    "X-Mistral-Key": creds.headers["X-Mistral-Key"]
                },
                body: formData
            });

            uploadProgressFill.style.width = "80%";
            uploadProgressText.textContent = "Calcul des embeddings & indexation...";

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Échec de l'indexation");
            }

            const data = await res.json();
            uploadProgressFill.style.width = "100%";
            uploadProgressText.textContent = "Indexation terminée !";
            
            setTimeout(() => {
                uploadProgressContainer.classList.add("hidden");
            }, 2000);
            
            loadDocuments();
        } catch (e) {
            alert(`Erreur d'importation : ${e.message}`);
            uploadProgressContainer.classList.add("hidden");
        }
    }

    // Lightweight HTML Markdown Parser
    function parseMarkdown(text) {
        // Escape HTML
        let html = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
            
        // Code blocks: ```code```
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold: **text**
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Bullet lists
        html = html.replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
        
        // Clean double list wraps
        html = html.replace(/<\/ul>\s*<ul>/g, '');
        
        // Line breaks
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }

    // Render Chat Message Bubble
    function appendMessage(role, content, sources = null) {
        emptyState.classList.add("hidden");
        
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}`;
        
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        
        if (role === "user") {
            bubble.textContent = content;
            messageDiv.appendChild(bubble);
        } else {
            bubble.innerHTML = parseMarkdown(content);
            messageDiv.appendChild(bubble);
            
            // Add RAG Sources if present
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement("div");
                sourcesDiv.className = "message-sources";
                
                const toggle = document.createElement("div");
                toggle.className = "sources-toggle";
                toggle.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                    Afficher les sources (${sources.length})
                `;
                
                const list = document.createElement("div");
                list.className = "sources-list hidden";
                
                sources.forEach(src => {
                    const card = document.createElement("div");
                    card.className = "source-card";
                    card.innerHTML = `
                        <div class="source-meta">
                            <span>📄 ${src.source}</span>
                            <span class="source-meta-tag">Page ${src.page}</span>
                        </div>
                        <div class="source-text">"${src.content.substring(0, 180)}..."</div>
                    `;
                    list.appendChild(card);
                });
                
                toggle.addEventListener("click", () => {
                    const isHidden = list.classList.toggle("hidden");
                    toggle.classList.toggle("active", !isHidden);
                });
                
                sourcesDiv.appendChild(toggle);
                sourcesDiv.appendChild(list);
                messageDiv.appendChild(sourcesDiv);
            }
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageDiv;
    }

    // Append Typing Indicator
    function showTypingIndicator() {
        emptyState.classList.add("hidden");
        
        const indicatorDiv = document.createElement("div");
        indicatorDiv.className = "message assistant loading-message";
        indicatorDiv.innerHTML = `
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        
        messagesContainer.appendChild(indicatorDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return indicatorDiv;
    }

    // Call RAG Evaluation and Render Widget
    async function evaluateAndRenderWidget(messageDiv, query, answer, contexts) {
        const creds = getCredentials();
        
        // Create initial skeleton eval widget
        const evalWidget = document.createElement("div");
        evalWidget.className = "eval-widget";
        evalWidget.innerHTML = `
            <div class="eval-header">
                <div class="eval-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                    <span>Auto-Évaluation RAG</span>
                </div>
                <div class="eval-scores-preview">
                    <span class="eval-badge">Calcul en cours...</span>
                </div>
            </div>
        `;
        messageDiv.appendChild(evalWidget);
        
        try {
            const res = await fetch("/api/evaluate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Provider": creds.headers["X-Provider"],
                    "X-OpenAI-Key": creds.headers["X-OpenAI-Key"],
                    "X-Mistral-Key": creds.headers["X-Mistral-Key"]
                },
                body: JSON.stringify({
                    query,
                    answer,
                    contexts,
                    provider: creds.provider
                })
            });
            
            if (!res.ok) throw new Error("Evaluation failed");
            
            const evalData = await res.json();
            
            // Build star rating helper
            const getStars = (score) => {
                let starsStr = "";
                for (let i = 1; i <= 5; i++) {
                    starsStr += `<span class="star ${i <= score ? 'filled' : ''}">★</span>`;
                }
                return starsStr;
            };
            
            evalWidget.innerHTML = `
                <div class="eval-header" id="eval-toggle-${Math.random().toString(36).substr(2, 9)}">
                    <div class="eval-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        <span>Auto-Évaluation RAG</span>
                    </div>
                    <div class="eval-scores-preview">
                        <span class="eval-badge faith">Fidélité: ${evalData.faithfulness_score}/5</span>
                        <span class="eval-badge rel">Pertinence: ${evalData.relevance_score}/5</span>
                    </div>
                </div>
                <div class="eval-content hidden">
                    <div class="eval-metric">
                        <div class="eval-metric-name">
                            <span>Fidélité (Zéro Hallucination)</span>
                            <div class="eval-rating">${getStars(evalData.faithfulness_score)}</div>
                        </div>
                        <span class="eval-desc">${evalData.faithfulness_reason}</span>
                    </div>
                    <div class="eval-metric">
                        <div class="eval-metric-name">
                            <span>Pertinence de la réponse</span>
                            <div class="eval-rating">${getStars(evalData.relevance_score)}</div>
                        </div>
                        <span class="eval-desc">${evalData.relevance_reason}</span>
                    </div>
                </div>
            `;
            
            // Add toggle expand/collapse logic
            const header = evalWidget.querySelector(".eval-header");
            const content = evalWidget.querySelector(".eval-content");
            header.addEventListener("click", () => {
                content.classList.toggle("hidden");
            });
            
        } catch (e) {
            evalWidget.innerHTML = `
                <div class="eval-header">
                    <div class="eval-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span>Auto-Évaluation RAG</span>
                    </div>
                    <div class="eval-scores-preview">
                        <span class="eval-badge text-danger">Indisponible</span>
                    </div>
                </div>
            `;
            console.error("Evaluation failed", e);
        }
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Submit user message
    async function sendMessage() {
        const query = chatInput.value.trim();
        if (!query) return;
        
        const creds = getCredentials();
        if (!isConfiguredState()) {
            alert("Clé API manquante. Veuillez la renseigner dans la configuration.");
            return;
        }

        // Add user message to UI & reset input
        appendMessage("user", query);
        chatInput.value = "";
        chatInput.style.height = "auto";
        updateSendButtonState();
        
        // Show typing indicator
        const typingIndicator = showTypingIndicator();

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Provider": creds.headers["X-Provider"],
                    "X-OpenAI-Key": creds.headers["X-OpenAI-Key"],
                    "X-Mistral-Key": creds.headers["X-Mistral-Key"]
                },
                body: JSON.stringify({
                    query,
                    session_id: sessionId,
                    provider: creds.provider
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Erreur de communication avec l'assistant.");
            }

            const data = await res.json();
            
            // Remove typing indicator
            typingIndicator.remove();
            
            // Render AI answer with sources
            const msgDiv = appendMessage("assistant", data.answer, data.sources);
            
            // Fire RAG evaluation in the background if we retrieved documents
            if (data.sources && data.sources.length > 0) {
                const contexts = data.sources.map(s => s.content);
                evaluateAndRenderWidget(msgDiv, query, data.answer, contexts);
            }

        } catch (e) {
            typingIndicator.remove();
            appendMessage("assistant", `⚠️ Erreur : ${e.message}`);
        }
    }

    // Clear chat session memory
    async function clearConversation() {
        if (!confirm("Voulez-vous vraiment effacer cette conversation ?")) return;
        
        try {
            const res = await fetch("/api/clear-history", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ session_id: sessionId })
            });
            
            if (res.ok) {
                // Clear UI
                messagesContainer.innerHTML = "";
                emptyState.classList.remove("hidden");
                // Regenerate session id
                sessionId = generateUUID();
                localStorage.setItem("ag_session_id", sessionId);
            }
        } catch (e) {
            console.error("Failed to clear history", e);
        }
    }

    // --- Event Listeners ---

    // 1. Settings Configuration Changed
    providerSelect.addEventListener("change", (e) => {
        activeProvider = e.target.value;
        localStorage.setItem("ag_provider", activeProvider);
        
        toggleKeyInputs();
        updateActiveBadge();
        updateSendButtonState();
    });

    openaiKeyInput.addEventListener("input", (e) => {
        localStorage.setItem("ag_openai_key", e.target.value);
        updateSendButtonState();
    });

    mistralKeyInput.addEventListener("input", (e) => {
        localStorage.setItem("ag_mistral_key", e.target.value);
        updateSendButtonState();
    });

    // 2. Text Input Area Autosize & Key bindings
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight) + "px";
        updateSendButtonState();
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener("click", sendMessage);
    clearBtn.addEventListener("click", clearConversation);

    // 3. Drag and Drop Ingestion listeners
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("active");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("active");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        uploadFile(file);
    });

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        uploadFile(file);
    });
});

// Polyfill string endsWith helper
if (typeof String.prototype.endswith !== 'function') {
    String.prototype.endswith = function(suffix) {
        return this.indexOf(suffix, this.length - suffix.length) !== -1;
    };
}
