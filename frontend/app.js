let currentUser = null;
let currentDraft = null;
let currentThreadId = null;

// Elements
const loginView = document.getElementById('login-view');
const chatView = document.getElementById('chat-view');
const opsView = document.getElementById('ops-view');
const roleBanner = document.getElementById('role-banner');
const messagesArea = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const draftContainer = document.getElementById('draft-container');
const opsCards = document.getElementById('ops-cards');

const navChat = document.getElementById('nav-chat');
const navOps = document.getElementById('nav-ops');
const logoutBtn = document.getElementById('logout-btn');

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function login(userKey) {
    try {
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_key: userKey })
        });

        if (!res.ok) throw new Error("Login failed");

        const data = await res.json();
        currentUser = data;

        // Update banner
        document.getElementById('banner-avatar').textContent = data.user.role === 'customer' ? 'C' : 'A';
        document.getElementById('banner-name').textContent = data.user.user_key.split('.')[0].toUpperCase();
        document.getElementById('banner-role').textContent = `${data.user.role} ${data.user.account_id ? '(' + data.user.account_id + ')' : '(all accounts)'}`;

        roleBanner.classList.remove('hidden');
        loginView.classList.add('hidden');

        if (data.user.role === 'customer') {
            navOps.classList.add('hidden');
            showChat();
        } else {
            navOps.classList.remove('hidden');
            showChat();
        }
    } catch (e) {
        alert(e.message);
    }
}

function showChat() {
    chatView.classList.remove('hidden');
    opsView.classList.add('hidden');
    navChat.classList.add('active');
    navOps.classList.remove('active');
}

async function showOps() {
    chatView.classList.add('hidden');
    opsView.classList.remove('hidden');
    navOps.classList.add('active');
    navChat.classList.remove('active');

    try {
        opsCards.innerHTML = '<p>Loading issues...</p>';
        const res = await fetch(`/ops/issues?token=${currentUser.token}`);
        if (res.status === 403) {
            opsCards.innerHTML = '<p style="color: red">Access Denied. Internal users only.</p>';
            return;
        }
        if (!res.ok) throw new Error("Failed to load ops data");
        const issues = await res.json();

        opsCards.innerHTML = '';
        if (issues.length === 0) {
            opsCards.innerHTML = '<p>No proactive issues detected.</p>';
            return;
        }

        issues.forEach(issue => {
            const card = document.createElement('div');
            card.className = `ops-card ${issue.severity}`;
            const evidenceText = escapeHtml(issue.evidence_ids.join(', '));
            const titleText = escapeHtml(issue.title);
            const descText = escapeHtml(issue.description);
            const affectedText = escapeHtml(issue.affected_accounts.join(', '));
            card.innerHTML = `
                <span class="badge ${issue.severity}">${escapeHtml(issue.severity)}</span>
                <h3>${titleText}</h3>
                <p>${descText}</p>
                <div class="evidence"><strong>Evidence:</strong> ${evidenceText}</div>
                <div class="evidence"><strong>Accounts:</strong> ${affectedText}</div>
                <div style="margin-top: 1rem;">
                    <button class="nav-btn outline" style="border: 1px solid var(--accent-teal)" data-action='${escapeHtml(JSON.stringify(issue.suggested_action))}'>Take Action</button>
                </div>
            `;
            // Bind click handler safely
            const btn = card.querySelector('button[data-action]');
            btn.addEventListener('click', () => takeOpsAction(issue.suggested_action));
            opsCards.appendChild(card);
        });
    } catch (e) {
        opsCards.innerHTML = `<p style="color: red">Error: ${escapeHtml(e.message)}</p>`;
    }
}

navChat.addEventListener('click', showChat);
navOps.addEventListener('click', showOps);
logoutBtn.addEventListener('click', () => {
    currentUser = null;
    currentDraft = null;
    currentThreadId = null;
    roleBanner.classList.add('hidden');
    chatView.classList.add('hidden');
    opsView.classList.add('hidden');
    loginView.classList.remove('hidden');
    messagesArea.innerHTML = '';
    draftContainer.innerHTML = '';
});

document.getElementById('scan-btn').addEventListener('click', showOps);

function takeOpsAction(actionObj) {
    showChat();
    chatInput.value = `Please execute: ${actionObj.action_type} — ${JSON.stringify(actionObj.payload)}`;
    chatForm.dispatchEvent(new Event('submit'));
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!chatInput.value.trim() || !currentUser) return;

    const msg = chatInput.value.trim();
    chatInput.value = '';

    appendMessage('user', msg);

    // Create assistant message container
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'message assistant';
    assistantMsg.innerHTML = `
        <div class="avatar">AI</div>
        <div class="content">
            <div class="tools-area"></div>
            <div class="text-area"></div>
        </div>
    `;
    messagesArea.appendChild(assistantMsg);

    const toolsArea = assistantMsg.querySelector('.tools-area');
    const textArea = assistantMsg.querySelector('.text-area');

    let secondsElapsed = 0;
    let timerInterval = null;
    const timerSpan = document.createElement('div');
    timerSpan.className = 'thinking-loader';
    timerSpan.innerHTML = `<div class="thinking-spinner"></div> <span>Agent processing... (<span class="timer-sec">0</span>s)</span>`;
    toolsArea.appendChild(timerSpan);

    timerInterval = setInterval(() => {
        secondsElapsed++;
        const secSpan = timerSpan.querySelector('.timer-sec');
        if (secSpan) secSpan.textContent = secondsElapsed;
    }, 1000);

    let toolChipCounter = 0;

    try {
        const res = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, token: currentUser.token })
        });

        if (!res.ok) {
            clearInterval(timerInterval);
            timerSpan.remove();
            const errData = await res.json().catch(() => ({ detail: res.statusText }));
            textArea.innerHTML = `<span style="color: red">Error: ${escapeHtml(errData.detail || res.statusText)}</span>`;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split('\n\n');
            buffer = parts.pop();

            for (const part of parts) {
                if (part.startsWith('data: ')) {
                    const dataStr = part.replace('data: ', '');
                    if (dataStr === '[DONE]') {
                        clearInterval(timerInterval);
                        timerSpan.remove();
                        continue;
                    }

                    try {
                        const data = JSON.parse(dataStr);

                        if (data.type === 'token') {
                            fullText += data.content;
                            textArea.innerHTML = marked.parse(fullText);
                            messagesArea.scrollTop = messagesArea.scrollHeight;
                        } else if (data.type === 'tool_start') {
                            toolChipCounter++;
                            const chipId = `chip-${toolChipCounter}`;
                            const chip = document.createElement('div');
                            chip.className = 'tool-chip in-progress';
                            chip.id = chipId;
                            chip.dataset.tool = data.tool;
                            let args = '';
                            if (data.input) {
                                const argStr = JSON.stringify(data.input);
                                const truncated = argStr.length > 80 ? argStr.substring(0, 80) + '...' : argStr;
                                args = `<span class="tool-args" style="font-size: 0.75rem; opacity: 0.7; margin-left: 6px; font-family: monospace;">${escapeHtml(truncated)}</span>`;
                            }
                            chip.innerHTML = `<div class="tool-spinner"></div> <strong>${escapeHtml(data.tool)}</strong>${args}`;
                            toolsArea.appendChild(chip);
                            messagesArea.scrollTop = messagesArea.scrollHeight;
                        } else if (data.type === 'tool_end') {
                            const chips = Array.from(toolsArea.querySelectorAll(`.tool-chip.in-progress[data-tool="${data.tool}"]`));
                            if (chips.length > 0) {
                                const chip = chips[0]; // oldest in-progress chip for this tool
                                chip.classList.remove('in-progress');
                                const argsEl = chip.querySelector('.tool-args');
                                const argsHtml = argsEl ? argsEl.outerHTML : '';
                                chip.innerHTML = `✓ <strong>${escapeHtml(data.tool)}</strong>${argsHtml}`;
                                chip.style.borderColor = 'var(--accent-teal)';
                                chip.style.color = 'var(--accent-teal)';
                            }
                        } else if (data.type === 'intent') {
                            const chip = document.createElement('div');
                            chip.className = 'intent-chip';
                            chip.innerHTML = `<strong>plan:</strong> ${escapeHtml(data.intent)}`;
                            toolsArea.insertBefore(chip, toolsArea.firstChild);
                        } else if (data.type === 'plan_trace') {
                            if (Array.isArray(data.steps) && data.steps.length) {
                                const details = document.createElement('details');
                                details.className = 'plan-trace';
                                details.innerHTML =
                                    `<summary>Execution plan (${data.steps.length} steps)</summary><ol>` +
                                    data.steps.map(s => `<li>${escapeHtml(String(s))}</li>`).join('') +
                                    `</ol>`;
                                textArea.appendChild(details);
                            }
                        } else if (data.type === 'resolution') {
                            if (data.resolution && data.resolution.binding_source) {
                                const src = data.resolution.binding_source;
                                const typeLabel = src.authority >= 100 ? 'Contract' :
                                    src.authority >= 85 ? 'Current Policy' :
                                    src.authority >= 75 ? 'Product Guide' :
                                    src.authority >= 20 ? 'Deprecated' : 'Context';

                                const card = document.createElement('div');
                                card.className = `source-card ${typeLabel.replace(' ', '-').toLowerCase()}`;
                                card.innerHTML = `
                                    <strong>Binding: ${escapeHtml(src.title)}</strong> (Authority: ${src.authority}, ${escapeHtml(typeLabel)})
                                    <br><small>Scope: ${escapeHtml(src.scope)} | Confidence: ${escapeHtml(data.resolution.confidence || 'unknown')}</small>
                                    ${data.resolution.explanation ? `<br><small><em>${escapeHtml(data.resolution.explanation)}</em></small>` : ''}
                                `;
                                textArea.appendChild(card);
                            }
                        } else if (data.type === 'pending_action') {
                            currentDraft = data.draft.draft_id;
                            currentThreadId = data.thread_id || null;
                            renderConfirmCard(data.draft);
                        } else if (data.type === 'final_answer') {
                            // Stop the timer if it's still running
                            if (timerInterval) {
                                clearInterval(timerInterval);
                                timerInterval = null;
                                timerSpan.remove();
                            }

                            // Backend sends pre-parsed JSON in data.parsed (Pydantic-validated)
                            // Falls back to data.content (raw string) if parsing failed server-side
                            let parsed = data.parsed || null;

                            if (!parsed && data.content) {
                                // Attempt client-side parse as last resort
                                try {
                                    let cleanText = (data.content || '').replace(/```json\s*/gi, '').replace(/```\s*/gi, '').trim();
                                    const firstBrace = cleanText.indexOf('{');
                                    const lastBrace = cleanText.lastIndexOf('}');
                                    if (firstBrace !== -1 && lastBrace > firstBrace) {
                                        cleanText = cleanText.substring(firstBrace, lastBrace + 1);
                                    }
                                    parsed = JSON.parse(cleanText);
                                } catch (e) {
                                    // Give up — render raw
                                    parsed = null;
                                }
                            }

                            if (parsed && parsed.verdict) {
                                // If the ReAct path already streamed a good answer as tokens,
                                // don't overwrite it with a thin verdict box. Check if textArea
                                // already has substantial content from token streaming.
                                const existingText = textArea.innerText || '';
                                if (existingText.length > 100 && parsed.reasoning === '' && parsed.confidence === 'medium') {
                                    // ReAct path already provided the answer via streaming tokens.
                                    // Skip the verdict box to avoid the contradictory display.
                                    messagesArea.scrollTop = messagesArea.scrollHeight;
                                    continue;
                                }

                                let conflictsHtml = '';
                                if (parsed.conflicts && parsed.conflicts.length > 0) {
                                    conflictsHtml = `
                                        <h4 style="margin: 15px 0 5px 0; color: var(--accent-rose)">Conflicts Identified</h4>
                                        <ul style="margin: 0; padding-left: 1.2em; opacity: 0.9">${parsed.conflicts.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul>
                                    `;
                                }
                                let citationsHtml = '';
                                if (parsed.citations && parsed.citations.length > 0) {
                                    citationsHtml = `
                                        <h4 style="margin: 15px 0 5px 0; color: var(--text-secondary)">Sources</h4>
                                        <ul style="margin: 0; padding-left: 1.2em; font-size: 0.85rem; opacity: 0.8">${parsed.citations.map(c => `<li><strong>${escapeHtml(c.title || '')}</strong> (authority ${c.authority || '?'}) ${c.excerpt ? '— ' + escapeHtml(c.excerpt.substring(0, 120)) : ''}</li>`).join('')}</ul>
                                    `;
                                }
                                const verdictHtml = `
                                    <div class="verdict-box" style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--accent-teal); margin-top: 15px;">
                                        <h4 style="margin: 0 0 10px 0; color: var(--accent-teal)">Verdict</h4>
                                        <p style="margin: 0">${escapeHtml(parsed.verdict)}</p>
                                        <h4 style="margin: 15px 0 10px 0; color: var(--accent-amber)">Reasoning</h4>
                                        <p style="margin: 0; white-space: pre-wrap;">${escapeHtml(parsed.reasoning || '')}</p>
                                        ${conflictsHtml}
                                        ${citationsHtml}
                                        ${parsed.confidence ? `<div style="margin-top: 12px; opacity: 0.8"><strong>Confidence:</strong> ${escapeHtml(parsed.confidence)}</div>` : ''}
                                        ${parsed.suggested_action && parsed.suggested_action !== 'null' ? `<div style="margin-top: 5px; opacity: 0.8"><strong>Next step:</strong> ${escapeHtml(parsed.suggested_action)}</div>` : ''}
                                    </div>
                                `;
                                textArea.innerHTML += verdictHtml;
                            } else {
                                // Render whatever we have as markdown
                                const raw = data.content || (parsed ? JSON.stringify(parsed, null, 2) : 'No answer generated.');
                                textArea.innerHTML += `<div style="margin-top: 15px; border-left: 4px solid var(--accent-teal); padding: 10px 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">${marked.parse(raw)}</div>`;
                            }
                            messagesArea.scrollTop = messagesArea.scrollHeight;
                        } else if (data.type === 'error') {
                            textArea.innerHTML += `<br><span style="color: red">Error: ${escapeHtml(data.content)}</span>`;
                        }
                    } catch(err) {
                        console.error('JSON parse error', err, dataStr);
                    }
                }
            }
        }

        // If timer is still running (no [DONE] received), clean up
        if (timerInterval) {
            clearInterval(timerInterval);
            timerSpan.remove();
        }
    } catch(e) {
        if (timerInterval) clearInterval(timerInterval);
        timerSpan.remove();
        textArea.innerHTML = `<span style="color: red">Connection error: ${escapeHtml(e.message)}</span>`;
    }
});

function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `
        <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="content">${role === 'user' ? escapeHtml(text) : text}</div>
    `;
    messagesArea.appendChild(msg);
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function renderConfirmCard(draft) {
    const payloadStr = JSON.stringify(draft.payload || {}, null, 2);
    draftContainer.innerHTML = `
        <div class="confirm-card glass-panel" style="margin-left: 3.5rem; max-width: 80%;">
            <h3 style="color: var(--accent-amber); margin-bottom: 0.5rem;">⚠ Confirmation Required</h3>
            <p>Action: <strong>${escapeHtml(draft.action_type)}</strong></p>
            <pre style="background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.8rem;">${escapeHtml(payloadStr)}</pre>
            <div class="confirm-actions">
                <button class="btn-confirm" id="confirm-yes-btn">✓ Confirm & Execute</button>
                <button class="btn-cancel" id="confirm-no-btn">✗ Cancel</button>
            </div>
        </div>
    `;
    document.getElementById('confirm-yes-btn').addEventListener('click', () => confirmAction(true));
    document.getElementById('confirm-no-btn').addEventListener('click', () => confirmAction(false));
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

async function confirmAction(isConfirmed) {
    if (!currentDraft) return;

    const draftId = currentDraft;
    currentDraft = null;
    draftContainer.innerHTML = '';

    appendMessage('user', isConfirmed ? "Yes, please execute that." : "No, cancel that.");

    try {
        const res = await fetch('/chat/confirm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                draft_id: draftId,
                confirm: isConfirmed,
                token: currentUser.token,
                thread_id: currentThreadId,
            })
        });

        const data = await res.json();

        if (!res.ok) {
            appendMessage('assistant', `Failed: ${escapeHtml(data.detail || 'Unknown error')}`);
        } else {
            let msgText = data.message || 'Done.';
            try {
                const cleanText = msgText.replace(/```json/g, '').replace(/```/g, '').trim();
                const parsed = JSON.parse(cleanText);
                if (parsed.verdict) {
                    msgText = `
                        <div class="verdict-box" style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--accent-teal);">
                            <h4 style="margin: 0 0 10px 0; color: var(--accent-teal)">Result</h4>
                            <p style="margin: 0">${escapeHtml(parsed.verdict)}</p>
                        </div>
                    `;
                }
            } catch(e) {
                msgText = escapeHtml(msgText);
            }
            appendMessage('assistant', msgText);
        }
    } catch(e) {
        appendMessage('assistant', `Error: ${escapeHtml(e.message)}`);
    }
    currentThreadId = null;
}
