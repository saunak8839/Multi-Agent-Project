'use strict';
const $ = (id) => document.getElementById(id);
let current = null;
let busy = false;
let supportOpen = false;
let routingOpen = true;
let conversations = [];
try {
  const saved = JSON.parse(localStorage.getItem('support-desk-conversations') || '[]');
  if (Array.isArray(saved)) conversations = saved.filter(item => typeof item.id === 'string' && typeof item.title === 'string').slice(0, 30);
} catch { /* Local history is optional. */ }

function remember(id, title) {
  if (!conversations.some(item => item.id === id)) conversations.unshift({id, title: (title || 'Support conversation').slice(0, 65)});
  conversations = conversations.slice(0, 30);
  try { localStorage.setItem('support-desk-conversations', JSON.stringify(conversations)); } catch { /* Storage may be disabled. */ }
  renderList();
}
function renderList() {
  $('conversation-list').replaceChildren();
  if (!conversations.length) {
    const note = document.createElement('p');
    note.className = 'list-empty';
    note.textContent = 'Your conversations will appear here.';
    $('conversation-list').append(note);
  }
  for (const item of conversations) {
    const button = document.createElement('button');
    button.className = `conversation-item${current?.thread_id === item.id ? ' active' : ''}`;
    button.disabled = busy;
    if (current?.thread_id === item.id) button.setAttribute('aria-current', 'page');
    const title = document.createElement('span');
    title.textContent = item.title;
    button.append(title);
    button.addEventListener('click', () => openConversation(item.id));
    $('conversation-list').append(button);
  }
}
function setBusy(value) {
  busy = value;
  $('working').hidden = !value;
  $('new-chat').disabled = value;
  $('open-thread').disabled = value;
  $('message-input').disabled = value || (current && current.status !== 'ready');
  $('send-button').disabled = $('message-input').disabled;
  $('retry-turn').disabled = value;
  document.querySelectorAll('#approval-form button').forEach(button => { button.disabled = value; });
  $('state-pill').textContent = value ? 'Working on it' : current?.status === 'awaiting_approval' ? 'Awaiting review' : current?.status === 'interrupted' ? 'Needs attention' : 'Ready to help';
  renderList();
  renderRouting();
}
function showError(message) {
  $('error-text').textContent = message;
  $('error-banner').hidden = false;
}
function clearError() { $('error-banner').hidden = true; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: {'Content-Type': 'application/json', ...options.headers} });
  let body;
  try { body = await response.json(); } catch { throw new Error('The server returned an unexpected response. Please try again.'); }
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === 'string' ? detail : detail?.message || (response.status === 422 ? 'Please check your message or conversation ID.' : 'The request could not be completed.');
    const error = new Error(message);
    error.threadId = detail?.thread_id;
    error.status = response.status;
    throw error;
  }
  return body;
}
function addMessage(entry) {
  const article = document.createElement('article');
  article.className = `message ${entry.role}`;
  const label = document.createElement('span');
  label.className = 'message-label';
  label.textContent = entry.role === 'user' ? 'YOU' : 'SUPPORT DESK';
  const body = document.createElement('div');
  body.className = 'message-body';
  // Model and user text are always rendered as text, never executable HTML.
  body.textContent = entry.content;
  article.append(label, body);
  if (entry.created_at) {
    const time = document.createElement('time');
    time.className = 'message-time';
    time.dateTime = entry.created_at;
    time.textContent = new Date(entry.created_at).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    article.append(time);
  }
  $('messages').append(article);
}
function renderRouting() {
  const path = current?.routing_path || [];
  const labels = {
    intake: ['Intake', 'Classify the request and choose a route'],
    faq: ['FAQ agent', 'Find an answer in the knowledge base'],
    billing: ['Billing agent', 'Check payments and billing records'],
    technical: ['Technical agent', 'Investigate the device or ask for details'],
    escalation: ['Human review needed', 'Prepare guidance for an urgent request'],
    draft: ['Legacy checkpoint', 'Resume an older conversation'],
    qa: ['QA reviewer', 'Review grounding and correctness'],
    qa_fallback: ['Safe fallback', 'QA retry limit reached'],
    approval_gate: ['Human approval', 'Review the proposed refund'],
    execute_action: ['Action executor', 'Record the approved demo refund'],
    escalate: ['Support follow-up', 'Record the declined proposal'],
    finalize_turn: ['Reply saved', 'Save the completed assistant turn'],
  };
  const statuses = {completed: 'Completed', passed: 'Passed', failed: 'Revision requested', approved: 'Approved', rejected: 'Rejected', awaiting_approval: 'Awaiting decision', interrupted: 'Interrupted'};
  $('routing-steps').replaceChildren();
  $('routing-empty').hidden = Boolean(path.length);
  $('routing-summary').textContent = busy ? 'Processing · last available route shown' : !current ? 'Waiting for a message' : !path.length ? 'Routing will appear on the next turn' : current.status === 'awaiting_approval' ? 'Paused for human approval' : current.status === 'interrupted' ? 'Paused · resume to continue' : `Latest turn · ${path.length} steps`;
  document.querySelectorAll('[data-route]').forEach(branch => {
    branch.classList.toggle('selected', path.some(step => step.node === branch.dataset.route));
  });
  for (const [index, step] of path.entries()) {
    const [title, description] = labels[step.node] || ['Workflow step', ''];
    const item = document.createElement('li');
    item.className = `routing-step ${Object.hasOwn(statuses, step.status) ? step.status : 'completed'}`;
    const number = document.createElement('span');
    number.className = 'routing-number'; number.textContent = String(index + 1).padStart(2, '0');
    const content = document.createElement('div');
    const heading = document.createElement('strong'); heading.textContent = title;
    const detail = document.createElement('p'); detail.textContent = description;
    const badge = document.createElement('span'); badge.className = 'routing-status'; badge.textContent = statuses[step.status] || 'Completed';
    content.append(heading, detail, badge); item.append(number, content); $('routing-steps').append(item);
  }
}
function render(view) {
  current = view;
  $('messages').replaceChildren();
  for (const entry of view?.history || []) addMessage(entry);
  $('welcome').hidden = Boolean(view);
  $('approval-notice').hidden = view?.status !== 'awaiting_approval';
  $('retry-area').hidden = view?.status !== 'interrupted';
  $('page-title').textContent = view ? 'A little closer to sorted.' : "Let's work it out.";
  $('detail-id').textContent = view?.thread_id || 'Not started';
  $('detail-route').textContent = view?.specialist ? view.specialist[0].toUpperCase() + view.specialist.slice(1) : '—';
  $('detail-qa').textContent = view?.qa_passed === true ? `Passed · ${view.qa_attempts} review${view.qa_attempts === 1 ? '' : 's'}` : view?.qa_passed === false ? 'Needs revision' : '—';
  const pending = view?.pending_approval;
  $('approval-card').hidden = !pending;
  $('no-approval').hidden = Boolean(pending);
  if (pending) {
    let amount;
    try { amount = new Intl.NumberFormat(undefined, {style: 'currency', currency: pending.currency}).format(pending.amount_cents / 100); }
    catch { amount = `${pending.currency} ${(pending.amount_cents / 100).toFixed(2)}`; }
    $('approval-amount').textContent = amount;
    $('approval-order').textContent = `Order ${pending.order_id}`;
    $('reviewed-draft').textContent = pending.draft_reply;
  }
  if (view) {
    remember(view.thread_id, view.history.find(entry => entry.role === 'user')?.content);
    history.replaceState(null, '', `/#${view.thread_id}`);
  } else {
    history.replaceState(null, '', '/');
    $('support-key').value = '';
    $('decision-note').value = '';
  }
  setBusy(busy);
}
async function recover(error, title) {
  const id = error.threadId || current?.thread_id;
  let restored = false;
  if (id) {
    remember(id, title);
    try { render(await api(`/api/conversations/${encodeURIComponent(id)}`)); restored = true; }
    catch { /* An initial dependency failure may precede the first checkpoint. */ }
  }
  if (!restored) render(current);
  showError(error.message || 'Could not reach the support service. Check the connection and try again.');
}
async function openConversation(id) {
  if (busy) return;
  clearError();
  setBusy(true);
  try {
    render(await api(`/api/conversations/${encodeURIComponent(id.trim())}`));
    $('support-key').value = '';
    $('decision-note').value = '';
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}
$('message-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = $('message-input').value.trim();
  if (!message || busy || (current && current.status !== 'ready')) return;
  clearError();
  $('welcome').hidden = true;
  addMessage({role: 'user', content: message, created_at: new Date().toISOString()});
  setBusy(true);
  try {
    const path = current ? `/api/conversations/${current.thread_id}/messages` : '/api/conversations';
    render(await api(path, {method: 'POST', body: JSON.stringify({message})}));
    $('message-input').value = '';
  } catch (error) { await recover(error, message); }
  finally { setBusy(false); $('message-input').focus(); }
});
$('message-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('message-form').requestSubmit();
  }
});
$('new-chat').addEventListener('click', () => {
  if (busy) return;
  clearError(); render(null); $('message-input').value = ''; $('message-input').focus();
});
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => {
  $('message-input').value = button.dataset.prompt;
  $('message-input').focus();
});
$('support-toggle').addEventListener('click', () => {
  supportOpen = !supportOpen;
  routingOpen = !supportOpen;
  $('support-panel').hidden = !supportOpen;
  $('routing-panel').hidden = !routingOpen;
  $('routing-toggle').setAttribute('aria-expanded', String(routingOpen));
  $('support-toggle').setAttribute('aria-expanded', String(supportOpen));
  if (!supportOpen) $('support-key').value = '';
});
$('routing-toggle').addEventListener('click', () => {
  routingOpen = !routingOpen;
  $('routing-panel').hidden = !routingOpen;
  $('routing-toggle').setAttribute('aria-expanded', String(routingOpen));
  if (routingOpen) {
    supportOpen = false;
    $('support-panel').hidden = true;
    $('support-toggle').setAttribute('aria-expanded', 'false');
    $('support-key').value = '';
  }
});
$('dismiss-error').addEventListener('click', clearError);
$('retry-turn').addEventListener('click', async () => {
  if (!current || busy) return;
  clearError(); setBusy(true);
  try { render(await api(`/api/conversations/${current.thread_id}/retry`, {method: 'POST'})); }
  catch (error) { await recover(error); }
  finally { setBusy(false); }
});
$('approval-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!current?.pending_approval || busy) return;
  const approved = event.submitter?.value === 'approve';
  clearError(); setBusy(true);
  try {
    render(await api(`/api/conversations/${current.thread_id}/approval`, {
      method: 'POST', headers: {Authorization: `Bearer ${$('support-key').value}`},
      body: JSON.stringify({approved, note: $('decision-note').value}),
    }));
    $('support-key').value = ''; $('decision-note').value = '';
  } catch (error) { await recover(error); }
  finally { setBusy(false); }
});
$('open-thread').addEventListener('click', () => $('thread-dialog').showModal());
$('close-dialog').addEventListener('click', () => $('thread-dialog').close());
$('thread-form').addEventListener('submit', (event) => {
  event.preventDefault(); $('thread-dialog').close(); openConversation($('thread-input').value);
});
const initialId = location.hash.slice(1);
render(null);
if (initialId) openConversation(initialId);
