import { useEffect, useRef, useState } from 'react'

// Demo questions spanning every classification + an unanswerable distractor.
// Swapping the clearance and re-asking these is the whole governance demo.
const SUGGESTIONS = [
  { q: 'What is Aurora’s remote work policy?', tag: 'public' },
  { q: 'What was Aurora’s revenue in Q3?', tag: 'confidential' },
  { q: 'What is the Project Helios budget?', tag: 'restricted' },
  { q: 'What is Aurora’s stock ticker symbol?', tag: 'unanswerable' },
]

function cls(name) {
  return `cls-${name}`
}

function ClassBadge({ name }) {
  return <span className={`badge ${cls(name)}`}>{name}</span>
}

// ---- Sidebar panels --------------------------------------------------------

function PipelinePanel({ trace }) {
  const maxMs = Math.max(1, ...trace.pipeline.map((s) => s.ms))
  return (
    <section className="panel">
      <h3>Pipeline</h3>
      <div className="pipeline">
        {trace.pipeline.map((s, i) => (
          <div className="pipe-step" key={i}>
            <div className="pipe-head">
              <span className="pipe-num">{i + 1}</span>
              <span className="pipe-name">{s.step}</span>
              <span className="pipe-ms">{s.ms} ms</span>
            </div>
            <div className="pipe-detail">{s.detail}</div>
            <div className="bar"><div className="bar-fill" style={{ width: `${(s.ms / maxMs) * 100}%` }} /></div>
          </div>
        ))}
      </div>
    </section>
  )
}

function GovernancePanel({ trace }) {
  const g = trace.governance
  return (
    <section className="panel">
      <h3>Governance gate</h3>
      <div className={`leak ${g.leak ? 'leak-bad' : 'leak-ok'}`}>
        {g.leak
          ? `❌ LEAK: ${g.leak_count} unauthorized chunk(s) reached the model`
          : '✅ No leak — 0 unauthorized chunks reached the model'}
      </div>
      <div className="gov-row">
        <span className="gov-label">Acting as</span>
        <span className="gov-val">{trace.user} · clearance {trace.clearance}</span>
      </div>
      <div className="gov-row">
        <span className="gov-label">Can access</span>
        <span className="gov-val">{g.accessible.map((c) => <ClassBadge key={c} name={c} />)}</span>
      </div>
      <div className="gov-row">
        <span className="gov-label">Withheld</span>
        <span className="gov-val">
          {!g.withheld_shown
            ? <span className="muted">hidden (set RAGUARD_SHOW_WITHHELD=1 for the teaching view)</span>
            : g.withheld_count === 0
            ? <span className="muted">nothing (full access)</span>
            : <>{g.withheld_classes.map((c) => <ClassBadge key={c} name={c} />)}<span className="muted"> ({g.withheld_count} chunks)</span></>}
        </span>
      </div>
    </section>
  )
}

function ContextPanel({ trace }) {
  return (
    <section className="panel">
      <h3>Retrieved context <span className="count">{trace.retrieved.length}</span></h3>
      <p className="panel-sub">Chunks the model was actually allowed to read.</p>
      {trace.retrieved.map((c, i) => (
        <details className="ctx" key={i} open={i === 0}>
          <summary>
            <ClassBadge name={c.classification} />
            <span className="ctx-src">{c.source}</span>
            <span className="ctx-score" title="cosine similarity">{c.score}</span>
          </summary>
          <p className="ctx-text">{c.text}</p>
        </details>
      ))}
    </section>
  )
}

function WithheldPanel({ trace }) {
  if (!trace.withheld.length) return null
  return (
    <section className="panel">
      <h3>Withheld by clearance <span className="count count-lock">{trace.withheld.length}</span></h3>
      <p className="panel-sub">Relevant chunks the pre-filter blocked. Their text is never sent to the client.</p>
      {trace.withheld.map((c, i) => (
        <div className="ctx withheld" key={i}>
          <div className="withheld-head">
            <ClassBadge name={c.classification} />
            <span className="ctx-src">{c.source}</span>
            <span className="ctx-score">{c.score}</span>
          </div>
          <p className="ctx-text redacted">🔒 {'█'.repeat(30)} withheld</p>
        </div>
      ))}
    </section>
  )
}

function TimingsPanel({ trace }) {
  const t = trace.timings
  const rows = [
    ['Embed', t.embed_ms],
    ['Retrieval', t.retrieval_ms],
    ['Generation', t.generation_ms],
    ['Total', t.total_ms],
  ]
  return (
    <section className="panel">
      <h3>Latency</h3>
      <div className="timings">
        {rows.map(([k, v]) => (
          <div className={`timing ${k === 'Total' ? 'timing-total' : ''}`} key={k}>
            <span>{k}</span><span>{v} ms</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function Inspector({ trace }) {
  if (!trace) {
    return (
      <aside className="inspector empty">
        <div className="inspector-empty">
          <div className="eye">◉</div>
          <p>Ask a question to inspect the pipeline, the governance gate, and the retrieved context here.</p>
        </div>
      </aside>
    )
  }
  return (
    <aside className="inspector">
      <PipelinePanel trace={trace} />
      <GovernancePanel trace={trace} />
      <ContextPanel trace={trace} />
      <WithheldPanel trace={trace} />
      <TimingsPanel trace={trace} />
    </aside>
  )
}

// ---- Main app --------------------------------------------------------------

export default function App() {
  const [config, setConfig] = useState(null)
  const [clearance, setClearance] = useState(0)
  const [messages, setMessages] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const threadRef = useRef(null)

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then(setConfig)
      .catch(() => setError('Cannot reach the API on :8090. Is uvicorn running?'))
  }, [])

  useEffect(() => {
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight
  }, [messages, loading])

  const activeTrace = messages.find((m) => m.id === activeId)?.trace ?? null

  async function send(question) {
    const q = (question ?? input).trim()
    if (!q || loading) return
    setInput('')
    setError(null)
    const userMsg = { id: crypto.randomUUID(), role: 'user', content: q, clearance }
    setMessages((m) => [...m, userMsg])
    setLoading(true)
    try {
      const r = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, clearance }),
      })
      if (!r.ok) throw new Error(`API ${r.status}`)
      const data = await r.json()
      const botId = crypto.randomUUID()
      setMessages((m) => [...m, { id: botId, role: 'assistant', content: data.answer, trace: data }])
      setActiveId(botId)
    } catch (e) {
      setError(e.message || 'Request failed')
      setMessages((m) => [...m, { id: crypto.randomUUID(), role: 'assistant', content: '⚠️ Request failed.', error: true }])
    } finally {
      setLoading(false)
    }
  }

  function onKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const users = config?.users ?? []
  const current = users.find((u) => u.clearance === clearance)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">🛡️</span>
          <div>
            <h1>RAGuard</h1>
            <p>Governed RAG — retrieval, generation &amp; access control, observable</p>
          </div>
        </div>
        {config && (
          <div className="model-badges">
            <span className="mbadge">gen: {config.gen_model}</span>
            <span className="mbadge">embed: {config.embed_model} · {config.embed_dim}d</span>
            <span className="mbadge">top-k: {config.top_k}</span>
          </div>
        )}
      </header>

      <div className="layout">
        <main className="chat">
          <div className="clearance-bar">
            <span className="clearance-label">Acting as</span>
            <div className="seg">
              {users.map((u) => (
                <button
                  key={u.name}
                  className={`seg-btn ${u.clearance === clearance ? 'active' : ''}`}
                  onClick={() => setClearance(u.clearance)}
                  title={`Can access: ${u.accessible.join(', ')}`}
                >
                  <span className="seg-name">{u.name}</span>
                  <span className="seg-lvl">L{u.clearance} · {u.label}</span>
                </button>
              ))}
            </div>
            {current && (
              <div className="clearance-access">
                sees {current.accessible.map((c) => <ClassBadge key={c} name={c} />)}
              </div>
            )}
          </div>

          <div className="thread" ref={threadRef}>
            {messages.length === 0 && (
              <div className="welcome">
                <h2>Ask the Aurora knowledge base</h2>
                <p>Pick a clearance above, then ask. Try the same question at different clearances and watch the governance gate in the sidebar.</p>
                <div className="suggest">
                  {SUGGESTIONS.map((s) => (
                    <button key={s.q} className="chip" onClick={() => send(s.q)}>
                      <span className={`dot ${cls(s.tag)}`} />{s.q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`msg ${m.role} ${m.id === activeId ? 'selected' : ''} ${m.trace ? 'has-trace' : ''}`}
                onClick={() => m.trace && setActiveId(m.id)}
              >
                <div className="msg-role">
                  {m.role === 'user'
                    ? <>You <span className="msg-clear">L{m.clearance}</span></>
                    : 'RAGuard'}
                </div>
                <div className={`bubble ${m.error ? 'bubble-error' : ''}`}>{m.content}</div>
                {m.trace && (
                  <div className="msg-sources">
                    {m.trace.retrieved.length > 0 && [...new Set(m.trace.retrieved.map((c) => c.source))].map((src) => {
                      const c = m.trace.retrieved.find((x) => x.source === src)
                      return <span key={src} className={`src-chip ${cls(c.classification)}`}>{src}</span>
                    })}
                    {m.trace.governance.withheld_count > 0 && (
                      <span className="src-chip locked">🔒 {m.trace.governance.withheld_count} withheld</span>
                    )}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="msg assistant">
                <div className="msg-role">RAGuard</div>
                <div className="bubble typing"><span /><span /><span /></div>
              </div>
            )}
          </div>

          {error && <div className="error-bar">{error}</div>}

          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder={`Ask as ${current?.name ?? '...'} (clearance ${clearance})…  ↵ to send, ⇧↵ for newline`}
              rows={1}
            />
            <button className="send" onClick={() => send()} disabled={loading || !input.trim()}>
              {loading ? '…' : 'Send'}
            </button>
          </div>
        </main>

        <Inspector trace={activeTrace} />
      </div>
    </div>
  )
}
