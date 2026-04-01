"use client";

import { useState, useEffect, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Data & constants                                                   */
/* ------------------------------------------------------------------ */

const STAGES = [
  { key: "document", label: "Document", icon: "DOC" },
  { key: "chunk",    label: "Chunk",    icon: "CHK" },
  { key: "embed",    label: "Embed",    icon: "VEC" },
  { key: "store",    label: "Store",    icon: "DB"  },
  { key: "retrieve", label: "Retrieve", icon: "QRY" },
  { key: "generate", label: "Generate", icon: "GEN" },
] as const;

type StageKey = (typeof STAGES)[number]["key"];

const SAMPLE_DOC =
  "AI Congress uses a multi-agent architecture where autonomous LLM agents " +
  "collaborate through structured debate and weighted voting to produce " +
  "consensus answers. Each agent can leverage tools including web search, " +
  "RAG retrieval, and chain-of-thought reasoning. The swarm orchestrator " +
  "coordinates parallel execution across models like Mistral, Phi-3, and " +
  "LLaMA, collecting responses that are ranked by a voting engine using " +
  "confidence-based ensemble methods.";

const CC = ["#facc15", "#38bdf8", "#4ade80", "#f472b6", "#a78bfa"];

const CHUNKS = [
  { text: "AI Congress uses a multi-agent architecture where autonomous LLM agents collaborate through structured debate and weighted voting", c: CC[0] },
  { text: "weighted voting to produce consensus answers. Each agent can leverage tools including web search, RAG retrieval", c: CC[1] },
  { text: "RAG retrieval, and chain-of-thought reasoning. The swarm orchestrator coordinates parallel execution across models", c: CC[2] },
  { text: "parallel execution across models like Mistral, Phi-3, and LLaMA, collecting responses that are ranked", c: CC[3] },
  { text: "ranked by a voting engine using confidence-based ensemble methods.", c: CC[4] },
];

const EMB = [
  [0.23, -0.15, 0.87, 0.42, -0.31, 0.09, 0.55, -0.72],
  [0.18, 0.61, -0.44, 0.33, 0.07, -0.56, 0.29, 0.81],
  [-0.39, 0.47, 0.12, -0.68, 0.54, 0.21, -0.15, 0.63],
  [0.71, -0.22, 0.38, 0.15, -0.49, 0.66, -0.08, 0.44],
  [-0.11, 0.34, -0.57, 0.82, 0.26, -0.41, 0.73, -0.19],
];

type QData = { qv: number[]; m: number[]; d: number[]; a: string };

const QUERIES: Record<string, QData> = {
  "How do agents collaborate?": {
    qv: [0.21, -0.12, 0.83, 0.38, -0.28, 0.11, 0.52, -0.69],
    m: [0, 1, 2],
    d: [0.034, 0.187, 0.241],
    a: "Agents collaborate through a multi-agent architecture [1] where they " +
       "engage in structured debate and weighted voting to reach consensus [1]. " +
       "Each agent leverages specialized tools including RAG retrieval and " +
       "chain-of-thought reasoning [2], coordinated by a swarm orchestrator " +
       "that manages parallel execution [3].",
  },
  "What models are used?": {
    qv: [0.68, -0.19, 0.35, 0.11, -0.45, 0.62, -0.05, 0.41],
    m: [3, 4, 2],
    d: [0.029, 0.156, 0.312],
    a: "The system runs models including Mistral, Phi-3, and LLaMA in " +
       "parallel execution [1]. Responses from these models are collected " +
       "and ranked by a voting engine [2] that uses confidence-based ensemble " +
       "methods. The swarm orchestrator coordinates this parallel model " +
       "execution [3].",
  },
  "How does voting work?": {
    qv: [0.15, 0.58, -0.40, 0.30, 0.10, -0.52, 0.25, 0.77],
    m: [1, 4, 0],
    d: [0.042, 0.118, 0.274],
    a: "Voting works through weighted consensus: each agent produces an answer " +
       "with a confidence score [1], then a voting engine applies " +
       "confidence-based ensemble methods to rank all responses [2]. The " +
       "multi-agent architecture uses structured debate combined with " +
       "weighted voting to produce consensus answers [3].",
  },
};

const Q_OPTS = Object.keys(QUERIES);

const mono = { fontFamily: "var(--font-mono)" } as const;
const labelS = { ...mono, fontSize: "0.72rem", color: "#a1a1aa", marginBottom: 8 } as const;
const panelBg = { background: "rgba(0,0,0,0.3)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.06)" } as const;
const slideIn = (i: number) => ({ animation: `slideInRight 0.3s ease-out ${i * 0.08}s both` });

function useTypewriter(text: string, speed: number, active: boolean) {
  const [out, setOut] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!active) { setOut(""); setDone(false); return; }
    let i = 0; setOut(""); setDone(false);
    const iv = setInterval(() => {
      i++; setOut(text.slice(0, i));
      if (i >= text.length) { clearInterval(iv); setDone(true); }
    }, speed);
    return () => clearInterval(iv);
  }, [text, speed, active]);
  return { out, done };
}

export default function RAGPipelineWidget() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  const [query, setQuery] = useState(Q_OPTS[0]);
  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState<StageKey | null>(null);
  const [completed, setCompleted] = useState<Set<StageKey>>(new Set());
  const [frozen, setFrozen] = useState<StageKey | null>(null);
  const [stageReady, setStageReady] = useState<Record<string, boolean>>({});
  const [genOn, setGenOn] = useState(false);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const q = QUERIES[query];
  const { out: genText, done: genDone } = useTypewriter(q.a, 18, genOn);

  const clearTimers = useCallback(() => { timers.current.forEach(clearTimeout); timers.current = []; }, []);
  const reset = useCallback(() => { clearTimers(); setRunning(false); setActiveStage(null); setCompleted(new Set()); setFrozen(null); setStageReady({}); setGenOn(false); }, [clearTimers]);
  const sched = useCallback((fn: () => void, ms: number) => { const t = setTimeout(fn, ms); timers.current.push(t); }, []);

  /* Animate through all pipeline stages */
  const runPipeline = useCallback(() => {
    reset();
    setRunning(true);
    const dur = [800, 1000, 900, 700, 1000, 100];
    let acc = 200;

    STAGES.forEach((s, i) => {
      sched(() => {
        setActiveStage(s.key);
        setStageReady((p) => ({ ...p, [s.key]: true }));
        if (s.key === "generate") setGenOn(true);
      }, acc);
      acc += dur[i];
      if (i < STAGES.length - 1) {
        sched(() => {
          setCompleted((p) => new Set([...p, s.key]));
          setActiveStage(null);
        }, acc);
      }
    });

    const typeTime = q.a.length * 18 + 400;
    sched(() => {
      setCompleted((p) => new Set([...p, "generate"]));
      setActiveStage(null);
      setRunning(false);
    }, acc + typeTime);
  }, [reset, sched, q.a.length]);

  /* Which stage detail to show */
  const inspect = frozen ?? activeStage;

  /* Render citation-highlighted text */
  const cited = (text: string) =>
    text.split(/(\[\d\])/g).map((part, i) => {
      const m = part.match(/^\[(\d)\]$/);
      if (m) {
        const idx = +m[1] - 1;
        const col = q.m[idx] !== undefined ? CHUNKS[q.m[idx]].c : "#a1a1aa";
        return (
          <span key={i} style={{ color: col, fontWeight: 700, fontSize: "0.72rem", verticalAlign: "super", cursor: "pointer" }} title={`Chunk ${q.m[idx]}`}>
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });

  if (!mounted) return null;

  /* CSS class for each stage box */
  const cls = (k: StageKey) =>
    ["pipeline-stage", activeStage === k && "active", completed.has(k) && "completed", frozen === k && "active"]
      .filter(Boolean)
      .join(" ");

  /* ---------------------------------------------------------------- */
  /*  Stage detail renderer                                            */
  /* ---------------------------------------------------------------- */

  const detail = (k: StageKey) => {
    const fade = { animation: "fadeIn 0.4s ease-out" } as const;

    switch (k) {
      case "document":
        return (
          <div style={fade}>
            <div style={labelS}>Parsed document (PDF / DOCX / MD)</div>
            <div style={{ ...panelBg, background: "rgba(0,0,0,0.4)", padding: "0.75rem", ...mono, fontSize: "0.8rem", lineHeight: 1.7, color: "#b4b4bc" }}>
              {SAMPLE_DOC}
            </div>
            <div style={{ marginTop: 8, fontSize: "0.7rem", color: "#a1a1aa" }}>
              {SAMPLE_DOC.split(" ").length} words | ~{Math.ceil(SAMPLE_DOC.length / 4)} tokens
            </div>
          </div>
        );

      case "chunk":
        return (
          <div style={fade}>
            <div style={labelS}>Chunked (512 tokens, 50 token overlap)</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {CHUNKS.map((ch, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, ...slideIn(i) }}>
                  <span style={{ flexShrink: 0, width: 20, height: 20, borderRadius: 4, background: ch.c, opacity: 0.8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.6rem", fontWeight: 700, color: "#000" }}>
                    {i}
                  </span>
                  <span style={{ fontSize: "0.75rem", color: "#b4b4bc", borderLeft: `2px solid ${ch.c}`, paddingLeft: 8, lineHeight: 1.5 }}>
                    {ch.text}
                  </span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10, fontSize: "0.68rem", color: "#a1a1aa", fontStyle: "italic" }}>
              Overlapping regions shown where chunks share tokens
            </div>
          </div>
        );

      case "embed":
        return (
          <div style={fade}>
            <div style={labelS}>sentence-transformers embedding (768-dim, first 8 shown)</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {EMB.map((v, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, ...slideIn(i) }}>
                  <span style={{ flexShrink: 0, width: 8, height: 8, borderRadius: 2, background: CC[i], opacity: 0.8 }} />
                  <span style={{ ...mono, fontSize: "0.72rem", color: "#8a8a96" }}>
                    [{v.map((n) => n.toFixed(2)).join(", ")}, ...]
                  </span>
                </div>
              ))}
            </div>
          </div>
        );

      case "store":
        return (
          <div style={fade}>
            <div style={labelS}>Vector Store (Oracle Vector DB)</div>
            <div style={{ position: "relative", height: 120, ...panelBg, overflow: "hidden" }}>
              {EMB.map((v, i) => {
                const x = ((v[0] + 1) / 2) * 80 + 5;
                const y = ((v[1] + 1) / 2) * 70 + 10;
                return (
                  <div
                    key={i}
                    title={`Chunk ${i}`}
                    style={{
                      position: "absolute",
                      left: `${x}%`,
                      top: `${y}%`,
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: CC[i],
                      opacity: 0.85,
                      transform: "translate(-50%,-50%)",
                      animation: `fadeIn 0.3s ease-out ${i * 0.1}s both`,
                      boxShadow: `0 0 6px ${CC[i]}44`,
                    }}
                  />
                );
              })}
              <div style={{ position: "absolute", bottom: 6, right: 10, ...mono, fontSize: "0.62rem", color: "#555" }}>
                5 vectors stored
              </div>
            </div>
          </div>
        );

      case "retrieve":
        return (
          <div style={fade}>
            <div style={labelS}>Similarity search: top-3 nearest chunks</div>
            <div style={{ ...panelBg, padding: "0.6rem 0.75rem", border: "1px solid rgba(251,146,60,0.2)", marginBottom: 8 }}>
              <div style={{ ...mono, fontSize: "0.68rem", color: "#fb923c", marginBottom: 4 }}>
                Query vector
              </div>
              <div style={{ ...mono, fontSize: "0.72rem", color: "#8a8a96" }}>
                [{q.qv.map((n) => n.toFixed(2)).join(", ")}, ...]
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {q.m.map((ci, rank) => (
                <div
                  key={rank}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "0.4rem 0.6rem",
                    borderRadius: 6,
                    background: "rgba(255,255,255,0.02)",
                    border: `1px solid ${CHUNKS[ci].c}33`,
                    ...slideIn(rank),
                  }}
                >
                  <span style={{ flexShrink: 0, width: 18, height: 18, borderRadius: 4, background: CHUNKS[ci].c, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.58rem", fontWeight: 700, color: "#000" }}>
                    {ci}
                  </span>
                  <span style={{ flex: 1, fontSize: "0.72rem", color: "#b4b4bc", lineHeight: 1.4 }}>
                    {CHUNKS[ci].text.slice(0, 70)}...
                  </span>
                  <span style={{ flexShrink: 0, ...mono, fontSize: "0.68rem", color: q.d[rank] < 0.1 ? "#4ade80" : "#a1a1aa" }}>
                    d={q.d[rank].toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );

      case "generate":
        return (
          <div style={fade}>
            <div style={labelS}>LLM generation with retrieved context</div>
            <div style={{ ...panelBg, padding: "0.6rem 0.75rem", marginBottom: 10 }}>
              <div style={{ ...mono, fontSize: "0.65rem", color: "#555", marginBottom: 4 }}>
                PROMPT CONTEXT
              </div>
              <div style={{ ...mono, fontSize: "0.7rem", color: "#666", lineHeight: 1.5 }}>
                Given the following context chunks, answer the question.
                <br />
                [Chunk {q.m[0]}] [Chunk {q.m[1]}] [Chunk {q.m[2]}]
                <br />
                Question: {query}
              </div>
            </div>
            <div style={{ fontSize: "0.82rem", color: "#e4e4e7", lineHeight: 1.75, minHeight: 60 }}>
              {genDone ? (
                cited(q.a)
              ) : (
                <span>
                  {cited(genText)}
                  <span className="typewriter-cursor" />
                </span>
              )}
            </div>
            {genDone && (
              <div style={{ marginTop: 12, padding: "0.5rem 0.65rem", background: "rgba(0,0,0,0.25)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.05)", ...mono, fontSize: "0.68rem", color: "#777", animation: "fadeIn 0.5s ease-out" }}>
                <div style={{ fontWeight: 600, color: "#a1a1aa", marginBottom: 4 }}>
                  Source Attribution
                </div>
                {q.m.map((ci, i) => (
                  <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 3 }}>
                    <span style={{ color: CHUNKS[ci].c, fontWeight: 700 }}>[{i + 1}]</span>
                    <span>Chunk {ci}: {CHUNKS[ci].text.slice(0, 55)}...</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  /* Arrow connector between stage boxes */
  const Arrow = () => (
    <svg width="24" height="16" viewBox="0 0 24 16" style={{ flexShrink: 0, opacity: 0.3 }}>
      <path d="M0 8 L18 8 M14 3 L20 8 L14 13" stroke="#a1a1aa" strokeWidth="1.5" fill="none" />
    </svg>
  );

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="widget-container ch6">
      <div className="widget-label">Interactive &middot; RAG Pipeline</div>

      {/* Query selector */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ ...mono, fontSize: "0.72rem", color: "#a1a1aa", marginBottom: 6 }}>
          Query
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {Q_OPTS.map((o) => (
            <button
              key={o}
              className={`btn-mono${query === o ? " active" : ""}`}
              disabled={running}
              style={{ opacity: running && query !== o ? 0.4 : 1 }}
              onClick={() => {
                if (!running) { setQuery(o); reset(); }
              }}
            >
              {o}
            </button>
          ))}
        </div>
      </div>

      {/* Pipeline stage boxes */}
      <div
        className="scrollbar-hide"
        style={{ display: "flex", alignItems: "center", gap: 6, overflowX: "auto", paddingBottom: 4, marginBottom: 16 }}
      >
        {STAGES.map((s, i) => (
          <div key={s.key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              className={cls(s.key)}
              onClick={() => {
                if (completed.has(s.key) || activeStage === s.key)
                  setFrozen((p) => (p === s.key ? null : s.key));
              }}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, padding: "0.55rem 0.7rem", minWidth: 72, position: "relative" }}
            >
              <span
                style={{
                  ...mono,
                  fontSize: "0.68rem",
                  fontWeight: 700,
                  letterSpacing: "0.05em",
                  color: activeStage === s.key || frozen === s.key ? "#fb923c" : completed.has(s.key) ? "#4ade80" : "#555",
                }}
              >
                {s.icon}
              </span>
              <span
                style={{
                  fontSize: "0.7rem",
                  color: activeStage === s.key || frozen === s.key ? "#e4e4e7" : completed.has(s.key) ? "#a1a1aa" : "#666",
                }}
              >
                {s.label}
              </span>
              {frozen === s.key && (
                <span style={{ position: "absolute", top: -5, right: -5, width: 14, height: 14, borderRadius: "50%", background: "#fb923c", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.52rem", color: "#000", fontWeight: 700 }}>
                  i
                </span>
              )}
            </div>
            {i < STAGES.length - 1 && <Arrow />}
          </div>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 16 }}>
        <button className="btn-mono" onClick={runPipeline} disabled={running}>
          {running ? "Running..." : "Run Pipeline"}
        </button>
        {(completed.size > 0 || running) && (
          <button className="btn-mono" onClick={reset}>Reset</button>
        )}
        {frozen && (
          <span style={{ ...mono, fontSize: "0.7rem", color: "#fb923c", display: "flex", alignItems: "center", gap: 4 }}>
            Inspecting: {frozen}
            <button className="btn-mono" style={{ padding: "0.2rem 0.5rem", fontSize: "0.65rem" }} onClick={() => setFrozen(null)}>
              dismiss
            </button>
          </span>
        )}
      </div>

      {/* Progress bar */}
      {running && (
        <div style={{ height: 2, background: "rgba(255,255,255,0.06)", borderRadius: 1, marginBottom: 16, overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              background: "linear-gradient(90deg, #facc15, #fb923c)",
              borderRadius: 1,
              width: `${((completed.size + (activeStage ? 0.5 : 0)) / STAGES.length) * 100}%`,
              transition: "width 0.4s ease-out",
            }}
          />
        </div>
      )}

      {/* Stage detail panel */}
      {inspect && stageReady[inspect] && (
        <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: "1rem", marginBottom: 12 }}>
          <div style={{ ...mono, fontSize: "0.68rem", color: "#fb923c", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Stage {STAGES.findIndex((s) => s.key === inspect) + 1} / {STAGES.length} &mdash;{" "}
            {STAGES.find((s) => s.key === inspect)?.label}
          </div>
          {detail(inspect)}
        </div>
      )}

      {/* Idle hint */}
      {!running && completed.size === 0 && (
        <div style={{ textAlign: "center", padding: "2rem 1rem", color: "#555", ...mono, fontSize: "0.8rem" }}>
          Select a query above, then click{" "}
          <span style={{ color: "#a1a1aa" }}>Run Pipeline</span> to watch RAG in action.
          <br />
          <span style={{ fontSize: "0.7rem", color: "#444", marginTop: 4, display: "inline-block" }}>
            Click any completed stage to inspect its intermediate output.
          </span>
        </div>
      )}

      {/* Completed summary */}
      {!running && completed.size === STAGES.length && !frozen && (
        <div style={{ background: "rgba(74,222,128,0.04)", border: "1px solid rgba(74,222,128,0.15)", borderRadius: 8, padding: "0.75rem 1rem", animation: "fadeIn 0.6s ease-out" }}>
          <div style={{ ...mono, fontSize: "0.72rem", color: "#4ade80", marginBottom: 8, fontWeight: 600 }}>
            Pipeline Complete
          </div>
          <div style={{ fontSize: "0.78rem", color: "#b4b4bc", lineHeight: 1.7 }}>
            {cited(q.a)}
          </div>
          <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6, ...mono, fontSize: "0.68rem" }}>
            {q.m.map((ci, i) => (
              <span
                key={i}
                style={{
                  padding: "0.15rem 0.45rem",
                  borderRadius: 4,
                  background: `${CHUNKS[ci].c}18`,
                  border: `1px solid ${CHUNKS[ci].c}33`,
                  color: CHUNKS[ci].c,
                }}
              >
                [{i + 1}] chunk {ci} (d={q.d[i].toFixed(3)})
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
