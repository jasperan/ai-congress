"use client";
import { useState, useEffect, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */
type AgentId = "planner" | "worker1" | "worker2" | "critic" | "judge" | "synthesizer";
type MsgKind = "task" | "result" | "critique" | "approval";

interface AgentNode {
  id: AgentId;
  label: string;
  short: string;
  role: string;
  cx: number;
  cy: number;
  color: string;
}

interface FlyingMsg {
  id: number;
  kind: MsgKind;
  from: AgentId;
  to: AgentId;
  progress: number; // 0-1
  label: string;
}

interface LogEntry {
  ts: number;
  text: string;
  color: string;
}

interface AgentStats {
  sent: number;
  received: number;
  retries: number;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */
const MSG_COLORS: Record<MsgKind, string> = {
  task: "#f97316",
  result: "#4ade80",
  critique: "#f472b6",
  approval: "#22d3ee",
};

const AGENTS: AgentNode[] = [
  { id: "planner",     label: "Planner",     short: "P",  role: "Decomposes user queries into sub-tasks and dispatches them to workers.", cx: 300, cy: 52,  color: "#f97316" },
  { id: "worker1",     label: "Worker 1",    short: "W1", role: "Executes sub-task A using tool calls, retrieval, or code generation.",     cx: 120, cy: 160, color: "#38bdf8" },
  { id: "worker2",     label: "Worker 2",    short: "W2", role: "Executes sub-task B in parallel with Worker 1.",                          cx: 300, cy: 160, color: "#38bdf8" },
  { id: "critic",      label: "Critic",      short: "Cr", role: "Reviews worker outputs for correctness, completeness, and consistency.",   cx: 480, cy: 160, color: "#f472b6" },
  { id: "judge",       label: "Judge",       short: "J",  role: "Evaluates revised results and decides if quality bar is met.",             cx: 180, cy: 275, color: "#a78bfa" },
  { id: "synthesizer", label: "Synthesizer", short: "Sy", role: "Merges approved partial results into a single coherent final answer.",     cx: 420, cy: 275, color: "#4ade80" },
];

const AGENT_MAP = Object.fromEntries(AGENTS.map((a) => [a.id, a])) as Record<AgentId, AgentNode>;

/* The full orchestration sequence */
const SEQUENCE: { from: AgentId; to: AgentId; kind: MsgKind; label: string; delay: number }[] = [
  { from: "planner",  to: "worker1",     kind: "task",     label: "Sub-task A dispatched",       delay: 0 },
  { from: "planner",  to: "worker2",     kind: "task",     label: "Sub-task B dispatched",       delay: 200 },
  { from: "worker1",  to: "critic",      kind: "result",   label: "Worker 1 result submitted",   delay: 1400 },
  { from: "worker2",  to: "critic",      kind: "result",   label: "Worker 2 result submitted",   delay: 1700 },
  { from: "critic",   to: "worker1",     kind: "critique", label: "Critic: revise section 2",    delay: 3000 },
  { from: "critic",   to: "worker2",     kind: "critique", label: "Critic: add citations",       delay: 3200 },
  { from: "worker1",  to: "judge",       kind: "result",   label: "Worker 1 revised result",     delay: 4600 },
  { from: "worker2",  to: "judge",       kind: "result",   label: "Worker 2 revised result",     delay: 4900 },
  { from: "judge",    to: "synthesizer", kind: "approval", label: "Judge: approved, merge",       delay: 6200 },
  { from: "synthesizer", to: "planner",  kind: "result",   label: "Final answer synthesized",    delay: 7500 },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function formatTs(ts: number) {
  const s = ((ts % 60000) / 1000).toFixed(1);
  return `+${s.padStart(5, " ")}s`;
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */
export default function ACPWidget() {
  const [mounted, setMounted] = useState(false);
  const [running, setRunning] = useState(false);
  const [messages, setMessages] = useState<FlyingMsg[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentId | null>(null);
  const [heartbeats, setHeartbeats] = useState<Record<AgentId, number>>({} as Record<AgentId, number>);
  const [phase, setPhase] = useState<string>("idle");
  const [stepIndex, setStepIndex] = useState(-1);
  const [agentStats, setAgentStats] = useState<Record<AgentId, AgentStats>>(() => {
    const init: Record<string, AgentStats> = {};
    AGENTS.forEach((a) => { init[a.id] = { sent: 0, received: 0, retries: 0 }; });
    return init as Record<AgentId, AgentStats>;
  });
  const [supervisorRetries, setSupervisorRetries] = useState(0);

  const msgIdRef = useRef(0);
  const rafRef = useRef<number>(0);
  const startRef = useRef(0);
  const spawnedRef = useRef<Set<number>>(new Set());
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setMounted(true); }, []);

  /* Heartbeat pulse */
  useEffect(() => {
    if (!mounted) return;
    const iv = setInterval(() => {
      const now = Date.now();
      const beats: Record<string, number> = {};
      AGENTS.forEach((a) => { beats[a.id] = now; });
      setHeartbeats(beats as Record<AgentId, number>);
    }, 2000);
    return () => clearInterval(iv);
  }, [mounted]);

  /* Auto-scroll log */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  /* Animation loop */
  const tick = useCallback(() => {
    const elapsed = Date.now() - startRef.current;

    /* Spawn messages according to sequence timing */
    SEQUENCE.forEach((step, i) => {
      if (!spawnedRef.current.has(i) && elapsed >= step.delay) {
        spawnedRef.current.add(i);
        const id = ++msgIdRef.current;
        setMessages((prev) => [...prev, { id, kind: step.kind, from: step.from, to: step.to, progress: 0, label: step.label }]);
        setLog((prev) => [...prev, { ts: elapsed, text: step.label, color: MSG_COLORS[step.kind] }]);
        setStepIndex(i);

        /* Update agent stats */
        setAgentStats((prev) => {
          const next = { ...prev };
          next[step.from] = { ...next[step.from], sent: next[step.from].sent + 1 };
          next[step.to] = { ...next[step.to], received: next[step.to].received + 1 };
          /* Simulate a retry on critic feedback steps */
          if (step.kind === "critique") {
            next[step.to] = { ...next[step.to], retries: next[step.to].retries + 1 };
            setSupervisorRetries((r) => r + 1);
          }
          return next;
        });

        /* Update phase label */
        if (i === 0) setPhase("dispatching sub-tasks");
        else if (i === 2) setPhase("workers reporting");
        else if (i === 4) setPhase("critic reviewing");
        else if (i === 6) setPhase("revised results");
        else if (i === 8) setPhase("judge approving");
        else if (i === 9) setPhase("synthesizing answer");
      }
    });

    /* Advance flying messages */
    setMessages((prev) => {
      const next = prev.map((m) => ({ ...m, progress: Math.min(1, m.progress + 0.018) }));
      return next.filter((m) => m.progress < 1);
    });

    /* Check if done */
    const allSpawned = spawnedRef.current.size === SEQUENCE.length;
    setMessages((prev) => {
      if (allSpawned && prev.length === 0) {
        setRunning(false);
        setPhase("complete");
        return prev;
      }
      return prev;
    });

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const handleSendTask = useCallback(() => {
    if (running) return;
    setRunning(true);
    setMessages([]);
    setLog([{ ts: 0, text: 'User query: "Summarize AI safety research"', color: "#e4e4e7" }]);
    setPhase("planning");
    setStepIndex(-1);
    setSupervisorRetries(0);
    setAgentStats(() => {
      const init: Record<string, AgentStats> = {};
      AGENTS.forEach((a) => { init[a.id] = { sent: 0, received: 0, retries: 0 }; });
      return init as Record<AgentId, AgentStats>;
    });
    spawnedRef.current = new Set();
    startRef.current = Date.now();
    rafRef.current = requestAnimationFrame(tick);
  }, [running, tick]);

  /* Cleanup */
  useEffect(() => {
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  const handleReset = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setRunning(false);
    setMessages([]);
    setLog([]);
    setPhase("idle");
    setSelectedAgent(null);
    setStepIndex(-1);
    setSupervisorRetries(0);
    setAgentStats(() => {
      const init: Record<string, AgentStats> = {};
      AGENTS.forEach((a) => { init[a.id] = { sent: 0, received: 0, retries: 0 }; });
      return init as Record<AgentId, AgentStats>;
    });
    spawnedRef.current = new Set();
  }, []);

  if (!mounted) return null;

  /* ---- Render helpers ---- */
  const svgW = 600;
  const svgH = 330;

  function renderEdge(from: AgentNode, to: AgentNode, key: string) {
    return (
      <line
        key={key}
        x1={from.cx} y1={from.cy}
        x2={to.cx} y2={to.cy}
        stroke="rgba(255,255,255,0.06)"
        strokeWidth={1.5}
        strokeDasharray="4 4"
      />
    );
  }

  const edges: [AgentId, AgentId][] = [
    ["planner", "worker1"], ["planner", "worker2"], ["planner", "critic"],
    ["worker1", "critic"], ["worker2", "critic"],
    ["worker1", "judge"], ["worker2", "judge"],
    ["critic", "worker1"], ["critic", "worker2"],
    ["judge", "synthesizer"],
    ["synthesizer", "planner"],
  ];

  function heartbeatOpacity(agentId: AgentId): number {
    const last = heartbeats[agentId];
    if (!last) return 0;
    const age = Date.now() - last;
    return Math.max(0, 1 - age / 1800);
  }

  const selected = selectedAgent ? AGENT_MAP[selectedAgent] : null;

  return (
    <div className="widget-container ch5">
      <div className="widget-label">Interactive · Agent Communication Protocol</div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn-mono" onClick={handleSendTask} disabled={running}>
          {running ? "Running..." : "Send Task"}
        </button>
        <button className="btn-mono" onClick={handleReset}>Reset</button>
        <span style={{ fontSize: "0.75rem", color: "#a1a1aa", marginLeft: 8 }}>
          Phase: <span style={{ color: phase === "complete" ? "#4ade80" : "#f9a8d4" }}>{phase}</span>
        </span>
      </div>

      {/* Progress stepper */}
      {stepIndex >= 0 && (
        <div style={{
          display: "flex", gap: 2, marginBottom: 10, alignItems: "center",
        }}>
          {SEQUENCE.map((step, i) => {
            const done = i <= stepIndex;
            const active = i === stepIndex && running;
            return (
              <div key={i} style={{ display: "flex", alignItems: "center" }}>
                <div
                  title={step.label}
                  style={{
                    width: 10, height: 10, borderRadius: "50%",
                    background: done ? MSG_COLORS[step.kind] : "rgba(255,255,255,0.06)",
                    border: active ? `2px solid ${MSG_COLORS[step.kind]}` : "1px solid rgba(255,255,255,0.1)",
                    transition: "background 0.3s",
                  }}
                />
                {i < SEQUENCE.length - 1 && (
                  <div style={{
                    width: 16, height: 1,
                    background: done ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.04)",
                  }} />
                )}
              </div>
            );
          })}
          <span style={{ marginLeft: 8, fontSize: "0.62rem", fontFamily: "monospace", color: "#52525b" }}>
            {stepIndex + 1}/{SEQUENCE.length} steps
          </span>
        </div>
      )}

      {/* Supervisor retry bar */}
      {supervisorRetries > 0 && (
        <div style={{
          display: "flex", gap: 8, alignItems: "center", marginBottom: 10,
          padding: "4px 10px", borderRadius: 6,
          background: "rgba(244,114,182,0.06)",
          border: "1px solid rgba(244,114,182,0.15)",
          fontSize: "0.68rem", fontFamily: "monospace", color: "#f472b6",
        }}>
          <span>Supervisor</span>
          <span style={{ color: "#a1a1aa" }}>|</span>
          <span>Retries triggered: {supervisorRetries}</span>
          <span style={{ color: "#a1a1aa" }}>|</span>
          <span style={{ color: "#a1a1aa" }}>Backoff: exponential (base 200ms)</span>
        </div>
      )}

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        {/* SVG Diagram */}
        <div style={{ flex: "1 1 380px", minWidth: 0 }}>
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            style={{ width: "100%", height: "auto", maxHeight: 340, background: "rgba(0,0,0,0.25)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.06)" }}
          >
            {/* Edges */}
            {edges.map(([fId, tId]) => renderEdge(AGENT_MAP[fId], AGENT_MAP[tId], `${fId}-${tId}`))}

            {/* Supervisor label */}
            <text x={svgW / 2} y={svgH - 6} textAnchor="middle" fill="#a1a1aa" fontSize={10} fontFamily="monospace">
              Supervisor / Heartbeat Monitor
            </text>

            {/* Agent nodes */}
            {AGENTS.map((agent) => {
              const isSelected = selectedAgent === agent.id;
              const hbOp = heartbeatOpacity(agent.id);
              return (
                <g key={agent.id} style={{ cursor: "pointer" }} onClick={() => setSelectedAgent(isSelected ? null : agent.id)}>
                  {/* Heartbeat ring */}
                  <circle cx={agent.cx} cy={agent.cy} r={30} fill="none" stroke={agent.color} strokeWidth={2} opacity={hbOp * 0.5} />
                  {/* Node bg */}
                  <circle
                    cx={agent.cx} cy={agent.cy} r={24}
                    fill={isSelected ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.04)"}
                    stroke={agent.color}
                    strokeWidth={isSelected ? 2 : 1}
                  />
                  {/* Short label */}
                  <text x={agent.cx} y={agent.cy - 4} textAnchor="middle" fill={agent.color} fontSize={13} fontWeight="bold" fontFamily="monospace">
                    {agent.short}
                  </text>
                  <text x={agent.cx} y={agent.cy + 10} textAnchor="middle" fill="#a1a1aa" fontSize={8.5} fontFamily="monospace">
                    {agent.label}
                  </text>
                  {/* Heartbeat dot */}
                  <circle cx={agent.cx + 20} cy={agent.cy - 20} r={3} fill={hbOp > 0.3 ? "#4ade80" : "#52525b"}>
                    {hbOp > 0.3 && (
                      <animate attributeName="r" values="3;5;3" dur="1.2s" repeatCount="indefinite" />
                    )}
                  </circle>
                </g>
              );
            })}

            {/* Flying messages */}
            {messages.map((m) => {
              const from = AGENT_MAP[m.from];
              const to = AGENT_MAP[m.to];
              const x = lerp(from.cx, to.cx, m.progress);
              const y = lerp(from.cy, to.cy, m.progress);
              const col = MSG_COLORS[m.kind];
              return (
                <g key={m.id}>
                  <circle cx={x} cy={y} r={5} fill={col} opacity={0.9}>
                    <animate attributeName="r" values="4;6;4" dur="0.6s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={x} cy={y} r={10} fill="none" stroke={col} strokeWidth={1} opacity={0.3}>
                    <animate attributeName="r" values="6;14;6" dur="0.8s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0;0.4" dur="0.8s" repeatCount="indefinite" />
                  </circle>
                  {/* Message type label */}
                  <text x={x} y={y - 10} textAnchor="middle" fill={col} fontSize={8} fontFamily="monospace" opacity={0.85}>
                    {m.kind}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div style={{ display: "flex", gap: 14, marginTop: 8, flexWrap: "wrap" }}>
            {(Object.entries(MSG_COLORS) as [MsgKind, string][]).map(([kind, col]) => (
              <span key={kind} style={{ fontSize: "0.68rem", fontFamily: "monospace", color: col, display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: col }} />
                {kind}
              </span>
            ))}
          </div>
        </div>

        {/* Right panel: agent detail + message log */}
        <div style={{ flex: "0 0 220px", display: "flex", flexDirection: "column", gap: 10, minWidth: 200 }}>
          {/* Agent detail card */}
          <div style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 8,
            padding: "10px 12px",
            minHeight: 90,
          }}>
            {selected ? (() => {
              const stats = agentStats[selected.id];
              return (
                <>
                  <div style={{ fontSize: "0.82rem", fontWeight: 600, color: selected.color, fontFamily: "monospace", marginBottom: 4 }}>
                    {selected.label}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "#a1a1aa", lineHeight: 1.5 }}>
                    {selected.role}
                  </div>
                  <div style={{ marginTop: 6, fontSize: "0.68rem", color: "#52525b", fontFamily: "monospace" }}>
                    Status: {running ? "active" : "idle"} | Heartbeat: OK
                  </div>
                  <div style={{ marginTop: 4, display: "flex", gap: 10, fontSize: "0.65rem", fontFamily: "monospace" }}>
                    <span style={{ color: "#f97316" }}>Sent: {stats.sent}</span>
                    <span style={{ color: "#4ade80" }}>Recv: {stats.received}</span>
                    {stats.retries > 0 && (
                      <span style={{ color: "#f472b6" }}>Retries: {stats.retries}</span>
                    )}
                  </div>
                </>
              );
            })() : (
              <div style={{ fontSize: "0.72rem", color: "#52525b", fontFamily: "monospace" }}>
                Click an agent node to inspect its role and status.
              </div>
            )}
          </div>

          {/* Message log */}
          <div style={{
            background: "rgba(0,0,0,0.25)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 8,
            padding: "8px 10px",
            flex: 1,
            minHeight: 140,
            maxHeight: 220,
            overflowY: "auto",
            fontFamily: "monospace",
            fontSize: "0.67rem",
          }} ref={logRef}>
            <div style={{ color: "#52525b", marginBottom: 4, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Message Log
            </div>
            {log.length === 0 ? (
              <div style={{ color: "#3f3f46" }}>No messages yet. Press Send Task.</div>
            ) : (
              log.map((entry, i) => (
                <div key={i} style={{ color: entry.color, marginBottom: 2, lineHeight: 1.5 }}>
                  <span style={{ color: "#52525b" }}>{formatTs(entry.ts)}</span>{" "}
                  {entry.text}
                </div>
              ))
            )}
            {phase === "complete" && (
              <div style={{ color: "#4ade80", marginTop: 6, borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 6 }}>
                Orchestration complete. Final answer delivered.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Module summary strip */}
      <div style={{ marginTop: 14, display: "flex", gap: 6, flexWrap: "wrap" }}>
        {[
          { name: "Registry",  desc: "Agent discovery & registration" },
          { name: "Msg Bus",   desc: "Pub/sub messaging backbone" },
          { name: "Roles",     desc: "Planner, Worker, Critic, Judge, Synthesizer" },
          { name: "Supervisor", desc: "OTP-style retry with backoff" },
          { name: "Handoff",   desc: "Capability-based task delegation" },
          { name: "Org Chart", desc: "Hierarchical team structure" },
          { name: "Heartbeat", desc: "Health monitoring pulse" },
          { name: "Routing",   desc: "Content-based message routing" },
          { name: "Backpressure", desc: "Flow control & rate limiting" },
          { name: "Schema",    desc: "Typed message contracts" },
          { name: "Auth",      desc: "Agent identity & capability tokens" },
          { name: "Consensus", desc: "Multi-agent agreement protocol" },
          { name: "Replay",    desc: "Deterministic message replay" },
          { name: "Telemetry", desc: "Distributed tracing & metrics" },
          { name: "Sandbox",   desc: "Isolated execution environments" },
        ].map((mod) => (
          <span
            key={mod.name}
            title={mod.desc}
            style={{
              fontSize: "0.62rem",
              fontFamily: "monospace",
              padding: "2px 7px",
              borderRadius: 4,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.06)",
              color: "#71717a",
              cursor: "default",
              whiteSpace: "nowrap",
            }}
          >
            {mod.name}
          </span>
        ))}
      </div>
    </div>
  );
}
