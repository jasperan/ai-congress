"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── Palette ──────────────────────────────────────────────────────────────────
const C = {
  bg: "#12131a",
  text: "#e4e4e7",
  muted: "#a1a1aa",
  border: "rgba(255,255,255,0.08)",
  accent: "#6366f1",
  green: "#22c55e",
  amber: "#f59e0b",
  rose: "#f43f5e",
  cyan: "#06b6d4",
  purple: "#a855f7",
  sky: "#38bdf8",
} as const;

// ── Seeded random ────────────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ── Types ────────────────────────────────────────────────────────────────────
type Mode = "multi-model" | "multi-request" | "hybrid" | "personality" | "streaming";

interface ModeInfo {
  key: Mode;
  label: string;
  short: string;
}

const MODES: ModeInfo[] = [
  { key: "multi-model", label: "Multi-Model", short: "Different LLMs vote on one query" },
  { key: "multi-request", label: "Multi-Request", short: "Same model, varied temperatures" },
  { key: "hybrid", label: "Hybrid", short: "Top models selected by weight" },
  { key: "personality", label: "Personality", short: "Custom personas debate & vote" },
  { key: "streaming", label: "Streaming", short: "Real-time streaming with live votes" },
];

// ── Model data ───────────────────────────────────────────────────────────────
const MODELS = [
  { name: "phi3", color: "#38bdf8" },
  { name: "mistral", color: "#f59e0b" },
  { name: "llama3", color: "#22c55e" },
  { name: "qwen", color: "#a855f7" },
  { name: "gemma", color: "#f43f5e" },
];

const TEMPS = [0.3, 0.5, 0.7, 1.0];

const PERSONAS = [
  { name: "Senator", trait: "Policy-driven", color: "#6366f1" },
  { name: "Scientist", trait: "Data-focused", color: "#06b6d4" },
  { name: "Actor", trait: "Persuasive", color: "#f59e0b" },
  { name: "Engineer", trait: "Pragmatic", color: "#22c55e" },
];

const HYBRID_WEIGHTS = [0.92, 0.85, 0.78, 0.55, 0.41];

// ── Component ────────────────────────────────────────────────────────────────
export default function SwarmModesWidget() {
  const [mounted, setMounted] = useState(false);
  const [mode, setMode] = useState<Mode>("multi-model");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1
  const [showResult, setShowResult] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef(0);

  useEffect(() => setMounted(true), []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const runQuery = useCallback(() => {
    if (running) return;
    setRunning(true);
    setShowResult(false);
    setProgress(0);
    startRef.current = performance.now();

    const duration = 2200;
    const tick = () => {
      const elapsed = performance.now() - startRef.current;
      const p = Math.min(elapsed / duration, 1);
      setProgress(p);
      if (p < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setRunning(false);
        setShowResult(true);
        timerRef.current = setTimeout(() => setShowResult(false), 4000);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [running]);

  const handleModeChange = useCallback((m: Mode) => {
    if (running) return;
    setMode(m);
    setShowResult(false);
    setProgress(0);
  }, [running]);

  // ── SVG helpers ──────────────────────────────────────────────────────────
  const svgW = 560;
  const svgH = 280;

  const rng = seededRandom(42);

  function drawArrow(
    x1: number, y1: number, x2: number, y2: number,
    color: string, animDelay: number, lit: boolean
  ) {
    const opacity = lit ? Math.min(progress * 3 - animDelay, 1) : 0.15;
    const clampedOpacity = Math.max(0, Math.min(1, opacity));
    return (
      <line
        key={`${x1}-${y1}-${x2}-${y2}`}
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color}
        strokeWidth={lit && clampedOpacity > 0.5 ? 2 : 1}
        strokeOpacity={running ? clampedOpacity : 0.25}
        strokeDasharray={lit && running ? "6 3" : "none"}
      />
    );
  }

  function drawNode(
    cx: number, cy: number, label: string, color: string,
    dimmed?: boolean, sub?: string
  ) {
    return (
      <g key={`node-${label}-${cx}-${cy}`}>
        <rect
          x={cx - 40} y={cy - 18} width={80} height={36} rx={8}
          fill={dimmed ? "rgba(255,255,255,0.03)" : `${color}18`}
          stroke={dimmed ? "rgba(255,255,255,0.06)" : color}
          strokeWidth={dimmed ? 0.5 : 1}
        />
        <text
          x={cx} y={sub ? cy - 3 : cy + 1} textAnchor="middle"
          dominantBaseline="middle"
          fill={dimmed ? C.muted : C.text} fontSize={11} fontFamily="monospace"
        >
          {label}
        </text>
        {sub && (
          <text
            x={cx} y={cy + 11} textAnchor="middle"
            dominantBaseline="middle"
            fill={C.muted} fontSize={9} fontFamily="monospace"
          >
            {sub}
          </text>
        )}
      </g>
    );
  }

  function drawQueryNode() {
    return (
      <g>
        <rect
          x={10} y={svgH / 2 - 22} width={90} height={44} rx={10}
          fill={`${C.accent}20`} stroke={C.accent} strokeWidth={1.5}
        />
        <text
          x={55} y={svgH / 2 - 4} textAnchor="middle" dominantBaseline="middle"
          fill={C.text} fontSize={11} fontWeight="bold" fontFamily="monospace"
        >
          Query
        </text>
        <text
          x={55} y={svgH / 2 + 10} textAnchor="middle" dominantBaseline="middle"
          fill={C.muted} fontSize={8} fontFamily="monospace"
        >
          &quot;Best framework?&quot;
        </text>
      </g>
    );
  }

  function drawVoteNode(x: number) {
    return (
      <g>
        <rect
          x={x - 35} y={svgH / 2 - 18} width={70} height={36} rx={8}
          fill={`${C.amber}18`} stroke={C.amber} strokeWidth={1}
        />
        <text
          x={x} y={svgH / 2 + 1} textAnchor="middle" dominantBaseline="middle"
          fill={C.text} fontSize={11} fontWeight="bold" fontFamily="monospace"
        >
          Vote
        </text>
      </g>
    );
  }

  function drawResultNode(x: number) {
    const glowing = showResult;
    return (
      <g>
        <rect
          x={x - 40} y={svgH / 2 - 18} width={80} height={36} rx={8}
          fill={glowing ? `${C.green}30` : `${C.green}10`}
          stroke={C.green}
          strokeWidth={glowing ? 2 : 1}
        />
        {glowing && (
          <rect
            x={x - 40} y={svgH / 2 - 18} width={80} height={36} rx={8}
            fill="none" stroke={C.green} strokeWidth={1}
            strokeOpacity={0.4}
            style={{ filter: "blur(4px)" }}
          />
        )}
        <text
          x={x} y={svgH / 2 + 1} textAnchor="middle" dominantBaseline="middle"
          fill={C.green} fontSize={11} fontWeight="bold" fontFamily="monospace"
        >
          {showResult ? "Winner!" : "Result"}
        </text>
      </g>
    );
  }

  // ── Streaming dots ─────────────────────────────────────────────────────
  function drawStreamingDots(
    x1: number, y1: number, x2: number, y2: number,
    color: string, offset: number
  ) {
    if (!running) return null;
    const dots = [];
    for (let i = 0; i < 4; i++) {
      const t = ((progress * 3 + offset + i * 0.15) % 1);
      const cx = x1 + (x2 - x1) * t;
      const cy = y1 + (y2 - y1) * t;
      dots.push(
        <circle
          key={`dot-${offset}-${i}`}
          cx={cx} cy={cy} r={2.5}
          fill={color} opacity={0.4 + t * 0.6}
        />
      );
    }
    return <>{dots}</>;
  }

  // ── Diagram renderers ──────────────────────────────────────────────────
  function renderMultiModel() {
    const modelX = 200;
    const voteX = 370;
    const resultX = 490;
    const spacing = svgH / (MODELS.length + 1);

    return (
      <g className="animate-slide-in">
        {drawQueryNode()}
        {MODELS.map((m, i) => {
          const my = spacing * (i + 1);
          return (
            <React.Fragment key={m.name}>
              {drawArrow(100, svgH / 2, modelX - 40, my, m.color, i * 0.15, running)}
              {drawNode(modelX, my, m.name, m.color)}
              {drawArrow(modelX + 40, my, voteX - 35, svgH / 2, C.amber, 0.5 + i * 0.1, running)}
              {running && drawStreamingDots(modelX + 40, my, voteX - 35, svgH / 2, m.color, i * 0.2)}
            </React.Fragment>
          );
        })}
        {drawVoteNode(voteX)}
        {drawArrow(voteX + 35, svgH / 2, resultX - 40, svgH / 2, C.green, 0.8, running)}
        {drawResultNode(resultX)}
      </g>
    );
  }

  function renderMultiRequest() {
    const modelX = 200;
    const voteX = 370;
    const resultX = 490;
    const spacing = svgH / (TEMPS.length + 1);

    return (
      <g className="animate-slide-in">
        {drawQueryNode()}
        {TEMPS.map((t, i) => {
          const my = spacing * (i + 1);
          return (
            <React.Fragment key={t}>
              {drawArrow(100, svgH / 2, modelX - 40, my, C.sky, i * 0.15, running)}
              {drawNode(modelX, my, "llama3", C.sky, false, `temp=${t}`)}
              {drawArrow(modelX + 40, my, voteX - 35, svgH / 2, C.amber, 0.5 + i * 0.1, running)}
              {running && drawStreamingDots(modelX + 40, my, voteX - 35, svgH / 2, C.sky, i * 0.25)}
            </React.Fragment>
          );
        })}
        {drawVoteNode(voteX)}
        {drawArrow(voteX + 35, svgH / 2, resultX - 40, svgH / 2, C.green, 0.8, running)}
        {drawResultNode(resultX)}
      </g>
    );
  }

  function renderHybrid() {
    const modelX = 200;
    const voteX = 370;
    const resultX = 490;
    const spacing = svgH / (MODELS.length + 1);

    return (
      <g className="animate-slide-in">
        {drawQueryNode()}
        {MODELS.map((m, i) => {
          const my = spacing * (i + 1);
          const weight = HYBRID_WEIGHTS[i];
          const selected = i < 3;
          return (
            <React.Fragment key={m.name}>
              {drawArrow(100, svgH / 2, modelX - 40, my, m.color, i * 0.15, running && selected)}
              {drawNode(modelX, my, m.name, m.color, !selected)}
              {/* Weight bar */}
              <rect
                x={modelX + 46} y={my - 5} width={40} height={10} rx={3}
                fill="rgba(255,255,255,0.05)" stroke={C.border}
              />
              <rect
                x={modelX + 46} y={my - 5}
                width={40 * weight} height={10} rx={3}
                fill={selected ? m.color : "rgba(255,255,255,0.08)"}
                opacity={selected ? 0.7 : 0.2}
              />
              <text
                x={modelX + 92} y={my + 1} dominantBaseline="middle"
                fill={selected ? C.text : C.muted} fontSize={8} fontFamily="monospace"
              >
                {(weight * 100).toFixed(0)}%
              </text>
              {selected && (
                <>
                  {drawArrow(modelX + 110, my, voteX - 35, svgH / 2, C.amber, 0.5 + i * 0.1, running)}
                  {running && drawStreamingDots(modelX + 110, my, voteX - 35, svgH / 2, m.color, i * 0.2)}
                </>
              )}
              {/* Selection glow */}
              {selected && (
                <circle
                  cx={modelX - 52} cy={my} r={4}
                  fill={C.green} opacity={running ? 0.8 : 0.5}
                />
              )}
            </React.Fragment>
          );
        })}
        {drawVoteNode(voteX)}
        {drawArrow(voteX + 35, svgH / 2, resultX - 40, svgH / 2, C.green, 0.8, running)}
        {drawResultNode(resultX)}
      </g>
    );
  }

  function renderPersonality() {
    const agentX = 200;
    const voteX = 380;
    const resultX = 500;
    const spacing = svgH / (PERSONAS.length + 1);

    return (
      <g className="animate-slide-in">
        {drawQueryNode()}
        {PERSONAS.map((p, i) => {
          const my = spacing * (i + 1);
          return (
            <React.Fragment key={p.name}>
              {drawArrow(100, svgH / 2, agentX - 48, my, p.color, i * 0.15, running)}
              {/* Avatar circle */}
              <circle
                cx={agentX - 56} cy={my} r={12}
                fill={`${p.color}25`} stroke={p.color} strokeWidth={1}
              />
              <text
                x={agentX - 56} y={my + 1} textAnchor="middle" dominantBaseline="middle"
                fill={p.color} fontSize={10} fontFamily="monospace" fontWeight="bold"
              >
                {p.name[0]}
              </text>
              {/* Name + trait node */}
              <rect
                x={agentX - 40} y={my - 18} width={95} height={36} rx={8}
                fill={`${p.color}12`} stroke={p.color} strokeWidth={1}
              />
              <text
                x={agentX + 8} y={my - 4} textAnchor="middle" dominantBaseline="middle"
                fill={C.text} fontSize={10} fontFamily="monospace"
              >
                {p.name}
              </text>
              <text
                x={agentX + 8} y={my + 10} textAnchor="middle" dominantBaseline="middle"
                fill={C.muted} fontSize={8} fontFamily="monospace"
              >
                {p.trait}
              </text>
              {drawArrow(agentX + 55, my, voteX - 35, svgH / 2, C.amber, 0.5 + i * 0.1, running)}
              {running && drawStreamingDots(agentX + 55, my, voteX - 35, svgH / 2, p.color, i * 0.2)}
            </React.Fragment>
          );
        })}
        {drawVoteNode(voteX)}
        {drawArrow(voteX + 35, svgH / 2, resultX - 40, svgH / 2, C.green, 0.8, running)}
        {drawResultNode(resultX)}
      </g>
    );
  }

  function renderStreaming() {
    const modelX = 190;
    const streamX = 340;
    const voteX = 440;
    const resultX = 520;
    const models = MODELS.slice(0, 4);
    const spacing = svgH / (models.length + 1);

    // Streaming-specific animated dots along horizontal "stream bars"
    function streamBar(y: number, color: string, idx: number) {
      const barW = 80;
      const dots = [];
      if (running) {
        for (let d = 0; d < 6; d++) {
          const seed = rng();
          const t = ((progress * 4 + seed + d * 0.12) % 1);
          dots.push(
            <circle
              key={`sb-${idx}-${d}`}
              cx={streamX - barW / 2 + barW * t}
              cy={y}
              r={2} fill={color}
              opacity={0.3 + t * 0.7}
            />
          );
        }
      }
      return (
        <g key={`stream-${idx}`}>
          <rect
            x={streamX - barW / 2} y={y - 4} width={barW} height={8} rx={4}
            fill="rgba(255,255,255,0.04)" stroke={C.border}
          />
          {/* Progress fill */}
          {running && (
            <rect
              x={streamX - barW / 2} y={y - 4}
              width={barW * Math.min(progress * 1.3, 1)} height={8} rx={4}
              fill={`${color}30`}
            />
          )}
          {dots}
        </g>
      );
    }

    return (
      <g className="animate-slide-in">
        {drawQueryNode()}
        {models.map((m, i) => {
          const my = spacing * (i + 1);
          return (
            <React.Fragment key={m.name}>
              {drawArrow(100, svgH / 2, modelX - 40, my, m.color, i * 0.1, running)}
              {drawNode(modelX, my, m.name, m.color)}
              {drawArrow(modelX + 40, my, streamX - 40, my, m.color, 0.3 + i * 0.1, running)}
              {streamBar(my, m.color, i)}
              {drawArrow(streamX + 40, my, voteX - 35, svgH / 2, C.amber, 0.6 + i * 0.08, running)}
            </React.Fragment>
          );
        })}
        {/* Live vote breakdown */}
        {running && progress > 0.4 && (
          <g>
            <text
              x={voteX} y={svgH / 2 - 30} textAnchor="middle"
              fill={C.muted} fontSize={8} fontFamily="monospace"
            >
              Live Tally
            </text>
            {models.map((m, i) => {
              const voteVal = Math.round(((i === 0 ? 0.38 : i === 1 ? 0.28 : i === 2 ? 0.22 : 0.12)) * Math.min((progress - 0.4) / 0.5, 1) * 100);
              return (
                <text
                  key={`vote-${m.name}`}
                  x={voteX + 42} y={svgH / 2 - 16 + i * 13} textAnchor="start"
                  fill={m.color} fontSize={8} fontFamily="monospace"
                  opacity={Math.min((progress - 0.4) * 3, 1)}
                >
                  {m.name}: {voteVal}%
                </text>
              );
            })}
          </g>
        )}
        {drawVoteNode(voteX)}
        {drawArrow(voteX + 35, svgH / 2, resultX - 40, svgH / 2, C.green, 0.85, running)}
        {drawResultNode(resultX)}
      </g>
    );
  }

  const diagramMap: Record<Mode, () => React.ReactNode> = {
    "multi-model": renderMultiModel,
    "multi-request": renderMultiRequest,
    "hybrid": renderHybrid,
    "personality": renderPersonality,
    "streaming": renderStreaming,
  };

  // ── Result text ────────────────────────────────────────────────────────
  const resultTexts: Record<Mode, string> = {
    "multi-model": "phi3, mistral, llama3, qwen, gemma voted -- llama3 wins (3/5 votes)",
    "multi-request": "4 runs at varied temps -- temp=0.7 consensus wins",
    "hybrid": "Top 3 by weight (phi3 92%, mistral 85%, llama3 78%) -- phi3 wins",
    "personality": "Senator, Scientist, Actor, Engineer debated -- Scientist wins",
    "streaming": "Real-time tally complete -- phi3 leads with 38% confidence",
  };

  // ── Render ─────────────────────────────────────────────────────────────
  const activeMode = MODES.find((m) => m.key === mode)!;

  return (
    <div className="widget-container ch1">
      <div className="widget-label">Interactive &middot; Swarm Modes</div>

      {/* Mode buttons */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
        {MODES.map((m) => (
          <button
            key={m.key}
            className={`btn-mono${mode === m.key ? " active" : ""}`}
            onClick={() => handleModeChange(m.key)}
            style={{ fontSize: 12, padding: "5px 12px", cursor: running ? "not-allowed" : "pointer" }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Mode description */}
      <div
        className="animate-slide-in"
        key={mode}
        style={{ color: C.muted, fontSize: 13, marginBottom: 12, fontFamily: "monospace" }}
      >
        {activeMode.short}
      </div>

      {/* SVG Diagram */}
      <div style={{ minHeight: 320 }}>
        {mounted && (
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            width="100%"
            style={{
              maxWidth: svgW,
              height: "auto",
              background: `linear-gradient(135deg, ${C.bg}, #1a1b26)`,
              borderRadius: 10,
              border: `1px solid ${C.border}`,
            }}
          >
            {/* Grid pattern for depth */}
            <defs>
              <pattern id="sw-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.02)" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width={svgW} height={svgH} fill="url(#sw-grid)" />
            {diagramMap[mode]()}
          </svg>
        )}

        {/* Progress bar */}
        {running && (
          <div
            style={{
              height: 3,
              background: "rgba(255,255,255,0.05)",
              borderRadius: 2,
              marginTop: 8,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${progress * 100}%`,
                background: `linear-gradient(90deg, ${C.accent}, ${C.green})`,
                borderRadius: 2,
                transition: "width 0.05s linear",
              }}
            />
          </div>
        )}

        {/* Result card */}
        {showResult && (
          <div
            className="animate-slide-in"
            style={{
              marginTop: 12,
              padding: "10px 14px",
              background: `${C.green}10`,
              border: `1px solid ${C.green}40`,
              borderRadius: 8,
              fontSize: 12,
              fontFamily: "monospace",
              color: C.green,
            }}
          >
            {resultTexts[mode]}
          </div>
        )}
      </div>

      {/* Run Query button */}
      <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 12 }}>
        <button
          className={`btn-mono${running ? " active" : ""}`}
          onClick={runQuery}
          disabled={running}
          style={{
            fontSize: 13,
            padding: "7px 20px",
            cursor: running ? "not-allowed" : "pointer",
            opacity: running ? 0.6 : 1,
          }}
        >
          {running ? "Running..." : "Run Query"}
        </button>
        {running && (
          <span style={{ color: C.muted, fontSize: 11, fontFamily: "monospace" }}>
            {Math.round(progress * 100)}% complete
          </span>
        )}
      </div>
    </div>
  );
}
