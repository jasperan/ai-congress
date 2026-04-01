"use client";

import { useState, useEffect, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Seeded PRNG (xorshift32) -- deterministic simulations              */
/*  Avoids Math.random() so SSR and CSR produce consistent output.     */
/* ------------------------------------------------------------------ */
function createRng(seed: number) {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ */
/*  Constants & types                                                  */
/* ------------------------------------------------------------------ */
const ALPHA = 0.3;           // EMA smoothing factor
const K = 32;                // ELO K-factor
const MAX_ROUNDS = 10;       // total simulation rounds
const COLORS = ["#38bdf8", "#f472b6", "#4ade80", "#facc15", "#a78bfa"];
const mono = "var(--font-mono), monospace";

interface Model {
  name: string;
  elo: number;
  weight: number;
  wins: number;
  rounds: number;
  trend: "up" | "down" | "stable";
}

interface HistoryPoint {
  round: number;
  elos: number[];
}

const INIT: Model[] = [
  { name: "GPT-4o",        elo: 1280, weight: 0.72, wins: 0, rounds: 0, trend: "stable" },
  { name: "Claude-3.5",    elo: 1310, weight: 0.76, wins: 0, rounds: 0, trend: "stable" },
  { name: "Gemini-Pro",    elo: 1200, weight: 0.60, wins: 0, rounds: 0, trend: "stable" },
  { name: "Llama-3-70B",   elo: 1150, weight: 0.50, wins: 0, rounds: 0, trend: "stable" },
  { name: "Mistral-Large", elo: 1230, weight: 0.64, wins: 0, rounds: 0, trend: "stable" },
];

/* ------------------------------------------------------------------ */
/*  ELO math                                                           */
/*  Standard chess-style expected score + K-factor rating update.      */
/* ------------------------------------------------------------------ */
function expScore(a: number, b: number) {
  return 1 / (1 + Math.pow(10, (b - a) / 400));
}

function eloUpdate(w: number, l: number) {
  const e = expScore(w, l);
  return { w: w + K * (1 - e), l: l + K * (0 - (1 - e)) };
}

/* ------------------------------------------------------------------ */
/*  Confidence calibration data (static)                               */
/*  Compares predicted win probability vs actual observed accuracy.     */
/* ------------------------------------------------------------------ */
const CAL = [
  { ex: 0.1, ac: 0.12 }, { ex: 0.2, ac: 0.18 }, { ex: 0.3, ac: 0.27 },
  { ex: 0.4, ac: 0.42 }, { ex: 0.5, ac: 0.48 }, { ex: 0.6, ac: 0.63 },
  { ex: 0.7, ac: 0.65 }, { ex: 0.8, ac: 0.82 }, { ex: 0.9, ac: 0.88 },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function ELORatingWidget() {
  const [mounted, setMounted] = useState(false);
  const [models, setModels] = useState<Model[]>(INIT);
  const [history, setHistory] = useState<HistoryPoint[]>([
    { round: 0, elos: INIT.map((m) => m.elo) },
  ]);
  const [roundNum, setRoundNum] = useState(0);
  const [simulating, setSimulating] = useState(false);
  const [activeModel, setActiveModel] = useState<number | null>(null);
  const [winnerIdx, setWinnerIdx] = useState<number | null>(null);
  const [flash, setFlash] = useState<{ idx: number; type: "up" | "down" } | null>(null);
  const rng = useRef(createRng(42));

  useEffect(() => { setMounted(true); }, []);

  /* Simulate one round */
  const simulateRound = useCallback(() => {
    if (simulating || roundNum >= MAX_ROUNDS) return;
    setSimulating(true);
    setWinnerIdx(null);
    const rand = rng.current;
    let step = 0;

    // Phase 1: cycle highlight across model cards
    const iv = setInterval(() => {
      setActiveModel(step % 5);
      step++;
      if (step >= 8) {
        clearInterval(iv);
        // Phase 2: pick winner weighted by model weight + noise
        const scores = models.map((m) => m.weight + rand() * 0.3);
        const winner = scores.indexOf(Math.max(...scores));
        setWinnerIdx(winner);
        setActiveModel(null);

        // Phase 3: update ratings + weights
        setTimeout(() => {
          setModels((prev) => {
            const next = prev.map((m) => ({ ...m }));
            const prevElos = next.map((m) => m.elo);
            // Winner beats every other model (round-robin)
            for (let i = 0; i < next.length; i++) {
              if (i === winner) continue;
              const r = eloUpdate(next[winner].elo, next[i].elo);
              next[winner].elo = r.w;
              next[i].elo = r.l;
            }
            // Update stats, EMA weights, trends
            for (let i = 0; i < next.length; i++) {
              next[i].rounds += 1;
              if (i === winner) next[i].wins += 1;
              const didWin = i === winner ? 1 : 0;
              next[i].weight = Math.min(1, Math.max(0,
                ALPHA * didWin + (1 - ALPHA) * next[i].weight));
              if (next[i].elo > prevElos[i] + 5) next[i].trend = "up";
              else if (next[i].elo < prevElos[i] - 5) next[i].trend = "down";
              else next[i].trend = "stable";
              next[i].elo = Math.round(next[i].elo);
            }
            return next;
          });
          setRoundNum((r) => r + 1);
          setSimulating(false);
          setTimeout(() => setWinnerIdx(null), 1200);
        }, 400);
      }
    }, 120);
  }, [simulating, roundNum, models]);

  /* Sync history after each round */
  useEffect(() => {
    if (roundNum === 0) return;
    setHistory((prev) => {
      if (prev.length > roundNum) return prev;
      return [...prev, { round: roundNum, elos: models.map((m) => m.elo) }];
    });
  }, [roundNum, models]);

  /* User feedback (thumbs up / down) */
  const handleFeedback = useCallback((idx: number, type: "up" | "down") => {
    setFlash({ idx, type });
    setTimeout(() => setFlash(null), 600);
    setModels((prev) => prev.map((m, i) => {
      if (i !== idx) return { ...m };
      const d = type === "up" ? 0.08 : -0.08;
      return { ...m, weight: Math.min(1, Math.max(0, m.weight + d)) };
    }));
  }, []);

  /* Reset */
  const handleReset = useCallback(() => {
    setModels(INIT.map((m) => ({ ...m })));
    setHistory([{ round: 0, elos: INIT.map((m) => m.elo) }]);
    setRoundNum(0);
    setSimulating(false);
    setActiveModel(null);
    setWinnerIdx(null);
    rng.current = createRng(42);
  }, []);

  if (!mounted) return null;

  /* ELO line-chart layout */
  const cW = 520, cH = 180, pL = 44, pR = 12, pT = 12, pB = 28;
  const plotW = cW - pL - pR, plotH = cH - pT - pB;
  const allElos = history.flatMap((h) => h.elos);
  const minE = Math.min(...allElos) - 30;
  const maxE = Math.max(...allElos) + 30;
  const eloRange = maxE - minE || 1;
  const toX = (r: number) => pL + (r / Math.max(MAX_ROUNDS, roundNum || 1)) * plotW;
  const toY = (e: number) => pT + plotH - ((e - minE) / eloRange) * plotH;

  /* Calibration mini-chart layout */
  const calW = 180, calH = 140, calP = 28;
  const calPW = calW - calP * 2, calPH = calH - calP * 2;

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */
  return (
    <div className="widget-container ch8">
      <div className="widget-label">Interactive &middot; Learning &amp; Adaptation</div>

      {/* Header + controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 600, color: "#e4e4e7" }}>
          ELO Rating &amp; Dynamic Weights
        </h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn-mono" onClick={simulateRound}
            disabled={simulating || roundNum >= MAX_ROUNDS}
            style={{ opacity: simulating || roundNum >= MAX_ROUNDS ? 0.4 : 1 }}>
            {simulating ? "Simulating..." : roundNum >= MAX_ROUNDS ? "Complete" : `Simulate Round ${roundNum + 1}`}
          </button>
          <button className="btn-mono" onClick={handleReset}>Reset</button>
        </div>
      </div>

      {/* Model cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 20 }}>
        {models.map((m, i) => {
          const isActive = activeModel === i;
          const isWinner = winnerIdx === i;
          const fb = flash?.idx === i ? flash.type : null;
          return (
            <div key={m.name} style={{
              background: isWinner ? "rgba(74,222,128,0.12)"
                : isActive ? "rgba(56,189,248,0.08)" : "rgba(255,255,255,0.02)",
              border: `1px solid ${isWinner ? "rgba(74,222,128,0.4)"
                : isActive ? "rgba(56,189,248,0.3)" : "rgba(255,255,255,0.08)"}`,
              borderRadius: 8, padding: "10px 10px 8px",
              transition: "all 0.2s", position: "relative", overflow: "hidden",
            }}>
              {/* Feedback flash overlay */}
              {fb && (
                <div style={{
                  position: "absolute", inset: 0, pointerEvents: "none",
                  background: fb === "up" ? "rgba(74,222,128,0.15)" : "rgba(244,63,94,0.15)",
                  animation: "fadeIn 0.1s ease-out",
                }} />
              )}

              {/* Name + trend arrow */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontSize: "0.78rem", fontWeight: 600, color: COLORS[i] }}>{m.name}</span>
                <span style={{ fontSize: "0.72rem" }}>
                  {m.trend === "up" && <span style={{ color: "#4ade80" }}>&#9650;</span>}
                  {m.trend === "down" && <span style={{ color: "#f43f5e" }}>&#9660;</span>}
                  {m.trend === "stable" && <span style={{ color: "#a1a1aa" }}>&#8212;</span>}
                </span>
              </div>

              {/* ELO number */}
              <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "#e4e4e7", marginBottom: 4 }}>
                {m.elo}
              </div>

              {/* Weight bar */}
              <div style={{ marginBottom: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "#a1a1aa", marginBottom: 2 }}>
                  <span>Weight</span><span>{m.weight.toFixed(2)}</span>
                </div>
                <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", width: `${m.weight * 100}%`,
                    background: COLORS[i], borderRadius: 2,
                    transition: "width 0.4s ease", opacity: 0.7,
                  }} />
                </div>
              </div>

              {/* Win rate */}
              <div style={{ fontSize: "0.65rem", color: "#a1a1aa", marginBottom: 6 }}>
                Win rate: {m.rounds > 0 ? `${Math.round((m.wins / m.rounds) * 100)}%` : "--"}
              </div>

              {/* Thumbs up / down */}
              <div style={{ display: "flex", gap: 4 }}>
                <button className="btn-mono" onClick={() => handleFeedback(i, "up")}
                  style={{ flex: 1, padding: "2px 0", fontSize: "0.72rem", textAlign: "center" }}
                  title="Positive feedback">&#128077;</button>
                <button className="btn-mono" onClick={() => handleFeedback(i, "down")}
                  style={{ flex: 1, padding: "2px 0", fontSize: "0.72rem", textAlign: "center" }}
                  title="Negative feedback">&#128078;</button>
              </div>

              {/* Winner badge */}
              {isWinner && (
                <div style={{
                  position: "absolute", top: 4, right: 6, fontSize: "0.6rem",
                  fontWeight: 700, color: "#4ade80", textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}>Winner</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Charts row */}
      <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        {/* ELO history line chart */}
        <div style={{ flex: "1 1 320px", minWidth: 0 }}>
          <div style={{ fontSize: "0.72rem", color: "#a1a1aa", marginBottom: 6, fontFamily: mono }}>
            ELO Rating Over Time
          </div>
          <svg viewBox={`0 0 ${cW} ${cH}`} style={{ width: "100%", height: "auto", maxHeight: 200 }}>
            {/* Horizontal grid + Y labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((f) => {
              const y = pT + plotH * (1 - f);
              return (
                <g key={f}>
                  <line x1={pL} y1={y} x2={pL + plotW} y2={y}
                    stroke="rgba(255,255,255,0.06)" strokeWidth={0.5} />
                  <text x={pL - 4} y={y + 3} textAnchor="end"
                    fill="#a1a1aa" fontSize={8} fontFamily="monospace">
                    {Math.round(minE + eloRange * f)}
                  </text>
                </g>
              );
            })}

            {/* X-axis round labels */}
            {Array.from({ length: Math.min(MAX_ROUNDS + 1, roundNum + 1) }, (_, i) => i).map((r) => (
              <text key={r} x={toX(r)} y={cH - 4} textAnchor="middle"
                fill="#a1a1aa" fontSize={8} fontFamily="monospace">{r}</text>
            ))}

            {/* Model lines */}
            {COLORS.map((color, mi) => {
              if (history.length < 2) return null;
              const pts = history.map((h) => `${toX(h.round)},${toY(h.elos[mi])}`).join(" ");
              const last = history[history.length - 1];
              return (
                <g key={mi}>
                  <polyline points={pts} fill="none" stroke={color}
                    strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
                    opacity={0.8} style={{
                      strokeDasharray: history.length * 100,
                      strokeDashoffset: 0,
                      transition: "stroke-dashoffset 0.6s ease",
                    }} />
                  <circle cx={toX(last.round)} cy={toY(last.elos[mi])}
                    r={3} fill={color} opacity={0.9} />
                </g>
              );
            })}

            {/* Axis lines */}
            <line x1={pL} y1={pT} x2={pL} y2={pT + plotH}
              stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
            <line x1={pL} y1={pT + plotH} x2={pL + plotW} y2={pT + plotH}
              stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
          </svg>

          {/* Legend */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 4 }}>
            {models.map((m, i) => (
              <div key={m.name} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.65rem", color: "#a1a1aa" }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: COLORS[i], opacity: 0.8 }} />
                {m.name}
              </div>
            ))}
          </div>
        </div>

        {/* Confidence calibration mini chart */}
        <div style={{ flex: "0 0 200px" }}>
          <div style={{ fontSize: "0.72rem", color: "#a1a1aa", marginBottom: 6, fontFamily: mono }}>
            Confidence Calibration
          </div>
          <svg viewBox={`0 0 ${calW} ${calH}`} style={{ width: "100%", height: "auto", maxHeight: 160 }}>
            {/* Perfect calibration diagonal (dashed) */}
            <line x1={calP} y1={calP + calPH} x2={calP + calPW} y2={calP}
              stroke="rgba(255,255,255,0.12)" strokeWidth={1} strokeDasharray="3 3" />

            {/* Grid + axis labels */}
            {[0, 0.5, 1].map((f) => {
              const x = calP + calPW * f, y = calP + calPH * (1 - f);
              return (
                <g key={f}>
                  <line x1={calP} y1={y} x2={calP + calPW} y2={y}
                    stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
                  <text x={calP - 4} y={y + 3} textAnchor="end"
                    fill="#a1a1aa" fontSize={7} fontFamily="monospace">{f.toFixed(1)}</text>
                  <text x={x} y={calP + calPH + 12} textAnchor="middle"
                    fill="#a1a1aa" fontSize={7} fontFamily="monospace">{f.toFixed(1)}</text>
                </g>
              );
            })}

            {/* Actual calibration curve */}
            <polyline
              points={CAL.map((p) => `${calP + p.ex * calPW},${calP + calPH - p.ac * calPH}`).join(" ")}
              fill="none" stroke="#38bdf8" strokeWidth={1.5}
              strokeLinecap="round" strokeLinejoin="round" opacity={0.8} />

            {/* Data point dots */}
            {CAL.map((p, i) => (
              <circle key={i} cx={calP + p.ex * calPW} cy={calP + calPH - p.ac * calPH}
                r={2.5} fill="#38bdf8" opacity={0.7} />
            ))}

            {/* Axis titles */}
            <text x={calP + calPW / 2} y={calH - 2} textAnchor="middle"
              fill="#a1a1aa" fontSize={7} fontFamily="monospace">Expected</text>
            <text x={6} y={calP + calPH / 2} textAnchor="middle"
              fill="#a1a1aa" fontSize={7} fontFamily="monospace"
              transform={`rotate(-90, 6, ${calP + calPH / 2})`}>Actual</text>

            {/* Axes */}
            <line x1={calP} y1={calP} x2={calP} y2={calP + calPH}
              stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
            <line x1={calP} y1={calP + calPH} x2={calP + calPW} y2={calP + calPH}
              stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
          </svg>
          <div style={{ fontSize: "0.6rem", color: "#71717a", marginTop: 2, textAlign: "center" }}>
            Blue curve vs. perfect diagonal
          </div>
        </div>
      </div>

      {/* EMA Formula block */}
      <div style={{
        background: "rgba(0,0,0,0.4)", border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 8, padding: "10px 14px", marginBottom: 12,
        fontFamily: mono, fontSize: "0.78rem", color: "#a1a1aa",
        lineHeight: 1.7, overflowX: "auto",
      }}>
        <span style={{ color: "#71717a" }}>// EMA weight update</span><br />
        <span style={{ color: "#38bdf8" }}>weight</span>
        <span style={{ color: "#71717a" }}>_new</span>
        {" = "}<span style={{ color: "#facc15" }}>{ALPHA}</span>
        {" * "}<span style={{ color: "#4ade80" }}>win</span>
        {" + (1 - "}<span style={{ color: "#facc15" }}>{ALPHA}</span>
        {") * "}<span style={{ color: "#38bdf8" }}>weight</span>
        <span style={{ color: "#71717a" }}>_old</span>
        <br /><br />
        <span style={{ color: "#71717a" }}>// ELO update (K={K})</span><br />
        <span style={{ color: "#38bdf8" }}>E</span>
        {" = 1 / (1 + 10"}
        <sup style={{ fontSize: "0.65rem" }}>
          {"(R"}<sub>b</sub>{" - R"}<sub>a</sub>{") / 400"}
        </sup>
        {")"}<br />
        <span style={{ color: "#38bdf8" }}>R&apos;</span>
        {" = R + "}<span style={{ color: "#facc15" }}>{K}</span>
        {" * (S - E)"}
      </div>

      {/* Status bar */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        fontSize: "0.68rem", color: "#71717a", fontFamily: mono,
      }}>
        <span>Round {roundNum}/{MAX_ROUNDS} &middot; Alpha={ALPHA} &middot; K={K}</span>
        <span>
          {roundNum >= MAX_ROUNDS
            ? <span style={{ color: "#4ade80" }}>Simulation complete</span>
            : "Click Simulate Round to begin"}
        </span>
      </div>
    </div>
  );
}
