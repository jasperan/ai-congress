"use client";

import { useState, useEffect, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Data                                                               */
/* ------------------------------------------------------------------ */

interface ModelResponse {
  id: number; model: string; weight: number; confidence: number;
  text: string; cluster: number; color: string;
}

const QUESTION = "What causes rain?";

const R: ModelResponse[] = [
  { id: 0, model: "GPT-4",       weight: 0.92, confidence: 0.95, text: "Rain forms when water vapor in the atmosphere condenses into droplets that become heavy enough to fall.",                           cluster: -1, color: "#38bdf8" },
  { id: 1, model: "Claude-3",    weight: 0.88, confidence: 0.91, text: "Atmospheric water vapor cools and condenses into water droplets, which coalesce until gravity pulls them down as rain.",            cluster: -1, color: "#a78bfa" },
  { id: 2, model: "Mistral-7B",  weight: 0.72, confidence: 0.84, text: "Evaporated water rises, cools at higher altitudes, and condenses into clouds. When droplets grow large enough, they fall as precipitation.", cluster: -1, color: "#4ade80" },
  { id: 3, model: "Phi-3",       weight: 0.65, confidence: 0.78, text: "Rain is caused by the sun heating oceans and lakes, causing evaporation which later condenses in clouds.",                         cluster: -1, color: "#facc15" },
  { id: 4, model: "Llama-3",     weight: 0.70, confidence: 0.80, text: "Warm moist air rises and cools, forming clouds. When cloud droplets merge and become heavy, rain falls.",                          cluster: -1, color: "#f472b6" },
  { id: 5, model: "DeepSeek-R1", weight: 0.60, confidence: 0.72, text: "Precipitation occurs due to the water cycle: evaporation, condensation into clouds, and coalescence of droplets.",                 cluster: -1, color: "#fb923c" },
];

/* Pre-computed similarity matrices */
const SEMANTIC_SIM: number[][] = [
  [1.00, 0.94, 0.88, 0.58, 0.91, 0.62],
  [0.94, 1.00, 0.90, 0.55, 0.89, 0.60],
  [0.88, 0.90, 1.00, 0.61, 0.87, 0.65],
  [0.58, 0.55, 0.61, 1.00, 0.57, 0.85],
  [0.91, 0.89, 0.87, 0.57, 1.00, 0.59],
  [0.62, 0.60, 0.65, 0.85, 0.59, 1.00],
];
const STRING_SIM: number[][] = [
  [1.00, 0.28, 0.22, 0.18, 0.24, 0.20],
  [0.28, 1.00, 0.20, 0.14, 0.26, 0.18],
  [0.22, 0.20, 1.00, 0.16, 0.22, 0.24],
  [0.18, 0.14, 0.16, 1.00, 0.20, 0.22],
  [0.24, 0.26, 0.22, 0.20, 1.00, 0.18],
  [0.20, 0.18, 0.24, 0.22, 0.18, 1.00],
];

const SEMANTIC_CLUSTERS: number[][] = [[0, 1, 2, 4], [3, 5]];
const STRING_CLUSTERS: number[][] = [[0], [1], [2], [3], [4], [5]];
const CC = ["#22d3ee", "#facc15", "#f472b6"]; // cluster colors
const THR = 0.75;

type Phase = "idle" | "judging" | "clustering" | "voting" | "done";
type Mode = "semantic" | "string";

const mono = "var(--font-mono), monospace";
const lbl = { fontSize: "0.72rem", color: "#a1a1aa", textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 10, fontFamily: mono };

function score(ids: number[]) {
  return ids.reduce((s, id) => s + R[id].weight * R[id].confidence, 0);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SemanticVotingWidget() {
  const [mounted, setMounted] = useState(false);
  const [mode, setMode] = useState<Mode>("semantic");
  const [phase, setPhase] = useState<Phase>("idle");
  const [judgingPair, setJudgingPair] = useState<[number, number] | null>(null);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [clusters, setClusters] = useState<number[][] | null>(null);
  const [winnerIdx, setWinnerIdx] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const running = useRef(false);

  useEffect(() => { setMounted(true); return () => { if (timer.current) clearTimeout(timer.current); }; }, []);

  const sim = mode === "semantic" ? SEMANTIC_SIM : STRING_SIM;

  const switchMode = useCallback((m: Mode) => {
    if (running.current) return;
    setMode(m); setPhase("idle"); setJudgingPair(null);
    setRevealed(new Set()); setClusters(null); setWinnerIdx(null);
  }, []);

  const runClustering = useCallback(() => {
    if (running.current) return;
    running.current = true;
    setPhase("judging"); setJudgingPair(null);
    setRevealed(new Set()); setClusters(null); setWinnerIdx(null);

    const pairs: [number, number][] = [];
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) pairs.push([i, j]);
    let step = 0;
    const rev = new Set<string>();

    const tick = () => {
      if (step < pairs.length) {
        const [a, b] = pairs[step];
        setJudgingPair([a, b]);
        rev.add(`${a}-${b}`);
        setRevealed(new Set(rev));
        step++;
        timer.current = setTimeout(tick, 180);
      } else {
        setJudgingPair(null); setPhase("clustering");
        timer.current = setTimeout(() => {
          const cd = mode === "semantic" ? SEMANTIC_CLUSTERS : STRING_CLUSTERS;
          setClusters(cd); setPhase("voting");
          timer.current = setTimeout(() => {
            let best = -1, bi = 0;
            cd.forEach((cl, i) => { const s = score(cl); if (s > best) { best = s; bi = i; } });
            setWinnerIdx(bi); setPhase("done"); running.current = false;
          }, 1000);
        }, 800);
      }
    };
    timer.current = setTimeout(tick, 300);
  }, [mode]);

  const reset = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    running.current = false; setPhase("idle"); setJudgingPair(null);
    setRevealed(new Set()); setClusters(null); setWinnerIdx(null);
  }, []);

  if (!mounted) return null;

  /* Derived */
  const scores = clusters ? clusters.map(cl => score(cl)) : [];
  const maxScore = Math.max(...scores, 0.01);

  const cellBg = (v: number) =>
    v >= THR ? "rgba(34,211,238,0.55)" : v >= 0.5 ? "rgba(250,204,21,0.35)" : "rgba(255,255,255,0.06)";

  const phaseText: Record<Phase, string> = {
    idle: "Ready", judging: "Judge evaluating pairwise similarity...",
    clustering: "Grouping into semantic clusters...",
    voting: "Calculating weighted votes...", done: "Consensus reached",
  };

  const clusterOf = (id: number) => {
    if (!clusters) return -1;
    for (let c = 0; c < clusters.length; c++) if (clusters[c].includes(id)) return c;
    return -1;
  };

  /* ---- Render ---- */
  return (
    <div className="widget-container ch2">
      <div className="widget-label">Interactive &middot; Semantic Voting</div>

      {/* Question */}
      <div style={{ marginBottom: 16 }}>
        <span style={{ color: "#a1a1aa", fontSize: "0.78rem" }}>Question:</span>
        <span style={{ color: "#e4e4e7", fontWeight: 600, marginLeft: 8, fontSize: "0.95rem" }}>
          &ldquo;{QUESTION}&rdquo;
        </span>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 18, flexWrap: "wrap", alignItems: "center" }}>
        <button className={`btn-mono ${mode === "semantic" ? "active" : ""}`}
          onClick={() => switchMode("semantic")} disabled={running.current}>Semantic Voting</button>
        <button className={`btn-mono ${mode === "string" ? "active" : ""}`}
          onClick={() => switchMode("string")} disabled={running.current}>String Match</button>
        <div style={{ flex: 1 }} />
        {phase === "idle"
          ? <button className="btn-mono" onClick={runClustering}
              style={{ color: "#22d3ee", borderColor: "rgba(34,211,238,0.4)" }}>Cluster Responses</button>
          : <button className="btn-mono" onClick={reset}>Reset</button>}
      </div>

      {/* Phase */}
      <div style={{ fontSize: "0.75rem", color: phase === "done" ? "#4ade80" : "#22d3ee",
        fontFamily: mono, marginBottom: 16, minHeight: 18, transition: "color 0.3s" }}>
        {phaseText[phase]}
      </div>

      {/* Two-panel layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>

        {/* LEFT: Responses */}
        <div>
          <div style={lbl}>Model Responses</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {R.map((r) => {
              const ci = clusterOf(r.id);
              const win = winnerIdx !== null && ci === winnerIdx;
              const judged = judgingPair !== null && (judgingPair[0] === r.id || judgingPair[1] === r.id);
              return (
                <div key={r.id} style={{
                  background: win ? "rgba(34,211,238,0.1)" : judged ? "rgba(250,204,21,0.08)" : "rgba(255,255,255,0.03)",
                  border: `1px solid ${win ? "rgba(34,211,238,0.5)" : ci >= 0 ? `${CC[ci % 3]}44` : "rgba(255,255,255,0.06)"}`,
                  borderRadius: 8, padding: "8px 10px", transition: "all 0.4s ease",
                  boxShadow: win ? "0 0 16px rgba(34,211,238,0.2)" : "none", position: "relative" as const,
                }}>
                  {ci >= 0 && <span style={{
                    position: "absolute", top: -6, right: 8, background: CC[ci % 3], color: "#12131a",
                    fontSize: "0.6rem", fontWeight: 700, padding: "1px 6px", borderRadius: 4, fontFamily: mono,
                  }}>C{ci + 1}</span>}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <span style={{ fontSize: "0.75rem", fontWeight: 600, color: r.color }}>{r.model}</span>
                    <span style={{ fontSize: "0.65rem", color: "#a1a1aa", fontFamily: mono }}>
                      w={r.weight.toFixed(2)} c={r.confidence.toFixed(2)}</span>
                  </div>
                  <div style={{ fontSize: "0.73rem", color: "#b4b4bc", lineHeight: 1.5 }}>{r.text}</div>
                  <div style={{ marginTop: 6, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${r.weight * 100}%`, height: "100%",
                      background: r.color, opacity: 0.6, borderRadius: 2 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* RIGHT: Matrix + Clusters */}
        <div>
          <div style={lbl}>{mode === "semantic" ? "Semantic" : "String"} Similarity Matrix</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", tableLayout: "fixed" }}>
              <thead><tr>
                <th style={{ width: 54, padding: 3 }} />
                {R.map((r) => (
                  <th key={r.id} style={{ fontSize: "0.6rem", color: r.color, fontWeight: 600,
                    padding: 3, textAlign: "center", fontFamily: mono }}>{r.model.slice(0, 5)}</th>
                ))}
              </tr></thead>
              <tbody>
                {R.map((row, i) => (
                  <tr key={row.id}>
                    <td style={{ fontSize: "0.6rem", color: row.color, fontWeight: 600,
                      padding: "3px 4px", fontFamily: mono, textAlign: "right" }}>{row.model.slice(0, 5)}</td>
                    {R.map((_, j) => {
                      const k = i < j ? `${i}-${j}` : `${j}-${i}`;
                      const vis = i === j || revealed.has(k) || phase === "done" || clusters !== null;
                      const v = sim[i][j];
                      const active = judgingPair !== null &&
                        ((judgingPair[0] === i && judgingPair[1] === j) || (judgingPair[0] === j && judgingPair[1] === i));
                      return (
                        <td key={j} style={{
                          textAlign: "center", fontSize: "0.62rem", fontFamily: mono, padding: 3,
                          background: vis ? (i === j ? "rgba(255,255,255,0.04)" : cellBg(v)) : "rgba(255,255,255,0.02)",
                          color: vis ? (v >= THR ? "#e4e4e7" : "#a1a1aa") : "transparent",
                          border: active ? "1px solid #facc15" : "1px solid rgba(255,255,255,0.04)",
                          borderRadius: 3, transition: "all 0.25s", fontWeight: v >= THR ? 700 : 400,
                        }}>{vis ? (i === j ? "\u2014" : v.toFixed(2)) : "\u00B7"}</td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div style={{ display: "flex", gap: 12, marginTop: 10, fontSize: "0.62rem", color: "#a1a1aa", fontFamily: mono, flexWrap: "wrap" }}>
            {[
              { bg: "rgba(34,211,238,0.55)", label: `\u2265${THR} (cluster)` },
              { bg: "rgba(250,204,21,0.35)", label: "\u22650.50" },
              { bg: "rgba(255,255,255,0.06)", label: "<0.50" },
            ].map(({ bg, label }) => (
              <span key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: bg, display: "inline-block" }} />
                {label}
              </span>
            ))}
          </div>

          {/* Cluster results */}
          {clusters && (
            <div style={{ marginTop: 20, animation: "fadeIn 0.5s ease-out" }}>
              <div style={lbl}>Clusters &amp; Weighted Votes</div>
              {clusters.map((cl, idx) => {
                const s = scores[idx];
                const win = winnerIdx === idx;
                const cc = CC[idx % 3];
                return (
                  <div key={idx} style={{
                    background: win ? "rgba(34,211,238,0.08)" : "rgba(255,255,255,0.03)",
                    border: `1px solid ${win ? "rgba(34,211,238,0.5)" : "rgba(255,255,255,0.06)"}`,
                    borderRadius: 8, padding: "10px 12px", marginBottom: 8,
                    transition: "all 0.5s ease", boxShadow: win ? "0 0 20px rgba(34,211,238,0.15)" : "none",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ background: cc, color: "#12131a", fontSize: "0.62rem",
                          fontWeight: 700, padding: "1px 6px", borderRadius: 4, fontFamily: mono }}>C{idx + 1}</span>
                        <span style={{ fontSize: "0.73rem", color: "#e4e4e7" }}>
                          {cl.length} response{cl.length !== 1 ? "s" : ""}</span>
                        {win && <span style={{ fontSize: "0.62rem", color: "#4ade80", fontWeight: 700,
                          fontFamily: mono, textTransform: "uppercase" }}>Winner</span>}
                      </div>
                      <span style={{ fontSize: "0.75rem", fontWeight: 700,
                        color: win ? "#22d3ee" : "#a1a1aa", fontFamily: mono }}>{s.toFixed(2)}</span>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
                      {cl.map((rid) => (
                        <span key={rid} style={{ fontSize: "0.62rem", color: R[rid].color,
                          background: "rgba(255,255,255,0.04)", padding: "1px 6px",
                          borderRadius: 4, fontFamily: mono }}>{R[rid].model}</span>
                      ))}
                    </div>
                    <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${(s / maxScore) * 100}%`, height: "100%",
                        background: win ? "linear-gradient(90deg, #22d3ee, #67e8f9)" : `${cc}88`,
                        borderRadius: 2, transition: "width 0.8s ease" }} />
                    </div>
                    <div style={{ marginTop: 6, fontSize: "0.6rem", color: "#a1a1aa", fontFamily: mono }}>
                      {cl.map((rid, k) => (
                        <span key={rid}>{k > 0 && " + "}{R[rid].weight.toFixed(2)}&times;{R[rid].confidence.toFixed(2)}</span>
                      ))}{" = "}{s.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Consensus summary */}
          {phase === "done" && winnerIdx !== null && clusters && (
            <div style={{ marginTop: 12, padding: "10px 12px", borderRadius: 8,
              background: "rgba(74,222,128,0.06)", border: "1px solid rgba(74,222,128,0.2)",
              animation: "fadeIn 0.5s ease-out" }}>
              <div style={{ fontSize: "0.72rem", fontWeight: 700, color: "#4ade80",
                marginBottom: 4, fontFamily: mono }}>Consensus Answer</div>
              <div style={{ fontSize: "0.73rem", color: "#e4e4e7", lineHeight: 1.5 }}>
                {mode === "semantic" ? (
                  <>Cluster C{winnerIdx + 1} wins with{" "}
                    <span style={{ color: "#22d3ee", fontWeight: 600 }}>{clusters[winnerIdx].length} agreeing models</span>
                    {" "}and combined score{" "}
                    <span style={{ color: "#22d3ee", fontWeight: 600 }}>{scores[winnerIdx].toFixed(2)}</span>.
                    Semantic similarity detected that differently-worded responses conveyed the same core meaning.</>
                ) : (
                  <>With string matching, every response is in its own cluster
                    (max similarity {Math.max(...STRING_SIM.flatMap((row, i) => row.filter((_, j) => j !== i))).toFixed(2)} &lt; {THR} threshold).
                    {" "}<span style={{ color: "#facc15", fontWeight: 600 }}>No consensus possible</span>
                    &mdash;the strongest single model wins by default, missing the collective agreement that semantic voting reveals.</>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 20, padding: "10px 12px", borderRadius: 8,
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)",
        fontSize: "0.68rem", color: "#a1a1aa", lineHeight: 1.6, fontFamily: mono }}>
        <strong style={{ color: "#e4e4e7" }}>How it works:</strong> A Judge model
        compares every pair of responses for semantic equivalence (not string matching).
        Responses above the {THR} threshold are grouped into clusters.
        Each cluster&apos;s vote is the sum of <em>weight &times; confidence</em> across its
        members. The cluster with the highest combined score becomes the consensus answer.
        {mode === "string" && (
          <span style={{ color: "#facc15", display: "block", marginTop: 6 }}>
            Notice how string matching fails to find agreement &mdash; even though all
            models are saying essentially the same thing in different words.</span>
        )}
      </div>
    </div>
  );
}
