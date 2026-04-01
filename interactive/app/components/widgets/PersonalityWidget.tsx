"use client";

import React, { useState, useEffect, useCallback } from "react";

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
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

// ── Types ────────────────────────────────────────────────────────────────────
type TabKey = "congress" | "hollywood" | "custom";
type Emotion = "calm" | "excited" | "concerned" | "optimistic";
type CommStyle = "Formal" | "Casual" | "Analytical";

interface BigFive {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface Agent {
  name: string;
  color: string;
  traits: BigFive;
  emotion: Emotion;
  style: CommStyle;
  quote: string;
}

const TRAIT_KEYS: (keyof BigFive)[] = [
  "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
];
const TRAIT_SHORT = ["O", "C", "E", "A", "N"];
const TRAIT_LABELS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"];
const EMOTION_COLORS: Record<Emotion, string> = {
  calm: "#38bdf8", excited: "#f59e0b", concerned: "#f43f5e", optimistic: "#22c55e",
};

// ── Personality sets ─────────────────────────────────────────────────────────
const rng = seededRandom(7749);
const r = (lo: number, hi: number) => Math.round(lo + rng() * (hi - lo));

const CONGRESS: Agent[] = [
  { name: "Sen. Johnson", color: "#6366f1",
    traits: { openness: r(30,50), conscientiousness: r(70,90), extraversion: r(50,70), agreeableness: r(40,60), neuroticism: r(20,40) },
    emotion: "calm", style: "Formal",
    quote: "We must consider the long-term fiscal implications before proceeding." },
  { name: "Rep. Vasquez", color: "#06b6d4",
    traits: { openness: r(60,80), conscientiousness: r(50,70), extraversion: r(70,90), agreeableness: r(60,80), neuroticism: r(30,50) },
    emotion: "optimistic", style: "Casual",
    quote: "This is exactly the kind of bold move our constituents need!" },
  { name: "Sen. Park", color: "#a855f7",
    traits: { openness: r(70,90), conscientiousness: r(60,80), extraversion: r(30,50), agreeableness: r(50,70), neuroticism: r(40,60) },
    emotion: "concerned", style: "Analytical",
    quote: "The data suggests a 73% probability of unintended consequences." },
  { name: "Rep. Clarke", color: "#f59e0b",
    traits: { openness: r(40,60), conscientiousness: r(80,95), extraversion: r(60,80), agreeableness: r(30,50), neuroticism: r(50,70) },
    emotion: "excited", style: "Formal",
    quote: "I move to table this amendment pending a full committee review." },
];

const HOLLYWOOD: Agent[] = [
  { name: "Morgan Freeman", color: "#f59e0b",
    traits: { openness: r(75,90), conscientiousness: r(70,85), extraversion: r(50,65), agreeableness: r(80,95), neuroticism: r(10,25) },
    emotion: "calm", style: "Formal",
    quote: "Every story has its own wisdom, if you listen closely enough." },
  { name: "Zendaya", color: "#a855f7",
    traits: { openness: r(80,95), conscientiousness: r(65,80), extraversion: r(60,80), agreeableness: r(70,85), neuroticism: r(25,40) },
    emotion: "optimistic", style: "Casual",
    quote: "I think we should just go for it and see what happens!" },
  { name: "Keanu Reeves", color: "#22c55e",
    traits: { openness: r(70,85), conscientiousness: r(55,70), extraversion: r(30,45), agreeableness: r(85,98), neuroticism: r(15,30) },
    emotion: "calm", style: "Casual",
    quote: "Whoa. That is... actually a really thoughtful approach." },
  { name: "Meryl Streep", color: "#f43f5e",
    traits: { openness: r(85,98), conscientiousness: r(80,95), extraversion: r(55,70), agreeableness: r(60,75), neuroticism: r(35,50) },
    emotion: "concerned", style: "Analytical",
    quote: "We need to examine every angle before committing to this." },
];

const CUSTOM_DEFAULT: Agent[] = [
  { name: "Agent Alpha", color: "#38bdf8",
    traits: { openness: 50, conscientiousness: 50, extraversion: 50, agreeableness: 50, neuroticism: 50 },
    emotion: "calm", style: "Formal",
    quote: "Balanced perspective, awaiting your configuration." },
  { name: "Agent Beta", color: "#22c55e",
    traits: { openness: 70, conscientiousness: 30, extraversion: 80, agreeableness: 60, neuroticism: 20 },
    emotion: "excited", style: "Casual",
    quote: "Ready to shake things up with some fresh ideas!" },
  { name: "Agent Gamma", color: "#f59e0b",
    traits: { openness: 30, conscientiousness: 90, extraversion: 40, agreeableness: 40, neuroticism: 70 },
    emotion: "concerned", style: "Analytical",
    quote: "Proceeding with caution based on risk assessment." },
  { name: "Agent Delta", color: "#a855f7",
    traits: { openness: 85, conscientiousness: 60, extraversion: 60, agreeableness: 80, neuroticism: 30 },
    emotion: "optimistic", style: "Formal",
    quote: "I see great potential in collaborative solutions here." },
];

const SAMPLE_QUESTION = "Should we adopt a fully autonomous AI governance model?";

// ── Derived personality functions ────────────────────────────────────────────
function deriveEmotion(t: BigFive): Emotion {
  if (t.neuroticism > 65) return "concerned";
  if (t.extraversion > 70 && t.openness > 60) return "excited";
  if (t.agreeableness > 70 && t.openness > 60) return "optimistic";
  return "calm";
}

function deriveStyle(t: BigFive): CommStyle {
  if (t.conscientiousness > 70) return "Formal";
  if (t.extraversion > 65 && t.openness > 60) return "Casual";
  return "Analytical";
}

function voteWeight(t: BigFive): number {
  const agree = t.agreeableness / 100 * 0.3;
  const consc = t.conscientiousness / 150;
  const neuro = t.neuroticism / 200;
  return Math.max(0.3, Math.min(1.0, 0.5 + agree + consc - neuro));
}

function voteStance(t: BigFive, em: Emotion): { label: string; value: number } {
  let base = (t.openness * 0.4 + t.agreeableness * 0.3 - t.neuroticism * 0.3) / 100;
  if (em === "optimistic") base += 0.15;
  if (em === "concerned") base -= 0.2;
  if (em === "excited") base += 0.1;
  const v = Math.max(0, Math.min(1, base * 0.5 + 0.5));
  if (v > 0.6) return { label: "For", value: v };
  if (v < 0.4) return { label: "Against", value: v };
  return { label: "Abstain", value: v };
}

function emotionModifier(em: Emotion): string {
  return em === "optimistic" ? "+0.15" : em === "excited" ? "+0.10"
    : em === "concerned" ? "-0.20" : "+0.00";
}

const stanceColor = (l: string) =>
  l === "For" ? C.green : l === "Against" ? C.rose : C.amber;

// ── SVG Radar chart ──────────────────────────────────────────────────────────
function RadarChart({ traits, color, size = 90 }: { traits: BigFive; color: string; size?: number }) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.38;
  const values = TRAIT_KEYS.map((k) => traits[k] / 100);
  const angles = TRAIT_KEYS.map((_, i) => (2 * Math.PI * i) / 5);

  const polarToXY = (angle: number, r: number) => ({
    x: cx + r * Math.cos(angle - Math.PI / 2),
    y: cy + r * Math.sin(angle - Math.PI / 2),
  });

  const ringScales = [0.25, 0.5, 0.75, 1.0];
  const dataPts = angles.map((a, i) => polarToXY(a, radius * values[i]));
  const axisPts = angles.map((a) => polarToXY(a, radius));
  const labelPts = angles.map((a) => polarToXY(a, radius + 14));

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {ringScales.map((s, i) => (
        <polygon key={i}
          points={angles.map((a) => { const p = polarToXY(a, radius * s); return `${p.x},${p.y}`; }).join(" ")}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={0.5} />
      ))}
      {axisPts.map((p, i) => (
        <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y}
          stroke="rgba(255,255,255,0.06)" strokeWidth={0.5} />
      ))}
      <polygon
        points={dataPts.map((p) => `${p.x},${p.y}`).join(" ")}
        fill={`${color}20`} stroke={color} strokeWidth={1.5}
        style={{ transition: "all 0.4s ease" }}
      />
      {dataPts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={2.5} fill={color}
          style={{ transition: "all 0.4s ease" }} />
      ))}
      {labelPts.map((p, i) => (
        <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle"
          fill={C.muted} fontSize={8} fontFamily="monospace">{TRAIT_SHORT[i]}</text>
      ))}
    </svg>
  );
}

// ── Main component ───────────────────────────────────────────────────────────
export default function PersonalityWidget() {
  const [mounted, setMounted] = useState(false);
  const [tab, setTab] = useState<TabKey>("congress");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [showVoting, setShowVoting] = useState(false);
  const [voteAnim, setVoteAnim] = useState(0);

  const [congressAgents, setCongressAgents] = useState<Agent[]>(CONGRESS);
  const [hollywoodAgents, setHollywoodAgents] = useState<Agent[]>(HOLLYWOOD);
  const [customAgents, setCustomAgents] = useState<Agent[]>(CUSTOM_DEFAULT);

  useEffect(() => { setMounted(true); }, []);

  const agents = tab === "congress"
    ? congressAgents : tab === "hollywood"
    ? hollywoodAgents : customAgents;

  const setAgents = useCallback(
    (fn: (prev: Agent[]) => Agent[]) => {
      if (tab === "congress") setCongressAgents(fn);
      else if (tab === "hollywood") setHollywoodAgents(fn);
      else setCustomAgents(fn);
    },
    [tab],
  );

  const updateTrait = useCallback(
    (agentIdx: number, key: keyof BigFive, value: number) => {
      setAgents((prev) => {
        const next = [...prev];
        const a = { ...next[agentIdx], traits: { ...next[agentIdx].traits, [key]: value } };
        a.emotion = deriveEmotion(a.traits);
        a.style = deriveStyle(a.traits);
        next[agentIdx] = a;
        return next;
      });
    },
    [setAgents],
  );

  const runVote = useCallback(() => {
    setShowVoting(true);
    setVoteAnim(0);
    let frame = 0;
    const id = setInterval(() => {
      frame++;
      setVoteAnim(Math.min(frame / 30, 1));
      if (frame >= 30) clearInterval(id);
    }, 50);
  }, []);

  const handleTab = useCallback((t: TabKey) => {
    setTab(t);
    setExpandedIdx(null);
    setShowVoting(false);
    setVoteAnim(0);
  }, []);

  if (!mounted) {
    return (
      <div className="widget-container ch10">
        <div className="widget-label">Interactive &middot; Personality &amp; Emotion</div>
        <div style={{ height: 400 }} />
      </div>
    );
  }

  return (
    <div className="widget-container ch10">
      <div className="widget-label">Interactive &middot; Personality &amp; Emotion</div>

      {/* ── Tab bar ───────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {(["congress", "hollywood", "custom"] as TabKey[]).map((t) => (
          <button
            key={t}
            className={`btn-mono${tab === t ? " active" : ""}`}
            onClick={() => handleTab(t)}
            style={{ fontSize: 12, padding: "5px 14px", textTransform: "capitalize" }}
          >
            {t === "congress" ? "Congress" : t === "hollywood" ? "Hollywood" : "Custom"}
          </button>
        ))}
      </div>

      {/* ── Agent cards grid ──────────────────────────────────────────── */}
      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16,
      }}>
        {agents.map((agent, idx) => {
          const expanded = expandedIdx === idx;
          const stance = voteStance(agent.traits, agent.emotion);
          const weight = voteWeight(agent.traits);

          return (
            <div
              key={`${tab}-${idx}`}
              onClick={() => setExpandedIdx(expanded ? null : idx)}
              style={{
                background: expanded
                  ? `linear-gradient(135deg, ${agent.color}08, ${agent.color}04)`
                  : "rgba(255,255,255,0.02)",
                border: `1px solid ${expanded ? agent.color + "40" : C.border}`,
                borderRadius: 10,
                padding: expanded ? 14 : 12,
                cursor: "pointer",
                transition: "all 0.3s ease",
                gridColumn: expanded ? "1 / -1" : "auto",
              }}
            >
              {/* Header row */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <div style={{
                  width: 10, height: 10, borderRadius: "50%",
                  background: agent.color, flexShrink: 0,
                }} />
                <span style={{
                  color: C.text, fontSize: 13, fontFamily: "monospace", fontWeight: 600,
                }}>
                  {agent.name}
                </span>
                <span style={{
                  marginLeft: "auto", fontSize: 9, fontFamily: "monospace",
                  padding: "2px 6px", borderRadius: 4,
                  background: `${agent.color}15`, color: agent.color,
                  border: `1px solid ${agent.color}30`, transition: "all 0.3s ease",
                }}>
                  {agent.style}
                </span>
              </div>

              {/* Radar chart + info column */}
              <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <div style={{ flexShrink: 0 }}>
                  <RadarChart traits={agent.traits} color={agent.color} size={expanded ? 120 : 90} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  {/* Emotional state */}
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                    <div style={{
                      width: 7, height: 7, borderRadius: "50%",
                      background: EMOTION_COLORS[agent.emotion],
                      boxShadow: `0 0 6px ${EMOTION_COLORS[agent.emotion]}60`,
                      transition: "all 0.4s ease",
                    }} />
                    <span style={{
                      fontSize: 10, fontFamily: "monospace",
                      color: EMOTION_COLORS[agent.emotion],
                      textTransform: "capitalize", transition: "color 0.4s ease",
                    }}>
                      {agent.emotion}
                    </span>
                  </div>
                  {/* Quote */}
                  <div style={{
                    fontSize: 10, fontFamily: "monospace", color: C.muted,
                    lineHeight: 1.4, fontStyle: "italic",
                  }}>
                    &ldquo;{agent.quote}&rdquo;
                  </div>
                  {/* Inline vote badge when voting active */}
                  {showVoting && (
                    <div style={{
                      marginTop: 6, display: "flex", alignItems: "center", gap: 6,
                      opacity: voteAnim, transition: "opacity 0.3s ease",
                    }}>
                      <span style={{
                        fontSize: 10, fontFamily: "monospace",
                        color: stanceColor(stance.label), fontWeight: 600,
                      }}>
                        {stance.label}
                      </span>
                      <span style={{ fontSize: 9, fontFamily: "monospace", color: C.muted }}>
                        w={weight.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* ── Expanded: Big Five sliders ───────────────────────── */}
              {expanded && (
                <div
                  style={{ marginTop: 12, paddingTop: 10, borderTop: `1px solid ${C.border}` }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <div style={{
                    fontSize: 10, fontFamily: "monospace", color: C.muted, marginBottom: 8,
                  }}>
                    Adjust Big Five Traits
                  </div>
                  {TRAIT_KEYS.map((key, ti) => (
                    <div key={key} style={{
                      display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
                    }}>
                      <span style={{
                        width: 110, fontSize: 10, fontFamily: "monospace", color: C.text,
                      }}>
                        {TRAIT_LABELS[ti]}
                      </span>
                      <input
                        type="range" min={0} max={100}
                        value={agent.traits[key]}
                        onChange={(e) => updateTrait(idx, key, parseInt(e.target.value, 10))}
                        style={{ flex: 1, accentColor: agent.color, height: 4, cursor: "pointer" }}
                      />
                      <span style={{
                        width: 28, textAlign: "right", fontSize: 10,
                        fontFamily: "monospace", color: agent.color,
                      }}>
                        {agent.traits[key]}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Voting section ────────────────────────────────────────────── */}
      <div style={{
        background: "rgba(255,255,255,0.02)",
        border: `1px solid ${C.border}`, borderRadius: 10, padding: 14,
      }}>
        <div style={{
          fontSize: 10, fontFamily: "monospace", color: C.muted,
          marginBottom: 6, textTransform: "uppercase", letterSpacing: 1,
        }}>
          Sample Question
        </div>
        <div style={{
          fontSize: 13, fontFamily: "monospace", color: C.text,
          marginBottom: 14, lineHeight: 1.4,
        }}>
          &ldquo;{SAMPLE_QUESTION}&rdquo;
        </div>

        <button
          className="btn-mono"
          onClick={runVote}
          style={{ fontSize: 12, padding: "6px 16px", marginBottom: 14 }}
        >
          {showVoting ? "Re-run Vote" : "Run Personality Vote"}
        </button>

        {/* Vote result rows */}
        {showVoting && (
          <div>
            {agents.map((agent, idx) => {
              const stance = voteStance(agent.traits, agent.emotion);
              const weight = voteWeight(agent.traits);
              const barWidth = Math.max(5, stance.value * 100 * voteAnim);

              return (
                <div key={`vote-${tab}-${idx}`} style={{
                  display: "flex", alignItems: "center", gap: 8, marginBottom: 8,
                  opacity: Math.min(voteAnim * 2 - idx * 0.2, 1),
                  transition: "opacity 0.3s ease",
                }}>
                  <span style={{
                    width: 100, fontSize: 10, fontFamily: "monospace", color: agent.color,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {agent.name}
                  </span>
                  <div style={{
                    flex: 1, height: 8, background: "rgba(255,255,255,0.04)",
                    borderRadius: 4, overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", width: `${barWidth}%`,
                      background: stanceColor(stance.label),
                      borderRadius: 4, opacity: 0.7, transition: "width 0.6s ease",
                    }} />
                  </div>
                  <span style={{
                    width: 50, fontSize: 10, fontFamily: "monospace", fontWeight: 600,
                    color: stanceColor(stance.label), textAlign: "center",
                  }}>
                    {stance.label}
                  </span>
                  <span style={{
                    fontSize: 8, fontFamily: "monospace", padding: "1px 5px", borderRadius: 3,
                    background: `${EMOTION_COLORS[agent.emotion]}15`,
                    color: EMOTION_COLORS[agent.emotion], whiteSpace: "nowrap",
                  }}>
                    {emotionModifier(agent.emotion)}
                  </span>
                  <span style={{
                    fontSize: 9, fontFamily: "monospace", color: C.muted,
                    width: 40, textAlign: "right",
                  }}>
                    x{weight.toFixed(2)}
                  </span>
                </div>
              );
            })}

            {/* Consensus result banner */}
            {voteAnim >= 1 && (
              <div style={{
                marginTop: 10, padding: "8px 12px",
                background: `${C.green}10`, border: `1px solid ${C.green}30`,
                borderRadius: 8, fontSize: 11, fontFamily: "monospace", color: C.green,
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span>
                  {(() => {
                    let forW = 0, againstW = 0, abstainW = 0;
                    agents.forEach((a) => {
                      const s = voteStance(a.traits, a.emotion);
                      const w = voteWeight(a.traits);
                      if (s.label === "For") forW += w;
                      else if (s.label === "Against") againstW += w;
                      else abstainW += w;
                    });
                    const total = forW + againstW + abstainW;
                    const winner = forW >= againstW && forW >= abstainW
                      ? "For" : againstW >= forW && againstW >= abstainW
                      ? "Against" : "Abstain";
                    const pct = (winner === "For" ? forW : winner === "Against" ? againstW : abstainW) / total;
                    return `Consensus: ${winner} (${(pct * 100).toFixed(0)}% weighted)`;
                  })()}
                </span>
                <span style={{ color: C.muted, fontSize: 9 }}>personality-weighted vote</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Footer hint ───────────────────────────────────────────────── */}
      <div style={{
        marginTop: 12, fontSize: 10, fontFamily: "monospace",
        color: C.muted, lineHeight: 1.5,
      }}>
        Click an agent card to expand and adjust Big Five traits. Personality shapes emotional
        state, communication style, and vote weight. Agreeable agents compromise more; neurotic
        agents vote more cautiously.
      </div>
    </div>
  );
}
