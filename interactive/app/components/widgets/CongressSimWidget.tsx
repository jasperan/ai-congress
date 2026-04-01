"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";

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

// ── Phase definitions ────────────────────────────────────────────────────────
const PHASES = [
  { key: "intro", label: "Introduction", short: "Intro" },
  { key: "committee", label: "Committee", short: "Committee" },
  { key: "floor", label: "Floor Debate", short: "Floor" },
  { key: "amendments", label: "Amendments", short: "Amend" },
  { key: "final", label: "Final Arguments", short: "Final" },
  { key: "voting", label: "Voting", short: "Vote" },
] as const;

type PhaseKey = (typeof PHASES)[number]["key"];
type Position = "For" | "Against" | "Undecided";
type Sentiment = "positive" | "negative" | "neutral";

// ── Agent data ───────────────────────────────────────────────────────────────
interface Agent {
  name: string;
  party: string;
  partyColor: string;
  initialPosition: Position;
  sentiment: Sentiment;
  statements: Record<PhaseKey, string>;
  conviction: number;
  finalVote: Position;
  committee: 0 | 1;
}

const AGENTS: Agent[] = [
  {
    name: "Sen. Rivera",
    party: "Progressive",
    partyColor: C.cyan,
    initialPosition: "For",
    sentiment: "positive",
    statements: {
      intro: "Federal oversight is critical to prevent unchecked AI deployment in sensitive sectors.",
      committee: "We need clear guardrails. Self-regulation has failed in every major industry.",
      floor: "Without federal law, we get a patchwork of 50 state regulations -- chaos for innovation and safety alike.",
      amendments: "I propose Amendment A: Require algorithmic impact assessments for high-risk AI systems.",
      final: "This isn't about stopping AI. It's about ensuring it serves everyone, not just those who build it.",
      voting: "Voting FOR the regulation bill.",
    },
    conviction: 0.85,
    finalVote: "For",
    committee: 0,
  },
  {
    name: "Rep. Chen",
    party: "Technocrat",
    partyColor: C.purple,
    initialPosition: "Against",
    sentiment: "negative",
    statements: {
      intro: "Heavy-handed regulation will push AI development overseas and stifle American innovation.",
      committee: "The EU's AI Act caused a 23% drop in AI startup funding. We cannot repeat that mistake.",
      floor: "I challenge Sen. Rivera: name one federal tech regulation that kept pace with the industry.",
      amendments: "I oppose Amendment A. Impact assessments become rubber stamps. We need technical standards instead.",
      final: "Regulate outcomes, not technology. Existing consumer protection laws already cover AI harms.",
      voting: "Voting AGAINST the regulation bill.",
    },
    conviction: 0.78,
    finalVote: "Against",
    committee: 0,
  },
  {
    name: "Sen. Okafor",
    party: "Centrist",
    partyColor: C.amber,
    initialPosition: "Undecided",
    sentiment: "neutral",
    statements: {
      intro: "Both sides raise valid points. I want to hear specifics before committing to a position.",
      committee: "What enforcement mechanism ensures compliance without creating a bureaucratic bottleneck?",
      floor: "Rep. Chen's data on the EU is compelling, but Sen. Rivera is right about the patchwork problem.",
      amendments: "I support Amendment A with a modification: exempt small companies under 50 employees.",
      final: "I've weighed both arguments carefully. Federal coordination is needed, but with a light touch.",
      voting: "Voting FOR the amended regulation bill.",
    },
    conviction: 0.62,
    finalVote: "For",
    committee: 1,
  },
  {
    name: "Rep. Walsh",
    party: "Libertarian",
    partyColor: C.rose,
    initialPosition: "Against",
    sentiment: "negative",
    statements: {
      intro: "The federal government has no business dictating how private companies develop software.",
      committee: "Every regulation creates compliance costs that favor big tech over small innovators.",
      floor: "Sen. Okafor, your 'light touch' always becomes heavy hands. That's how regulation works.",
      amendments: "I propose Amendment B: Sunset clause -- any regulation expires in 3 years unless renewed.",
      final: "Markets self-correct faster than Congress legislates. Let competition drive AI safety.",
      voting: "Voting AGAINST the regulation bill.",
    },
    conviction: 0.91,
    finalVote: "Against",
    committee: 1,
  },
  {
    name: "Sen. Park",
    party: "Progressive",
    partyColor: C.cyan,
    initialPosition: "For",
    sentiment: "positive",
    statements: {
      intro: "As ranking member of the tech committee, I've seen firsthand how AI bias harms marginalized communities.",
      committee: "The data is clear: algorithmic discrimination in hiring, lending, and policing demands federal action.",
      floor: "Rep. Walsh, your 'market self-correction' argument ignores the people harmed while markets 'figure it out.'",
      amendments: "I support both amendments. Impact assessments with a sunset clause is a reasonable compromise.",
      final: "We have a narrow window to get this right. History will judge us by whether we acted or dithered.",
      voting: "Voting FOR the regulation bill.",
    },
    conviction: 0.88,
    finalVote: "For",
    committee: 1,
  },
];

const AMENDMENTS = [
  {
    id: "A",
    title: "Algorithmic Impact Assessments",
    proposer: "Sen. Rivera",
    text: "Require impact assessments for all high-risk AI systems before deployment.",
    votesFor: 3,
    votesAgainst: 2,
  },
  {
    id: "B",
    title: "3-Year Sunset Clause",
    proposer: "Rep. Walsh",
    text: "All provisions expire after 3 years unless explicitly renewed by Congress.",
    votesFor: 4,
    votesAgainst: 1,
  },
];

const INITIAL_BILL =
  "The Artificial Intelligence Accountability Act shall establish federal guidelines for the development, deployment, and oversight of AI systems operating within the United States.";

// ── Sentiment icon (SVG, no emoji) ───────────────────────────────────────────
function SentimentIcon({ sentiment }: { sentiment: Sentiment }) {
  const size = 14;
  if (sentiment === "positive")
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
        <polygon points="8,2 14,14 2,14" fill={C.green} opacity={0.8} />
      </svg>
    );
  if (sentiment === "negative")
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
        <polygon points="8,14 14,2 2,2" fill={C.rose} opacity={0.8} />
      </svg>
    );
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
      <rect x="2" y="6" width="12" height="4" rx="2" fill={C.amber} opacity={0.8} />
    </svg>
  );
}

// ── Position badge ───────────────────────────────────────────────────────────
function PositionBadge({ position }: { position: Position }) {
  const color =
    position === "For" ? C.green : position === "Against" ? C.rose : C.amber;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 8px",
        borderRadius: 4,
        fontSize: 10,
        fontFamily: "monospace",
        fontWeight: 600,
        background: `${color}20`,
        color,
        border: `1px solid ${color}40`,
      }}
    >
      {position}
    </span>
  );
}

// ── Main component ───────────────────────────────────────────────────────────
export default function CongressSimWidget() {
  const [mounted, setMounted] = useState(false);
  const [phaseIdx, setPhaseIdx] = useState(-1); // -1 = not started
  const [revealedAgents, setRevealedAgents] = useState<number>(0);
  const [votesRevealed, setVotesRevealed] = useState<number>(0);
  const [transitioning, setTransitioning] = useState(false);
  const [billText, setBillText] = useState(INITIAL_BILL);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => setMounted(true), []);
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const phase = phaseIdx >= 0 ? PHASES[phaseIdx].key : null;

  // Derived positions evolve through simulation
  const getPosition = useCallback(
    (agent: Agent): Position => {
      if (!phase || phase === "intro") return agent.initialPosition;
      if (phase === "committee" || phase === "floor") {
        // Okafor starts shifting
        if (agent.name === "Sen. Okafor") return "Undecided";
        return agent.initialPosition;
      }
      // amendments onward: Okafor locks in
      if (agent.name === "Sen. Okafor") return "For";
      return agent.initialPosition;
    },
    [phase]
  );

  const getSentiment = useCallback(
    (agent: Agent): Sentiment => {
      if (!phase || phase === "intro") return agent.sentiment;
      if (phase === "floor" || phase === "amendments") {
        if (agent.name === "Sen. Okafor") return "positive";
      }
      return agent.sentiment;
    },
    [phase]
  );

  // Conviction grows toward final phase
  const getConviction = useCallback(
    (agent: Agent): number => {
      if (!phase) return 0.3;
      const phaseProgress =
        { intro: 0.3, committee: 0.45, floor: 0.6, amendments: 0.75, final: 0.9, voting: 1.0 }[phase] ?? 0.3;
      return agent.conviction * phaseProgress;
    },
    [phase]
  );

  // ── Phase advancement ──────────────────────────────────────────────────────
  const advancePhase = useCallback(() => {
    if (transitioning) return;
    const nextIdx = phaseIdx + 1;
    if (nextIdx >= PHASES.length) return;

    setTransitioning(true);
    setRevealedAgents(0);
    setVotesRevealed(0);

    // Brief transition delay
    timerRef.current = setTimeout(() => {
      setPhaseIdx(nextIdx);
      setTransitioning(false);

      // Phase 1 (intro): reveal agents one by one
      if (nextIdx === 0) {
        let count = 0;
        const revealInterval = setInterval(() => {
          count++;
          setRevealedAgents(count);
          if (count >= AGENTS.length) clearInterval(revealInterval);
        }, 400);
      } else if (nextIdx === 5) {
        // Voting phase: reveal votes one by one
        let count = 0;
        const voteInterval = setInterval(() => {
          count++;
          setVotesRevealed(count);
          if (count >= AGENTS.length) clearInterval(voteInterval);
        }, 700);
      } else {
        setRevealedAgents(AGENTS.length);
      }

      // Apply amendment text in phase 3
      if (nextIdx === 3) {
        setBillText(
          INITIAL_BILL +
            " [Amendment A: Impact assessments required for high-risk AI.]" +
            " [Amendment B: All provisions sunset after 3 years.]"
        );
      }
    }, 300);
  }, [phaseIdx, transitioning]);

  const reset = useCallback(() => {
    setPhaseIdx(-1);
    setRevealedAgents(0);
    setVotesRevealed(0);
    setTransitioning(false);
    setBillText(INITIAL_BILL);
  }, []);

  // ── Vote tally ─────────────────────────────────────────────────────────────
  const tally = { for: 0, against: 0, abstain: 0 };
  if (phase === "voting") {
    AGENTS.slice(0, votesRevealed).forEach((a) => {
      if (a.finalVote === "For") tally.for++;
      else if (a.finalVote === "Against") tally.against++;
      else tally.abstain++;
    });
  } else if (phase && phase !== "intro") {
    AGENTS.forEach((a) => {
      const pos = getPosition(a);
      if (pos === "For") tally.for++;
      else if (pos === "Against") tally.against++;
      else tally.abstain++;
    });
  }

  // ── Committee grouping ─────────────────────────────────────────────────────
  const committee0 = AGENTS.filter((a) => a.committee === 0);
  const committee1 = AGENTS.filter((a) => a.committee === 1);

  // ── Debate connections for Floor phase ─────────────────────────────────────
  const debateLinks = [
    [0, 1], // Rivera vs Chen
    [1, 2], // Chen vs Okafor
    [2, 3], // Okafor vs Walsh
    [3, 4], // Walsh vs Park
    [0, 4], // Rivera vs Park (allies)
  ];

  const isFinished = phaseIdx >= PHASES.length - 1 && votesRevealed >= AGENTS.length;
  const canAdvance = phaseIdx < PHASES.length - 1 && !transitioning;
  const notStarted = phaseIdx === -1;

  if (!mounted) return null;

  return (
    <div className="widget-container ch9">
      <div className="widget-label">Interactive &middot; Congressional Simulation</div>

      {/* ── Topic ───────────────────────────────────────────────────────── */}
      <div
        style={{
          padding: "10px 14px",
          background: `${C.accent}10`,
          border: `1px solid ${C.accent}30`,
          borderRadius: 8,
          marginBottom: 16,
          fontFamily: "monospace",
          fontSize: 13,
          color: C.text,
        }}
      >
        <span style={{ color: C.muted, fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>
          Topic Under Debate
        </span>
        <div style={{ marginTop: 4 }}>Should AI be regulated by federal law?</div>
      </div>

      {/* ── Phase progress bar ─────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: 2,
          marginBottom: 18,
          padding: "0 2px",
        }}
      >
        {PHASES.map((p, i) => {
          const isActive = i === phaseIdx;
          const isDone = i < phaseIdx;
          const bg = isActive
            ? `${C.accent}60`
            : isDone
              ? `${C.green}30`
              : "rgba(255,255,255,0.04)";
          const border = isActive
            ? C.accent
            : isDone
              ? `${C.green}60`
              : C.border;
          const textColor = isActive ? C.text : isDone ? C.green : C.muted;
          return (
            <div
              key={p.key}
              style={{
                flex: 1,
                textAlign: "center",
                padding: "6px 2px",
                background: bg,
                border: `1px solid ${border}`,
                borderRadius: 6,
                fontSize: 9,
                fontFamily: "monospace",
                color: textColor,
                fontWeight: isActive ? 700 : 400,
                transition: "all 0.4s ease",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {isActive && (
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 2,
                    background: C.accent,
                    borderRadius: "6px 6px 0 0",
                  }}
                />
              )}
              <div style={{ fontSize: 8, opacity: 0.6, marginBottom: 1 }}>{i + 1}</div>
              {p.short}
            </div>
          );
        })}
      </div>

      {/* ── Bill text ──────────────────────────────────────────────────── */}
      {phase && (
        <div
          style={{
            padding: "8px 12px",
            background: "rgba(255,255,255,0.02)",
            border: `1px solid ${C.border}`,
            borderRadius: 6,
            marginBottom: 16,
            fontSize: 11,
            fontFamily: "monospace",
            color: C.muted,
            lineHeight: 1.5,
            transition: "all 0.4s ease",
          }}
        >
          <span style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: 1, color: C.amber }}>
            Bill Text{phase === "amendments" || phase === "final" || phase === "voting" ? " (Amended)" : ""}
          </span>
          <div style={{ marginTop: 4 }}>{billText}</div>
        </div>
      )}

      {/* ── Not started state ──────────────────────────────────────────── */}
      {notStarted && (
        <div
          style={{
            textAlign: "center",
            padding: "40px 20px",
            color: C.muted,
            fontFamily: "monospace",
            fontSize: 13,
          }}
        >
          <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.3 }}>|||</div>
          <div>5 AI congress members are ready to deliberate.</div>
          <div style={{ fontSize: 11, marginTop: 4, opacity: 0.6 }}>
            Press &quot;Begin Session&quot; to start the 6-phase simulation.
          </div>
        </div>
      )}

      {/* ── Phase 1: Introduction ──────────────────────────────────────── */}
      {phase === "intro" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {AGENTS.map((agent, i) => {
            const visible = i < revealedAgents;
            return (
              <div
                key={agent.name}
                style={{
                  display: "flex",
                  gap: 12,
                  padding: "10px 12px",
                  background: visible ? "rgba(255,255,255,0.03)" : "transparent",
                  border: `1px solid ${visible ? agent.partyColor + "30" : "transparent"}`,
                  borderRadius: 8,
                  opacity: visible ? 1 : 0,
                  transform: visible ? "translateX(0)" : "translateX(-20px)",
                  transition: "all 0.4s ease",
                  alignItems: "flex-start",
                }}
              >
                {/* Agent avatar */}
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 8,
                    background: `${agent.partyColor}20`,
                    border: `1px solid ${agent.partyColor}50`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "monospace",
                    fontSize: 14,
                    fontWeight: 700,
                    color: agent.partyColor,
                    flexShrink: 0,
                  }}
                >
                  {agent.name.split(" ")[1][0]}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 600, color: C.text }}>
                      {agent.name}
                    </span>
                    <span
                      style={{
                        fontSize: 9,
                        fontFamily: "monospace",
                        padding: "1px 6px",
                        borderRadius: 3,
                        background: `${agent.partyColor}15`,
                        color: agent.partyColor,
                      }}
                    >
                      {agent.party}
                    </span>
                    <PositionBadge position={getPosition(agent)} />
                    <SentimentIcon sentiment={getSentiment(agent)} />
                  </div>
                  <div
                    style={{
                      marginTop: 4,
                      fontSize: 11,
                      fontFamily: "monospace",
                      color: C.muted,
                      lineHeight: 1.4,
                    }}
                  >
                    {agent.statements.intro}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Phase 2: Committee ─────────────────────────────────────────── */}
      {phase === "committee" && (
        <div style={{ display: "flex", gap: 12 }}>
          {[committee0, committee1].map((group, gi) => (
            <div
              key={gi}
              style={{
                flex: 1,
                padding: "10px",
                background: "rgba(255,255,255,0.02)",
                border: `1px solid ${C.border}`,
                borderRadius: 8,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontFamily: "monospace",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  color: C.accent,
                  marginBottom: 8,
                }}
              >
                Committee {gi === 0 ? "A" : "B"}
              </div>
              {group.map((agent) => (
                <div
                  key={agent.name}
                  style={{
                    padding: "8px",
                    marginBottom: 6,
                    background: `${agent.partyColor}08`,
                    border: `1px solid ${agent.partyColor}20`,
                    borderRadius: 6,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                    <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: agent.partyColor }}>
                      {agent.name}
                    </span>
                    <PositionBadge position={getPosition(agent)} />
                  </div>
                  <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, lineHeight: 1.4 }}>
                    {agent.statements.committee}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* ── Phase 3: Floor Debate ──────────────────────────────────────── */}
      {phase === "floor" && (
        <div>
          {/* Connection SVG */}
          <svg
            viewBox="0 0 500 60"
            width="100%"
            style={{ maxWidth: 500, height: 60, display: "block", margin: "0 auto 8px" }}
          >
            {debateLinks.map(([a, b], li) => {
              const x1 = 50 + a * 100;
              const x2 = 50 + b * 100;
              const isAlliance = (a === 0 && b === 4) || (a === 4 && b === 0);
              return (
                <line
                  key={li}
                  x1={x1}
                  y1={30}
                  x2={x2}
                  y2={30}
                  stroke={isAlliance ? C.green : C.rose}
                  strokeWidth={1}
                  strokeOpacity={0.3}
                  strokeDasharray={isAlliance ? "none" : "4 3"}
                />
              );
            })}
            {AGENTS.map((agent, i) => (
              <g key={agent.name}>
                <circle cx={50 + i * 100} cy={30} r={14} fill={`${agent.partyColor}25`} stroke={agent.partyColor} strokeWidth={1} />
                <text
                  x={50 + i * 100}
                  y={31}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={agent.partyColor}
                  fontSize={10}
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {agent.name.split(" ")[1][0]}
                </text>
              </g>
            ))}
          </svg>
          <div style={{ display: "flex", justifyContent: "center", gap: 16, marginBottom: 10 }}>
            <span style={{ fontSize: 9, fontFamily: "monospace", color: C.green }}>
              <span style={{ display: "inline-block", width: 16, height: 1, background: C.green, verticalAlign: "middle", marginRight: 4 }} />
              Alliance
            </span>
            <span style={{ fontSize: 9, fontFamily: "monospace", color: C.rose }}>
              <span style={{ display: "inline-block", width: 16, height: 1, background: C.rose, verticalAlign: "middle", marginRight: 4, borderTop: "1px dashed " + C.rose }} />
              Opposition
            </span>
          </div>
          {AGENTS.map((agent) => (
            <div
              key={agent.name}
              style={{
                padding: "8px 10px",
                marginBottom: 6,
                background: "rgba(255,255,255,0.02)",
                borderLeft: `3px solid ${agent.partyColor}`,
                borderRadius: "0 6px 6px 0",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: agent.partyColor }}>
                  {agent.name}
                </span>
                <PositionBadge position={getPosition(agent)} />
                <SentimentIcon sentiment={getSentiment(agent)} />
              </div>
              <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, lineHeight: 1.4 }}>
                {agent.statements.floor}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Phase 4: Amendments ────────────────────────────────────────── */}
      {phase === "amendments" && (
        <div>
          {AMENDMENTS.map((amend) => (
            <div
              key={amend.id}
              style={{
                padding: "10px 12px",
                marginBottom: 10,
                background: `${C.amber}08`,
                border: `1px solid ${C.amber}25`,
                borderRadius: 8,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: 10,
                    fontWeight: 700,
                    padding: "2px 8px",
                    background: `${C.amber}20`,
                    borderRadius: 4,
                    color: C.amber,
                  }}
                >
                  Amendment {amend.id}
                </span>
                <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: C.text }}>
                  {amend.title}
                </span>
              </div>
              <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, marginBottom: 6 }}>
                Proposed by {amend.proposer}: {amend.text}
              </div>
              {/* Mini vote bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div
                  style={{
                    flex: 1,
                    height: 6,
                    background: "rgba(255,255,255,0.05)",
                    borderRadius: 3,
                    overflow: "hidden",
                    display: "flex",
                  }}
                >
                  <div
                    style={{
                      width: `${(amend.votesFor / 5) * 100}%`,
                      height: "100%",
                      background: C.green,
                      borderRadius: "3px 0 0 3px",
                      transition: "width 0.6s ease",
                    }}
                  />
                  <div
                    style={{
                      width: `${(amend.votesAgainst / 5) * 100}%`,
                      height: "100%",
                      background: C.rose,
                      borderRadius: "0 3px 3px 0",
                    }}
                  />
                </div>
                <span style={{ fontFamily: "monospace", fontSize: 9, color: C.green }}>
                  {amend.votesFor} For
                </span>
                <span style={{ fontFamily: "monospace", fontSize: 9, color: C.rose }}>
                  {amend.votesAgainst} Against
                </span>
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: 9,
                    fontWeight: 700,
                    color: amend.votesFor > amend.votesAgainst ? C.green : C.rose,
                  }}
                >
                  {amend.votesFor > amend.votesAgainst ? "PASSED" : "FAILED"}
                </span>
              </div>
            </div>
          ))}
          {/* Agent reactions */}
          <div style={{ marginTop: 6 }}>
            {AGENTS.map((agent) => (
              <div
                key={agent.name}
                style={{
                  padding: "6px 10px",
                  marginBottom: 4,
                  fontSize: 10,
                  fontFamily: "monospace",
                  color: C.muted,
                  borderLeft: `2px solid ${agent.partyColor}40`,
                }}
              >
                <span style={{ color: agent.partyColor, fontWeight: 600 }}>{agent.name}:</span>{" "}
                {agent.statements.amendments}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Phase 5: Final Arguments ───────────────────────────────────── */}
      {phase === "final" && (
        <div>
          {AGENTS.map((agent) => {
            const conv = getConviction(agent);
            return (
              <div
                key={agent.name}
                style={{
                  padding: "10px 12px",
                  marginBottom: 8,
                  background: "rgba(255,255,255,0.02)",
                  border: `1px solid ${agent.partyColor}20`,
                  borderRadius: 8,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: agent.partyColor }}>
                    {agent.name}
                  </span>
                  <PositionBadge position={getPosition(agent)} />
                  <SentimentIcon sentiment={getSentiment(agent)} />
                </div>
                {/* Conviction bar */}
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 9, fontFamily: "monospace", color: C.muted, width: 60 }}>
                    Conviction
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: 6,
                      background: "rgba(255,255,255,0.05)",
                      borderRadius: 3,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${conv * 100}%`,
                        height: "100%",
                        background: agent.partyColor,
                        borderRadius: 3,
                        transition: "width 0.8s ease",
                      }}
                    />
                  </div>
                  <span style={{ fontSize: 9, fontFamily: "monospace", color: agent.partyColor, width: 30, textAlign: "right" }}>
                    {Math.round(conv * 100)}%
                  </span>
                </div>
                <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, lineHeight: 1.4, fontStyle: "italic" }}>
                  &ldquo;{agent.statements.final}&rdquo;
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Phase 6: Voting ────────────────────────────────────────────── */}
      {phase === "voting" && (
        <div>
          {AGENTS.map((agent, i) => {
            const revealed = i < votesRevealed;
            const vote = agent.finalVote;
            const voteColor = vote === "For" ? C.green : vote === "Against" ? C.rose : C.amber;
            return (
              <div
                key={agent.name}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "10px 12px",
                  marginBottom: 6,
                  background: revealed ? `${voteColor}08` : "rgba(255,255,255,0.02)",
                  border: `1px solid ${revealed ? voteColor + "30" : C.border}`,
                  borderRadius: 8,
                  transition: "all 0.5s ease",
                }}
              >
                {/* Avatar */}
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 6,
                    background: revealed ? `${voteColor}20` : "rgba(255,255,255,0.04)",
                    border: `1px solid ${revealed ? voteColor : C.border}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "monospace",
                    fontSize: 13,
                    fontWeight: 700,
                    color: revealed ? voteColor : C.muted,
                    flexShrink: 0,
                    transition: "all 0.5s ease",
                  }}
                >
                  {agent.name.split(" ")[1][0]}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: C.text }}>
                      {agent.name}
                    </span>
                    <span style={{ fontSize: 9, fontFamily: "monospace", color: agent.partyColor }}>
                      {agent.party}
                    </span>
                  </div>
                  {revealed && (
                    <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, marginTop: 2 }}>
                      {agent.statements.voting}
                    </div>
                  )}
                </div>
                {/* Vote result */}
                <div
                  style={{
                    padding: "4px 14px",
                    borderRadius: 6,
                    fontFamily: "monospace",
                    fontSize: 12,
                    fontWeight: 700,
                    background: revealed ? `${voteColor}20` : "rgba(255,255,255,0.04)",
                    color: revealed ? voteColor : C.muted,
                    border: `1px solid ${revealed ? voteColor + "40" : C.border}`,
                    transition: "all 0.5s ease",
                    minWidth: 60,
                    textAlign: "center",
                  }}
                >
                  {revealed ? vote.toUpperCase() : "???"}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Vote tally ─────────────────────────────────────────────────── */}
      {phase && (
        <div
          style={{
            marginTop: 16,
            padding: "10px 14px",
            background: "rgba(255,255,255,0.02)",
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 1, color: C.muted }}>
            {phase === "voting" ? "Final Tally" : "Current Positions"}
          </span>
          <div style={{ display: "flex", gap: 16 }}>
            <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600, color: C.green }}>
              For: {tally.for}
            </span>
            <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600, color: C.rose }}>
              Against: {tally.against}
            </span>
            <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600, color: C.amber }}>
              Abstain: {tally.abstain}
            </span>
          </div>
          {/* Result banner */}
          {isFinished && (
            <span
              style={{
                fontFamily: "monospace",
                fontSize: 11,
                fontWeight: 700,
                padding: "3px 10px",
                borderRadius: 4,
                background: tally.for > tally.against ? `${C.green}20` : `${C.rose}20`,
                color: tally.for > tally.against ? C.green : C.rose,
                border: `1px solid ${tally.for > tally.against ? C.green : C.rose}40`,
              }}
            >
              {tally.for > tally.against ? "BILL PASSED" : "BILL FAILED"}
            </span>
          )}
        </div>
      )}

      {/* ── Controls ───────────────────────────────────────────────────── */}
      <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 10 }}>
        {notStarted ? (
          <button
            className="btn-mono"
            onClick={advancePhase}
            style={{ fontSize: 13, padding: "7px 20px", cursor: "pointer" }}
          >
            Begin Session
          </button>
        ) : canAdvance ? (
          <button
            className="btn-mono"
            onClick={advancePhase}
            disabled={transitioning}
            style={{
              fontSize: 13,
              padding: "7px 20px",
              cursor: transitioning ? "not-allowed" : "pointer",
              opacity: transitioning ? 0.5 : 1,
            }}
          >
            {transitioning ? "Transitioning..." : `Next Phase: ${PHASES[phaseIdx + 1].label}`}
          </button>
        ) : isFinished ? (
          <button
            className="btn-mono"
            onClick={reset}
            style={{ fontSize: 13, padding: "7px 20px", cursor: "pointer" }}
          >
            Reset Simulation
          </button>
        ) : (
          <span style={{ fontFamily: "monospace", fontSize: 11, color: C.muted }}>
            Revealing votes...
          </span>
        )}
        {phase && (
          <span style={{ fontFamily: "monospace", fontSize: 11, color: C.muted }}>
            Phase {phaseIdx + 1} of {PHASES.length}: {PHASES[phaseIdx].label}
          </span>
        )}
      </div>
    </div>
  );
}
