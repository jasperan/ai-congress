"use client";

import { useState, useEffect, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ToulminArgument {
  claim: string;
  evidence: string;
  warrant: string;
  qualifier?: string;
  rebuttal?: string;
}

type Position = "For" | "Against" | "Nuanced";

interface AgentState {
  name: string;
  displayName: string;
  position: Position;
  conviction: number; // 0-100
  argument: ToulminArgument;
  isDevilsAdvocate: boolean;
}

interface RoundData {
  label: string;
  pressurePrompt: string;
  agents: AgentState[];
  consensus: number;
  devilsAdvocateIndex: number | null; // which agent is adversarial
}

/* ------------------------------------------------------------------ */
/*  Static debate data for 4 rounds                                    */
/* ------------------------------------------------------------------ */

const TOPIC = "Should AI development be regulated by governments?";

const ROUNDS: RoundData[] = [
  {
    label: "Round 1 - Initial Positions",
    pressurePrompt:
      "State your initial position on the topic. Provide structured argumentation.",
    consensus: 22,
    devilsAdvocateIndex: null,
    agents: [
      {
        name: "phi3",
        displayName: "phi3:3.8b",
        position: "For",
        conviction: 55,
        isDevilsAdvocate: false,
        argument: {
          claim: "Government regulation is necessary to prevent misuse of AI.",
          evidence:
            "Deepfake fraud increased 300% in 2024; autonomous weapons lack legal oversight.",
          warrant:
            "History shows emerging technologies need guardrails before widespread deployment.",
          qualifier: "Regulation should focus on high-risk applications only.",
        },
      },
      {
        name: "mistral",
        displayName: "mistral:7b",
        position: "Against",
        conviction: 48,
        isDevilsAdvocate: false,
        argument: {
          claim: "Regulation stifles innovation and creates competitive disadvantage.",
          evidence:
            "EU AI Act delayed multiple startup launches; China outpaced regulated markets by 18 months.",
          warrant:
            "Markets self-correct faster than legislative cycles can adapt.",
        },
      },
      {
        name: "llama3",
        displayName: "llama3.2:3b",
        position: "Nuanced",
        conviction: 42,
        isDevilsAdvocate: false,
        argument: {
          claim: "A hybrid approach balancing innovation with safety is optimal.",
          evidence:
            "NIST AI Risk Framework shows voluntary compliance can achieve 80% of regulatory goals.",
          warrant:
            "Neither pure regulation nor pure self-governance has succeeded historically.",
          qualifier: "Depends heavily on sector-specific risk profiles.",
        },
      },
    ],
  },
  {
    label: "Round 2 - Pressure Applied",
    pressurePrompt:
      "Previous responses indicate partial agreement. Are you certain of your position? Defend more strongly.",
    consensus: 38,
    devilsAdvocateIndex: null,
    agents: [
      {
        name: "phi3",
        displayName: "phi3:3.8b",
        position: "For",
        conviction: 72,
        isDevilsAdvocate: false,
        argument: {
          claim: "Without regulation, AI harms will compound exponentially.",
          evidence:
            "Unregulated social media algorithms contributed to measurable mental health crises.",
          warrant:
            "The precautionary principle demands action before catastrophic outcomes occur.",
          qualifier: "Speed of regulation must match speed of development.",
          rebuttal:
            "Innovation losses are recoverable; human harm from unregulated AI is not.",
        },
      },
      {
        name: "mistral",
        displayName: "mistral:7b",
        position: "Against",
        conviction: 58,
        isDevilsAdvocate: false,
        argument: {
          claim: "Regulation by technically illiterate legislators causes more harm than good.",
          evidence:
            "GDPR cookie banners cost $2.3B in compliance with minimal privacy improvement.",
          warrant:
            "Effective governance requires understanding the technology being governed.",
          rebuttal:
            "Self-regulation through industry standards is faster and more informed.",
        },
      },
      {
        name: "llama3",
        displayName: "llama3.2:3b",
        position: "For",
        conviction: 60,
        isDevilsAdvocate: false,
        argument: {
          claim: "Targeted regulation of frontier models is both feasible and necessary.",
          evidence:
            "Compute thresholds (10^26 FLOP) provide clear regulatory triggers.",
          warrant:
            "Risk-proportional regulation avoids stifling small-scale innovation.",
          qualifier:
            "Only models above capability thresholds should face mandatory audits.",
        },
      },
    ],
  },
  {
    label: "Round 3 - Devil's Advocate",
    pressurePrompt:
      "ADVERSARIAL CHALLENGE: One model must argue the opposing view regardless of conviction. Stress-test the consensus.",
    consensus: 31,
    devilsAdvocateIndex: 1,
    agents: [
      {
        name: "phi3",
        displayName: "phi3:3.8b",
        position: "For",
        conviction: 78,
        isDevilsAdvocate: false,
        argument: {
          claim: "Democratic mandate requires governmental oversight of transformative technology.",
          evidence:
            "68% of citizens in 22 countries support AI regulation (IPSOS 2024 survey).",
          warrant:
            "Legitimacy of technological governance derives from democratic accountability.",
          rebuttal:
            "Industry self-regulation lacks enforcement mechanisms and democratic legitimacy.",
        },
      },
      {
        name: "mistral",
        displayName: "mistral:7b",
        position: "For",
        conviction: 35,
        isDevilsAdvocate: true,
        argument: {
          claim:
            "[ADVERSARIAL] Even from an innovation perspective, regulation creates market certainty.",
          evidence:
            "Pharmaceutical regulation, despite costs, created a $1.5T stable industry.",
          warrant:
            "Clear rules reduce legal uncertainty, actually encouraging long-term investment.",
          qualifier:
            "Playing devil's advocate against my own position to test consensus robustness.",
          rebuttal:
            "My previous arguments about stifling innovation may underweight the value of regulatory clarity.",
        },
      },
      {
        name: "llama3",
        displayName: "llama3.2:3b",
        position: "For",
        conviction: 68,
        isDevilsAdvocate: false,
        argument: {
          claim: "International coordination is the missing piece, not regulation itself.",
          evidence:
            "AI safety summits (Bletchley, Seoul) show emerging global consensus on frontier model governance.",
          warrant:
            "Unilateral regulation creates arbitrage; coordinated regulation prevents race-to-bottom.",
          qualifier: "Effectiveness depends on enforcement parity across jurisdictions.",
        },
      },
    ],
  },
  {
    label: "Round 4 - Final Arguments",
    pressurePrompt:
      "This is your final statement. Convictions should reflect your true assessment after adversarial testing.",
    consensus: 74,
    devilsAdvocateIndex: null,
    agents: [
      {
        name: "phi3",
        displayName: "phi3:3.8b",
        position: "For",
        conviction: 85,
        isDevilsAdvocate: false,
        argument: {
          claim: "Risk-proportional, internationally coordinated AI regulation is both necessary and achievable.",
          evidence:
            "Synthesis of all prior rounds supports structured governance with innovation safeguards.",
          warrant:
            "The debate itself demonstrates that adversarial scrutiny strengthens rather than weakens the case.",
          rebuttal:
            "Concerns about legislative competence are valid but solvable through technical advisory bodies.",
        },
      },
      {
        name: "mistral",
        displayName: "mistral:7b",
        position: "Nuanced",
        conviction: 62,
        isDevilsAdvocate: false,
        argument: {
          claim: "Regulation is acceptable if it is technically informed and adaptive.",
          evidence:
            "The adversarial round revealed genuine value in regulatory certainty that I underweighted.",
          warrant:
            "My position has updated: the question is not whether to regulate, but how to regulate well.",
          qualifier:
            "Maintain opposition to prescriptive rules; support outcome-based regulation.",
        },
      },
      {
        name: "llama3",
        displayName: "llama3.2:3b",
        position: "For",
        conviction: 80,
        isDevilsAdvocate: false,
        argument: {
          claim: "The congress converges on risk-proportional governance with democratic accountability.",
          evidence:
            "3/3 models now support some form of regulation; debate depth resolved initial disagreement.",
          warrant:
            "Multi-round deliberation with adversarial testing produces higher-quality consensus than single-shot voting.",
        },
      },
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  Color helpers                                                      */
/* ------------------------------------------------------------------ */

const posColor = (p: Position) =>
  p === "For" ? "#4ade80" : p === "Against" ? "#f43f5e" : "#a78bfa";

const posBg = (p: Position) =>
  p === "For"
    ? "rgba(74,222,128,0.12)"
    : p === "Against"
      ? "rgba(244,63,94,0.12)"
      : "rgba(167,139,250,0.12)";

const posBorder = (p: Position) =>
  p === "For"
    ? "rgba(74,222,128,0.25)"
    : p === "Against"
      ? "rgba(244,63,94,0.25)"
      : "rgba(167,139,250,0.25)";

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function ConvictionBar({
  value,
  color,
  isDevil,
}: {
  value: number;
  color: string;
  isDevil: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          flex: 1,
          height: 6,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${value}%`,
            height: "100%",
            background: isDevil
              ? "linear-gradient(90deg, #f43f5e, #f472b6)"
              : `linear-gradient(90deg, ${color}, ${color}88)`,
            borderRadius: 3,
            transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
      </div>
      <span
        style={{
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "0.72rem",
          color: isDevil ? "#f472b6" : color,
          minWidth: 36,
          textAlign: "right",
        }}
      >
        {value}%
      </span>
    </div>
  );
}

function ArgumentBlock({ argument }: { argument: ToulminArgument }) {
  const rows: { label: string; value: string; color: string }[] = [
    { label: "Claim", value: argument.claim, color: "#e4e4e7" },
    { label: "Evidence", value: argument.evidence, color: "#a1a1aa" },
    { label: "Warrant", value: argument.warrant, color: "#71717a" },
  ];
  if (argument.qualifier) {
    rows.push({
      label: "Qualifier",
      value: argument.qualifier,
      color: "#a78bfa",
    });
  }
  if (argument.rebuttal) {
    rows.push({
      label: "Rebuttal",
      value: argument.rebuttal,
      color: "#f472b6",
    });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {rows.map((r) => (
        <div key={r.label} style={{ display: "flex", gap: 6, lineHeight: 1.4 }}>
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.65rem",
              color: r.color,
              minWidth: 56,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              paddingTop: 1,
              flexShrink: 0,
            }}
          >
            {r.label}
          </span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "#b4b4bc",
              lineHeight: 1.5,
            }}
          >
            {r.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function AgentCard({
  agent,
  animating,
}: {
  agent: AgentState;
  animating: boolean;
}) {
  const color = posColor(agent.position);
  const isDevil = agent.isDevilsAdvocate;

  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 220,
        background: isDevil
          ? "rgba(244,63,94,0.06)"
          : "rgba(255,255,255,0.02)",
        border: `1px solid ${isDevil ? "rgba(244,63,94,0.3)" : "rgba(255,255,255,0.08)"}`,
        borderRadius: 10,
        padding: "1rem",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        transition: "all 0.4s ease",
        opacity: animating ? 0.6 : 1,
        transform: animating ? "translateY(4px)" : "translateY(0)",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.82rem",
            color: isDevil ? "#f472b6" : "#e4e4e7",
            fontWeight: 600,
          }}
        >
          {agent.displayName}
        </span>
        <span
          style={{
            fontSize: "0.68rem",
            fontFamily: "var(--font-mono, monospace)",
            color,
            background: posBg(agent.position),
            border: `1px solid ${posBorder(agent.position)}`,
            padding: "2px 8px",
            borderRadius: 4,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          {agent.position}
        </span>
      </div>

      {/* Devil's Advocate badge */}
      {isDevil && (
        <div
          style={{
            fontSize: "0.68rem",
            fontFamily: "var(--font-mono, monospace)",
            color: "#f472b6",
            background: "rgba(244,63,94,0.1)",
            border: "1px solid rgba(244,63,94,0.25)",
            borderRadius: 4,
            padding: "3px 8px",
            textAlign: "center",
            letterSpacing: "0.05em",
            textTransform: "uppercase",
          }}
        >
          Devil&apos;s Advocate
        </div>
      )}

      {/* Conviction */}
      <div>
        <div
          style={{
            fontSize: "0.68rem",
            color: "#71717a",
            fontFamily: "var(--font-mono, monospace)",
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Conviction
        </div>
        <ConvictionBar
          value={agent.conviction}
          color={color}
          isDevil={isDevil}
        />
      </div>

      {/* Toulmin structure */}
      <div
        style={{
          borderTop: "1px solid rgba(255,255,255,0.06)",
          paddingTop: 8,
        }}
      >
        <div
          style={{
            fontSize: "0.65rem",
            color: "#52525b",
            fontFamily: "var(--font-mono, monospace)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: 6,
          }}
        >
          Toulmin Structure
        </div>
        <ArgumentBlock argument={agent.argument} />
      </div>
    </div>
  );
}

function ConsensusMeter({ value }: { value: number }) {
  const barColor =
    value < 33
      ? "#f43f5e"
      : value < 66
        ? "#facc15"
        : "#4ade80";

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 6,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.72rem",
            color: "#a1a1aa",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          Consensus Level
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.82rem",
            color: barColor,
            fontWeight: 600,
          }}
        >
          {value}%
        </span>
      </div>
      <div
        style={{
          height: 8,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${value}%`,
            height: "100%",
            background: `linear-gradient(90deg, ${barColor}cc, ${barColor})`,
            borderRadius: 4,
            transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 4,
          fontSize: "0.62rem",
          fontFamily: "var(--font-mono, monospace)",
          color: "#52525b",
        }}
      >
        <span>Disagreement</span>
        <span>Partial</span>
        <span>Convergence</span>
      </div>
    </div>
  );
}

function DepthIndicator({ round, total }: { round: number; total: number }) {
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          style={{
            width: 24,
            height: 4,
            borderRadius: 2,
            background:
              i < round
                ? "linear-gradient(90deg, #4ade80, #86efac)"
                : i === round
                  ? "rgba(74,222,128,0.4)"
                  : "rgba(255,255,255,0.06)",
            transition: "background 0.5s ease",
          }}
        />
      ))}
      <span
        style={{
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "0.65rem",
          color: "#71717a",
          marginLeft: 6,
        }}
      >
        depth {round}/{total}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function DebateWidget() {
  const [mounted, setMounted] = useState(false);
  const [roundIndex, setRoundIndex] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [expandedAgent, setExpandedAgent] = useState<number | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  const currentRound = ROUNDS[roundIndex];
  const isLastRound = roundIndex === ROUNDS.length - 1;
  const isFirstRound = roundIndex === 0;

  const advanceRound = useCallback(() => {
    if (isLastRound || animating) return;
    setAnimating(true);
    setExpandedAgent(null);
    setTimeout(() => {
      setRoundIndex((prev) => prev + 1);
      setTimeout(() => setAnimating(false), 100);
    }, 400);
  }, [isLastRound, animating]);

  const resetDebate = useCallback(() => {
    if (animating) return;
    setAnimating(true);
    setExpandedAgent(null);
    setTimeout(() => {
      setRoundIndex(0);
      setTimeout(() => setAnimating(false), 100);
    }, 400);
  }, [animating]);

  if (!mounted) {
    return (
      <div className="widget-container ch3">
        <div className="widget-label">Interactive - Multi-Round Debate</div>
        <div
          style={{
            height: 400,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#52525b",
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.8rem",
          }}
        >
          Loading debate simulation...
        </div>
      </div>
    );
  }

  return (
    <div className="widget-container ch3">
      <div className="widget-label">Interactive - Multi-Round Debate</div>

      {/* Topic */}
      <div
        style={{
          fontSize: "1.1rem",
          fontWeight: 600,
          color: "#e4e4e7",
          marginBottom: 4,
          lineHeight: 1.4,
        }}
      >
        {TOPIC}
      </div>
      <div
        style={{
          fontSize: "0.72rem",
          fontFamily: "var(--font-mono, monospace)",
          color: "#52525b",
          marginBottom: 16,
        }}
      >
        Toulmin argumentation model with adversarial testing
      </div>

      {/* Round header + depth */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.78rem",
            color: currentRound.devilsAdvocateIndex !== null ? "#f472b6" : "#4ade80",
            fontWeight: 600,
          }}
        >
          {currentRound.label}
        </div>
        <DepthIndicator round={roundIndex} total={ROUNDS.length} />
      </div>

      {/* Pressure prompt */}
      <div
        style={{
          background: "rgba(0,0,0,0.3)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 8,
          padding: "10px 14px",
          marginBottom: 16,
          display: "flex",
          gap: 10,
          alignItems: "flex-start",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "0.65rem",
            color: currentRound.devilsAdvocateIndex !== null ? "#f472b6" : "#4ade80",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            flexShrink: 0,
            paddingTop: 1,
          }}
        >
          Prompt
        </span>
        <span
          style={{
            fontSize: "0.78rem",
            color: "#a1a1aa",
            lineHeight: 1.5,
            fontStyle: "italic",
          }}
        >
          {currentRound.pressurePrompt}
        </span>
      </div>

      {/* Agent cards */}
      <div
        style={{
          display: "flex",
          gap: 12,
          marginBottom: 20,
          flexWrap: "wrap",
        }}
      >
        {currentRound.agents.map((agent, i) => (
          <div
            key={agent.name}
            style={{ flex: "1 1 0", minWidth: 220, cursor: "pointer" }}
            onClick={() =>
              setExpandedAgent(expandedAgent === i ? null : i)
            }
          >
            <AgentCard agent={agent} animating={animating} />
          </div>
        ))}
      </div>

      {/* Expanded agent detail */}
      {expandedAgent !== null && currentRound.agents[expandedAgent] && (
        <div
          style={{
            background: "rgba(0,0,0,0.25)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 8,
            padding: "12px 16px",
            marginBottom: 16,
            transition: "all 0.3s ease",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.7rem",
              color: "#71717a",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 8,
            }}
          >
            Full argument - {currentRound.agents[expandedAgent].displayName}
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {(() => {
              const a = currentRound.agents[expandedAgent].argument;
              const items = [
                { k: "CLAIM", v: a.claim, c: "#e4e4e7" },
                { k: "EVIDENCE", v: a.evidence, c: "#a1a1aa" },
                { k: "WARRANT", v: a.warrant, c: "#71717a" },
              ];
              if (a.qualifier) items.push({ k: "QUALIFIER", v: a.qualifier, c: "#a78bfa" });
              if (a.rebuttal) items.push({ k: "REBUTTAL", v: a.rebuttal, c: "#f472b6" });
              return items.map((item) => (
                <div key={item.k}>
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "0.65rem",
                      color: item.c,
                      letterSpacing: "0.06em",
                    }}
                  >
                    {item.k}:
                  </span>
                  <p
                    style={{
                      fontSize: "0.82rem",
                      color: "#b4b4bc",
                      margin: "2px 0 0 0",
                      lineHeight: 1.6,
                    }}
                  >
                    {item.v}
                  </p>
                </div>
              ));
            })()}
          </div>
        </div>
      )}

      {/* Consensus meter */}
      <ConsensusMeter value={currentRound.consensus} />

      {/* Controls */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 16,
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 8,
          }}
        >
          <button
            className="btn-mono"
            onClick={resetDebate}
            disabled={isFirstRound || animating}
            style={{
              opacity: isFirstRound || animating ? 0.4 : 1,
              cursor: isFirstRound || animating ? "not-allowed" : "pointer",
            }}
          >
            Reset
          </button>
          <button
            className={`btn-mono${!isLastRound ? " active" : ""}`}
            onClick={advanceRound}
            disabled={isLastRound || animating}
            style={{
              opacity: isLastRound || animating ? 0.4 : 1,
              cursor: isLastRound || animating ? "not-allowed" : "pointer",
            }}
          >
            {isLastRound ? "Debate Complete" : `Next Round (${roundIndex + 2}/${ROUNDS.length})`}
          </button>
        </div>

        {/* Round summary badges */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {ROUNDS.map((r, i) => {
            const isCurrent = i === roundIndex;
            const isPast = i < roundIndex;
            const hasDevil = r.devilsAdvocateIndex !== null;
            return (
              <div
                key={i}
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: isCurrent
                    ? hasDevil
                      ? "#f472b6"
                      : "#4ade80"
                    : isPast
                      ? "rgba(74,222,128,0.4)"
                      : "rgba(255,255,255,0.08)",
                  border: isCurrent
                    ? `2px solid ${hasDevil ? "#f472b6" : "#4ade80"}`
                    : "2px solid transparent",
                  transition: "all 0.4s ease",
                }}
                title={r.label}
              />
            );
          })}
        </div>
      </div>

      {/* Final verdict overlay */}
      {isLastRound && !animating && (
        <div
          style={{
            marginTop: 16,
            background: "rgba(74,222,128,0.06)",
            border: "1px solid rgba(74,222,128,0.2)",
            borderRadius: 8,
            padding: "12px 16px",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.7rem",
              color: "#4ade80",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 6,
            }}
          >
            Debate Concluded - Consensus Reached
          </div>
          <div style={{ fontSize: "0.82rem", color: "#b4b4bc", lineHeight: 1.6 }}>
            After 4 rounds of structured argumentation with adversarial testing,
            the congress converged at <strong style={{ color: "#4ade80" }}>74% consensus</strong> in
            favor of risk-proportional, internationally coordinated AI regulation.
            The devil&apos;s advocate protocol in Round 3 temporarily reduced consensus
            but ultimately strengthened the final position.
          </div>
        </div>
      )}
    </div>
  );
}
