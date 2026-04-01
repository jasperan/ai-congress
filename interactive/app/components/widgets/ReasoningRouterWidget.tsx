"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ---------- Types ----------

type ReasoningMode = "direct" | "cot" | "react";
type Domain = "math" | "coding" | "science" | "general";

interface PresetQuery {
  text: string;
  mode: ReasoningMode;
  domain: Domain;
  domainLabel: string;
}

interface ReasoningStep {
  label: string;
  content: string;
}

interface AnimationState {
  phase:
    | "idle"
    | "classifying"
    | "domain-reveal"
    | "routing"
    | "reasoning"
    | "done";
  domain: Domain | null;
  mode: ReasoningMode | null;
  steps: ReasoningStep[];
  visibleSteps: number;
  typedChars: number;
}

// ---------- Data ----------

const PRESETS: PresetQuery[] = [
  {
    text: "What is the capital of France?",
    mode: "direct",
    domain: "general",
    domainLabel: "General Knowledge",
  },
  {
    text: "If a train leaves at 3pm going 60mph and another at 4pm going 90mph, when do they meet?",
    mode: "cot",
    domain: "general",
    domainLabel: "Multi-step Reasoning",
  },
  {
    text: "What is 847 * 293?",
    mode: "react",
    domain: "math",
    domainLabel: "Mathematics",
  },
  {
    text: "What is the latest news about AI?",
    mode: "react",
    domain: "science",
    domainLabel: "Science / Current Events",
  },
];

const REASONING_DATA: Record<
  string,
  { steps: ReasoningStep[]; answer: string }
> = {
  direct_general: {
    steps: [{ label: "Answer", content: "The capital of France is Paris." }],
    answer: "Paris",
  },
  cot_general: {
    steps: [
      {
        label: "Step 1",
        content:
          "Train A departs at 3:00 PM traveling at 60 mph. Train B departs at 4:00 PM at 90 mph.",
      },
      {
        label: "Step 2",
        content:
          "By 4:00 PM, Train A has a 1-hour head start = 60 miles ahead.",
      },
      {
        label: "Step 3",
        content:
          "Relative closing speed: 90 - 60 = 30 mph. Time to close 60 miles: 60 / 30 = 2 hours.",
      },
      {
        label: "Step 4",
        content:
          "They meet at 4:00 PM + 2 hours = 6:00 PM, at mile marker 180.",
      },
    ],
    answer: "6:00 PM",
  },
  react_math: {
    steps: [
      {
        label: "Thought",
        content: "I need to multiply 847 by 293. Let me use the calculator tool.",
      },
      {
        label: "Action",
        content: "calculate(847 * 293)",
      },
      {
        label: "Observation",
        content: "Result: 248,171",
      },
      {
        label: "Answer",
        content: "847 * 293 = 248,171",
      },
    ],
    answer: "248,171",
  },
  react_science: {
    steps: [
      {
        label: "Thought",
        content:
          "The user wants current news about AI. I need to search the web for recent information.",
      },
      {
        label: "Action",
        content: 'web_search("latest AI news 2026")',
      },
      {
        label: "Observation",
        content:
          "Found 12 results: top stories about new model releases, regulation updates, and industry partnerships.",
      },
      {
        label: "Thought",
        content:
          "Let me synthesize the top results into a concise summary for the user.",
      },
      {
        label: "Answer",
        content:
          "Here are the latest AI developments: new frontier models, updated EU AI Act provisions, and major industry collaborations.",
      },
    ],
    answer: "Latest AI news synthesized from web search",
  },
};

// ---------- Colors ----------

const MODE_COLORS: Record<ReasoningMode, string> = {
  direct: "#facc15",
  cot: "#a78bfa",
  react: "#22d3ee",
};

const STEP_LABEL_COLORS: Record<string, string> = {
  Thought: "#22d3ee",
  Action: "#facc15",
  Observation: "#4ade80",
  Answer: "#a78bfa",
  "Step 1": "#a78bfa",
  "Step 2": "#a78bfa",
  "Step 3": "#a78bfa",
  "Step 4": "#a78bfa",
};

// ---------- Helpers ----------

function classifyQuery(text: string): { mode: ReasoningMode; domain: Domain; domainLabel: string } {
  const lower = text.toLowerCase();
  const preset = PRESETS.find(
    (p) => lower.includes(p.text.toLowerCase().slice(0, 20))
  );
  if (preset) return { mode: preset.mode, domain: preset.domain, domainLabel: preset.domainLabel };

  if (/\d+\s*[\*\+\-\/\^]\s*\d+/.test(text) || /calc|math|multiply|divide|sum/i.test(text))
    return { mode: "react", domain: "math", domainLabel: "Mathematics" };
  if (/news|latest|current|search|find|look up/i.test(text))
    return { mode: "react", domain: "science", domainLabel: "Science / Current Events" };
  if (/code|function|bug|program|debug|implement/i.test(text))
    return { mode: "cot", domain: "coding", domainLabel: "Code Analysis" };
  if (/why|how|explain|reason|if.*then|step/i.test(text))
    return { mode: "cot", domain: "general", domainLabel: "Multi-step Reasoning" };

  return { mode: "direct", domain: "general", domainLabel: "General Knowledge" };
}

function getReasoningSteps(mode: ReasoningMode, domain: Domain): ReasoningStep[] {
  const key = `${mode}_${domain}`;
  if (REASONING_DATA[key]) return REASONING_DATA[key].steps;
  if (mode === "direct")
    return [{ label: "Answer", content: "Response generated directly from knowledge." }];
  if (mode === "cot")
    return [
      { label: "Step 1", content: "Analyze the query components." },
      { label: "Step 2", content: "Apply relevant reasoning framework." },
      { label: "Step 3", content: "Synthesize conclusion from analysis." },
    ];
  return [
    { label: "Thought", content: "I need a tool to answer this accurately." },
    { label: "Action", content: "search(query)" },
    { label: "Observation", content: "Retrieved relevant data from tool." },
    { label: "Answer", content: "Here is the synthesized response." },
  ];
}

// ---------- Component ----------

export default function ReasoningRouterWidget() {
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState("");
  const [anim, setAnim] = useState<AnimationState>({
    phase: "idle",
    domain: null,
    mode: null,
    steps: [],
    visibleSteps: 0,
    typedChars: 0,
  });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    setMounted(true);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const clearTimers = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Typewriter effect for current step
  useEffect(() => {
    if (anim.phase !== "reasoning") return;
    const currentStepIdx = anim.visibleSteps;
    if (currentStepIdx >= anim.steps.length) {
      timerRef.current = setTimeout(() => {
        setAnim((prev) => ({ ...prev, phase: "done" }));
      }, 400);
      return;
    }
    const stepContent = anim.steps[currentStepIdx].content;
    if (anim.typedChars < stepContent.length) {
      timerRef.current = setTimeout(() => {
        setAnim((prev) => ({ ...prev, typedChars: prev.typedChars + 1 }));
      }, 18);
    } else {
      timerRef.current = setTimeout(() => {
        setAnim((prev) => ({
          ...prev,
          visibleSteps: prev.visibleSteps + 1,
          typedChars: 0,
        }));
      }, 350);
    }
    return () => clearTimers();
  }, [anim.phase, anim.visibleSteps, anim.typedChars, anim.steps, clearTimers]);

  const runRouting = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      clearTimers();

      const classification = classifyQuery(text);
      const steps = getReasoningSteps(classification.mode, classification.domain);

      setAnim({
        phase: "classifying",
        domain: null,
        mode: null,
        steps,
        visibleSteps: 0,
        typedChars: 0,
      });

      timerRef.current = setTimeout(() => {
        setAnim((prev) => ({
          ...prev,
          phase: "domain-reveal",
          domain: classification.domain,
        }));
        timerRef.current = setTimeout(() => {
          setAnim((prev) => ({
            ...prev,
            phase: "routing",
            mode: classification.mode,
          }));
          timerRef.current = setTimeout(() => {
            setAnim((prev) => ({ ...prev, phase: "reasoning" }));
          }, 700);
        }, 700);
      }, 800);
    },
    [clearTimers]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runRouting(query);
  };

  const handlePreset = (preset: PresetQuery) => {
    setQuery(preset.text);
    runRouting(preset.text);
  };

  const reset = () => {
    clearTimers();
    setAnim({
      phase: "idle",
      domain: null,
      mode: null,
      steps: [],
      visibleSteps: 0,
      typedChars: 0,
    });
  };

  if (!mounted) return null;

  // ---------- SVG Layout Constants ----------
  const W = 680;
  const H = 320;
  const classifierBox = { x: 260, y: 10, w: 160, h: 50 };
  const modeBoxes: Record<ReasoningMode, { x: number; y: number; w: number; h: number }> = {
    direct: { x: 40, y: 220, w: 140, h: 50 },
    cot: { x: 270, y: 220, w: 140, h: 50 },
    react: { x: 500, y: 220, w: 140, h: 50 },
  };
  const domainY = 130;
  const domainBox = { x: 270, y: domainY, w: 140, h: 40 };

  const isActive = anim.phase !== "idle";
  const classification = query ? classifyQuery(query) : null;

  return (
    <div className="widget-container ch4">
      <div className="widget-label">Interactive &middot; Reasoning Router</div>

      {/* Query Input */}
      <form onSubmit={handleSubmit} style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a query or choose a preset below..."
            style={{
              flex: 1,
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.82rem",
              padding: "0.5rem 0.75rem",
              borderRadius: "6px",
              border: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(0,0,0,0.3)",
              color: "#e4e4e7",
              outline: "none",
            }}
          />
          <button type="submit" className="btn-mono" style={{ color: "#a78bfa" }}>
            Route
          </button>
          {isActive && (
            <button type="button" className="btn-mono" onClick={reset}>
              Reset
            </button>
          )}
        </div>
      </form>

      {/* Preset Buttons */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.4rem",
          marginBottom: "1.25rem",
        }}
      >
        {PRESETS.map((p, i) => (
          <button
            key={i}
            className="btn-mono"
            onClick={() => handlePreset(p)}
            style={{
              fontSize: "0.72rem",
              borderColor:
                anim.mode === p.mode && anim.phase !== "idle"
                  ? MODE_COLORS[p.mode]
                  : undefined,
              color:
                anim.mode === p.mode && anim.phase !== "idle"
                  ? MODE_COLORS[p.mode]
                  : undefined,
            }}
          >
            {p.text.length > 40 ? p.text.slice(0, 37) + "..." : p.text}
          </button>
        ))}
      </div>

      {/* SVG Routing Diagram */}
      <div
        style={{
          background: "rgba(0,0,0,0.25)",
          borderRadius: "8px",
          border: "1px solid rgba(255,255,255,0.06)",
          padding: "0.75rem",
          marginBottom: "1.25rem",
          overflowX: "auto",
        }}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          style={{ maxHeight: 320, display: "block" }}
        >
          <defs>
            <filter id="glow-yellow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="glow-purple">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="glow-cyan">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <marker id="arrow-dim" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.15)" />
            </marker>
            {(["direct", "cot", "react"] as ReasoningMode[]).map((m) => (
              <marker key={m} id={`arrow-${m}`} viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={MODE_COLORS[m]} />
              </marker>
            ))}
          </defs>

          {/* Classifier Box */}
          <rect
            x={classifierBox.x}
            y={classifierBox.y}
            width={classifierBox.w}
            height={classifierBox.h}
            rx={8}
            fill={
              anim.phase === "classifying"
                ? "rgba(167,139,250,0.15)"
                : "rgba(255,255,255,0.04)"
            }
            stroke={
              anim.phase === "classifying" ? "#a78bfa" : "rgba(255,255,255,0.1)"
            }
            strokeWidth={anim.phase === "classifying" ? 2 : 1}
          >
            {anim.phase === "classifying" && (
              <animate attributeName="opacity" values="0.7;1;0.7" dur="0.8s" repeatCount="indefinite" />
            )}
          </rect>
          <text
            x={classifierBox.x + classifierBox.w / 2}
            y={classifierBox.y + 22}
            textAnchor="middle"
            fill="#e4e4e7"
            fontSize={13}
            fontFamily="var(--font-mono, monospace)"
            fontWeight={600}
          >
            Query Classifier
          </text>
          <text
            x={classifierBox.x + classifierBox.w / 2}
            y={classifierBox.y + 40}
            textAnchor="middle"
            fill="#a1a1aa"
            fontSize={10}
            fontFamily="var(--font-mono, monospace)"
          >
            domain detection
          </text>

          {/* Domain Box (appears on domain-reveal) */}
          <rect
            x={domainBox.x}
            y={domainBox.y}
            width={domainBox.w}
            height={domainBox.h}
            rx={6}
            fill={
              anim.domain
                ? "rgba(167,139,250,0.1)"
                : "rgba(255,255,255,0.02)"
            }
            stroke={
              anim.domain ? "#a78bfa" : "rgba(255,255,255,0.06)"
            }
            strokeWidth={1}
            opacity={anim.phase === "idle" ? 0.3 : 1}
          />
          <text
            x={domainBox.x + domainBox.w / 2}
            y={domainBox.y + 25}
            textAnchor="middle"
            fill={anim.domain ? "#a78bfa" : "#a1a1aa"}
            fontSize={11}
            fontFamily="var(--font-mono, monospace)"
            fontWeight={anim.domain ? 600 : 400}
          >
            {anim.domain
              ? classification?.domainLabel ?? anim.domain
              : "domain?"}
          </text>

          {/* Arrow: Classifier -> Domain */}
          <line
            x1={classifierBox.x + classifierBox.w / 2}
            y1={classifierBox.y + classifierBox.h}
            x2={domainBox.x + domainBox.w / 2}
            y2={domainBox.y}
            stroke={
              anim.phase !== "idle"
                ? "rgba(167,139,250,0.5)"
                : "rgba(255,255,255,0.08)"
            }
            strokeWidth={1.5}
            markerEnd={anim.phase !== "idle" ? undefined : "url(#arrow-dim)"}
            strokeDasharray={anim.phase === "classifying" ? "4 4" : "none"}
          >
            {anim.phase === "classifying" && (
              <animate attributeName="stroke-dashoffset" from="8" to="0" dur="0.4s" repeatCount="indefinite" />
            )}
          </line>

          {/* Mode Boxes + Arrows from Domain */}
          {(["direct", "cot", "react"] as ReasoningMode[]).map((m) => {
            const box = modeBoxes[m];
            const isChosen = anim.mode === m;
            const color = MODE_COLORS[m];
            const labels: Record<ReasoningMode, { name: string; desc: string }> = {
              direct: { name: "Direct", desc: "immediate answer" },
              cot: { name: "Chain-of-Thought", desc: "step-by-step" },
              react: { name: "ReAct", desc: "tool-augmented" },
            };
            const phaseRouting =
              anim.phase === "routing" ||
              anim.phase === "reasoning" ||
              anim.phase === "done";

            return (
              <g key={m}>
                {/* Arrow from domain to mode box */}
                <line
                  x1={domainBox.x + domainBox.w / 2}
                  y1={domainBox.y + domainBox.h}
                  x2={box.x + box.w / 2}
                  y2={box.y}
                  stroke={
                    isChosen && phaseRouting
                      ? color
                      : "rgba(255,255,255,0.06)"
                  }
                  strokeWidth={isChosen && phaseRouting ? 2 : 1}
                  markerEnd={
                    isChosen && phaseRouting
                      ? `url(#arrow-${m})`
                      : "url(#arrow-dim)"
                  }
                  opacity={
                    phaseRouting && !isChosen ? 0.2 : 1
                  }
                >
                  {isChosen && anim.phase === "routing" && (
                    <animate
                      attributeName="stroke-dasharray"
                      values="0 200;200 0"
                      dur="0.6s"
                      fill="freeze"
                    />
                  )}
                </line>

                {/* Mode Box */}
                <rect
                  x={box.x}
                  y={box.y}
                  width={box.w}
                  height={box.h}
                  rx={8}
                  fill={
                    isChosen && phaseRouting
                      ? `${color}15`
                      : "rgba(255,255,255,0.03)"
                  }
                  stroke={
                    isChosen && phaseRouting
                      ? color
                      : "rgba(255,255,255,0.08)"
                  }
                  strokeWidth={isChosen && phaseRouting ? 2 : 1}
                  opacity={phaseRouting && !isChosen ? 0.3 : 1}
                  filter={
                    isChosen && phaseRouting
                      ? `url(#glow-${m === "direct" ? "yellow" : m === "cot" ? "purple" : "cyan"})`
                      : undefined
                  }
                />
                <text
                  x={box.x + box.w / 2}
                  y={box.y + 22}
                  textAnchor="middle"
                  fill={isChosen && phaseRouting ? color : "#e4e4e7"}
                  fontSize={12}
                  fontFamily="var(--font-mono, monospace)"
                  fontWeight={600}
                >
                  {labels[m].name}
                </text>
                <text
                  x={box.x + box.w / 2}
                  y={box.y + 38}
                  textAnchor="middle"
                  fill="#a1a1aa"
                  fontSize={9}
                  fontFamily="var(--font-mono, monospace)"
                >
                  {labels[m].desc}
                </text>
              </g>
            );
          })}

          {/* Phase indicator pulse on classifier */}
          {anim.phase === "classifying" && (
            <circle
              cx={classifierBox.x + classifierBox.w / 2}
              cy={classifierBox.y + classifierBox.h / 2}
              r={35}
              fill="none"
              stroke="#a78bfa"
              strokeWidth={1}
              opacity={0.4}
            >
              <animate attributeName="r" from="30" to="50" dur="0.8s" repeatCount="indefinite" />
              <animate attributeName="opacity" from="0.4" to="0" dur="0.8s" repeatCount="indefinite" />
            </circle>
          )}
        </svg>
      </div>

      {/* Reasoning Output */}
      {(anim.phase === "reasoning" || anim.phase === "done") && anim.mode && (
        <div
          style={{
            background: "rgba(0,0,0,0.3)",
            borderRadius: "8px",
            border: `1px solid ${MODE_COLORS[anim.mode]}33`,
            padding: "1rem",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.72rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: MODE_COLORS[anim.mode],
              marginBottom: "0.75rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: MODE_COLORS[anim.mode],
                display: "inline-block",
                boxShadow: `0 0 8px ${MODE_COLORS[anim.mode]}`,
              }}
            />
            {anim.mode === "direct"
              ? "Direct Mode"
              : anim.mode === "cot"
              ? "Chain-of-Thought"
              : "ReAct Loop"}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            {anim.steps.map((step, i) => {
              if (i > anim.visibleSteps) return null;
              const isCurrent = i === anim.visibleSteps && anim.phase === "reasoning";
              const labelColor = STEP_LABEL_COLORS[step.label] ?? "#a1a1aa";
              const displayedContent = isCurrent
                ? step.content.slice(0, anim.typedChars)
                : i < anim.visibleSteps || anim.phase === "done"
                ? step.content
                : "";

              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    gap: "0.6rem",
                    alignItems: "flex-start",
                    opacity: i < anim.visibleSteps || anim.phase === "done" ? 1 : 0.9,
                  }}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "0.72rem",
                      fontWeight: 600,
                      color: labelColor,
                      minWidth: 80,
                      flexShrink: 0,
                      paddingTop: 2,
                    }}
                  >
                    {step.label === "Action" ? (
                      <span
                        style={{
                          background: "rgba(250,204,21,0.12)",
                          padding: "1px 6px",
                          borderRadius: 4,
                          border: "1px solid rgba(250,204,21,0.25)",
                        }}
                      >
                        {step.label}
                      </span>
                    ) : step.label === "Observation" ? (
                      <span
                        style={{
                          background: "rgba(74,222,128,0.1)",
                          padding: "1px 6px",
                          borderRadius: 4,
                          border: "1px solid rgba(74,222,128,0.2)",
                        }}
                      >
                        {step.label}
                      </span>
                    ) : (
                      step.label
                    )}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "0.8rem",
                      color: "#e4e4e7",
                      lineHeight: 1.6,
                    }}
                  >
                    {displayedContent}
                    {isCurrent && (
                      <span
                        style={{
                          display: "inline-block",
                          width: 6,
                          height: 14,
                          background: MODE_COLORS[anim.mode!],
                          marginLeft: 1,
                          verticalAlign: "text-bottom",
                          animation: "blink 0.8s infinite",
                        }}
                      />
                    )}
                  </span>
                </div>
              );
            })}
          </div>

          {anim.phase === "done" && (
            <div
              style={{
                marginTop: "0.75rem",
                paddingTop: "0.65rem",
                borderTop: "1px solid rgba(255,255,255,0.06)",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "0.7rem",
                color: "#a1a1aa",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="6" stroke="#4ade80" strokeWidth="1.5" />
                <path d="M4 7l2 2 4-4" stroke="#4ade80" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Routing complete &middot; {anim.steps.length} step{anim.steps.length > 1 ? "s" : ""} &middot;{" "}
              {anim.mode === "direct"
                ? "no tools needed"
                : anim.mode === "cot"
                ? "multi-step reasoning"
                : "tool-augmented reasoning"}
            </div>
          )}
        </div>
      )}

      {/* Idle / legend state */}
      {anim.phase === "idle" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "0.5rem",
            marginTop: "0.25rem",
          }}
        >
          {(
            [
              {
                mode: "direct" as ReasoningMode,
                label: "Direct Mode",
                desc: "Simple factual queries routed to immediate answer generation. No intermediate steps needed.",
                icon: "M3 7h18M3 12h12",
              },
              {
                mode: "cot" as ReasoningMode,
                label: "Chain-of-Thought",
                desc: "Complex reasoning decomposed into sequential steps. Each step builds on the previous one.",
                icon: "M4 6h16M4 10h16M4 14h16M4 18h10",
              },
              {
                mode: "react" as ReasoningMode,
                label: "ReAct Loop",
                desc: "Tool-augmented reasoning with Thought, Action, and Observation cycles until resolved.",
                icon: "M12 3v3m0 12v3m-9-9H0m21 0h3M5.6 5.6L3.5 3.5m13 13l2.1 2.1M5.6 18.4l-2.1 2.1m13-13l2.1-2.1",
              },
            ] as const
          ).map(({ mode, label, desc, icon }) => (
            <div
              key={mode}
              style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 8,
                padding: "0.75rem",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  marginBottom: "0.35rem",
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={MODE_COLORS[mode]} strokeWidth="2" strokeLinecap="round">
                  <path d={icon} />
                </svg>
                <span
                  style={{
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: MODE_COLORS[mode],
                  }}
                >
                  {label}
                </span>
              </div>
              <p
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "0.68rem",
                  color: "#a1a1aa",
                  lineHeight: 1.5,
                  margin: 0,
                }}
              >
                {desc}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
