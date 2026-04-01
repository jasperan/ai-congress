"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

type CircuitState = "closed" | "open" | "half-open";
type HealthStatus = "healthy" | "slow" | "failed";
type DegradationLevel = "full" | "simplified" | "single";

interface ModelState {
  name: string;
  status: HealthStatus;
  circuit: CircuitState;
  responseTime: number; // ms
  timeoutThreshold: number; // ms — EMA-adjusted
  failures: number;
  vram: number; // MB
}

const MODEL_NAMES = ["phi3", "mistral", "llama3", "qwen", "deepseek"] as const;

const INITIAL_MODELS: ModelState[] = MODEL_NAMES.map((name, i) => ({
  name,
  status: "healthy",
  circuit: "closed",
  responseTime: 120 + i * 30,
  timeoutThreshold: 500,
  failures: 0,
  vram: 1200 + i * 400,
}));

const TOTAL_VRAM = 24_000; // 24 GB
const ACCENT = "#fb923c";

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function statusColor(s: HealthStatus) {
  if (s === "healthy") return "#4ade80";
  if (s === "slow") return "#facc15";
  return "#f43f5e";
}

function circuitBadgeColor(c: CircuitState) {
  if (c === "closed") return "#4ade80";
  if (c === "open") return "#f43f5e";
  return "#facc15";
}

function degradationColor(d: DegradationLevel) {
  if (d === "full") return "#4ade80";
  if (d === "simplified") return "#facc15";
  return "#f43f5e";
}

function degradationLabel(d: DegradationLevel) {
  if (d === "full") return "Full (5 models)";
  if (d === "simplified") return "Simplified (3 models)";
  return "Single (1 model)";
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function CircuitBreakerWidget() {
  const [mounted, setMounted] = useState(false);
  const [models, setModels] = useState<ModelState[]>(INITIAL_MODELS);
  const [degradation, setDegradation] = useState<DegradationLevel>("full");
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [routingAnim, setRoutingAnim] = useState<string | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setMounted(true);
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, []);

  /* Live response-time jitter */
  useEffect(() => {
    if (!mounted) return;
    tickRef.current = setInterval(() => {
      setModels((prev) =>
        prev.map((m) => {
          if (m.status === "failed") return m;
          const jitter = (Math.random() - 0.5) * 40;
          const base = m.status === "slow" ? 420 : 140;
          const next = Math.max(50, Math.min(600, base + jitter));
          return { ...m, responseTime: Math.round(next) };
        })
      );
    }, 800);
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, [mounted]);

  /* Derive degradation level from active model count */
  useEffect(() => {
    const active = models.filter((m) => m.circuit !== "open").length;
    if (active >= 4) setDegradation("full");
    else if (active >= 2) setDegradation("simplified");
    else setDegradation("single");
  }, [models]);

  const log = useCallback((msg: string) => {
    setEventLog((prev) => [msg, ...prev].slice(0, 12));
  }, []);

  /* ---- Actions ---- */

  const simulateFailure = useCallback(() => {
    setModels((prev) => {
      const healthy = prev.filter((m) => m.status !== "failed");
      if (healthy.length === 0) return prev;
      const pick = healthy[Math.floor(Math.random() * healthy.length)];
      setRoutingAnim(pick.name);
      setTimeout(() => setRoutingAnim(null), 1500);
      log(`Circuit OPEN on ${pick.name} -- routing around`);
      return prev.map((m) =>
        m.name === pick.name
          ? { ...m, status: "failed" as HealthStatus, circuit: "open" as CircuitState, failures: m.failures + 1, responseTime: 0 }
          : m
      );
    });
  }, [log]);

  const simulateSlow = useCallback(() => {
    setModels((prev) => {
      const candidates = prev.filter((m) => m.status === "healthy");
      if (candidates.length === 0) return prev;
      const pick = candidates[Math.floor(Math.random() * candidates.length)];
      const newTimeout = Math.round(pick.timeoutThreshold * 1.35);
      log(`${pick.name} latency spike -- timeout EMA adjusted to ${newTimeout}ms`);
      return prev.map((m) =>
        m.name === pick.name
          ? { ...m, status: "slow" as HealthStatus, responseTime: 420, timeoutThreshold: newTimeout }
          : m
      );
    });
  }, [log]);

  const recovery = useCallback(() => {
    setModels((prev) => {
      const broken = prev.filter((m) => m.circuit === "open");
      if (broken.length === 0) {
        /* Try recovering slow models */
        const slow = prev.filter((m) => m.status === "slow");
        if (slow.length === 0) return prev;
        const pick = slow[0];
        log(`${pick.name} recovered -- circuit closed, timeout normalized`);
        return prev.map((m) =>
          m.name === pick.name
            ? { ...m, status: "healthy" as HealthStatus, timeoutThreshold: 500, responseTime: 150 }
            : m
        );
      }
      const pick = broken[0];
      /* First press: half-open */
      if (pick.circuit === "open") {
        log(`${pick.name} entering HALF-OPEN -- probing...`);
        return prev.map((m) =>
          m.name === pick.name
            ? { ...m, circuit: "half-open" as CircuitState, status: "slow" as HealthStatus, responseTime: 300 }
            : m
        );
      }
      return prev;
    });

    /* Delayed full close */
    setTimeout(() => {
      setModels((prev) => {
        const halfOpen = prev.filter((m) => m.circuit === "half-open");
        if (halfOpen.length === 0) return prev;
        const pick = halfOpen[0];
        log(`${pick.name} probe succeeded -- circuit CLOSED`);
        return prev.map((m) =>
          m.name === pick.name
            ? { ...m, circuit: "closed" as CircuitState, status: "healthy" as HealthStatus, responseTime: 150, timeoutThreshold: 500 }
            : m
        );
      });
    }, 1200);
  }, [log]);

  const resetAll = useCallback(() => {
    setModels(INITIAL_MODELS.map((m) => ({ ...m })));
    setEventLog([]);
    setRoutingAnim(null);
    log("All models reset to healthy");
  }, [log]);

  /* ---- Derived values ---- */
  const totalVram = models.reduce((sum, m) => (m.circuit !== "open" ? sum + m.vram : sum), 0);
  const vramPct = Math.min(100, Math.round((totalVram / TOTAL_VRAM) * 100));
  const totalFailures = models.reduce((sum, m) => sum + m.failures, 0);

  if (!mounted) return null;

  /* ================================================================ */
  /*  Render                                                          */
  /* ================================================================ */

  return (
    <div className="widget-container ch7">
      <div className="widget-label">Interactive &middot; Resilience &amp; Coordination</div>

      {/* ---- Header Stats ---- */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
        <div style={{ display: "flex", gap: "1.5rem", alignItems: "center", flexWrap: "wrap" }}>
          {/* Degradation mode */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.72rem", color: "#a1a1aa", fontFamily: "var(--font-mono), monospace", textTransform: "uppercase", letterSpacing: "0.05em" }}>Mode</span>
            <span
              style={{
                fontSize: "0.78rem",
                fontFamily: "var(--font-mono), monospace",
                fontWeight: 600,
                color: degradationColor(degradation),
                padding: "0.15rem 0.55rem",
                borderRadius: 4,
                border: `1px solid ${degradationColor(degradation)}33`,
                background: `${degradationColor(degradation)}11`,
                transition: "all 0.4s ease",
              }}
            >
              {degradationLabel(degradation)}
            </span>
          </div>
          {/* Failure count */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ fontSize: "0.72rem", color: "#a1a1aa", fontFamily: "var(--font-mono), monospace", textTransform: "uppercase", letterSpacing: "0.05em" }}>Failures</span>
            <span style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono), monospace", fontWeight: 700, color: totalFailures > 0 ? "#f43f5e" : "#4ade80" }}>{totalFailures}</span>
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button className="btn-mono" onClick={simulateFailure} style={{ borderColor: "#f43f5e44", color: "#f43f5e" }}>Simulate Failure</button>
          <button className="btn-mono" onClick={simulateSlow} style={{ borderColor: "#facc1544", color: "#facc15" }}>Simulate Slow</button>
          <button className="btn-mono" onClick={recovery} style={{ borderColor: `${ACCENT}44`, color: ACCENT }}>Recovery</button>
          <button className="btn-mono" onClick={resetAll}>Reset All</button>
        </div>
      </div>

      {/* ---- Model Cards ---- */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "0.65rem",
          marginBottom: "1.25rem",
        }}
      >
        {models.map((m) => (
          <ModelCard key={m.name} model={m} isRouting={routingAnim === m.name} />
        ))}
      </div>

      {/* ---- Routing animation overlay text ---- */}
      {routingAnim && (
        <div
          style={{
            textAlign: "center",
            fontSize: "0.75rem",
            fontFamily: "var(--font-mono), monospace",
            color: ACCENT,
            marginBottom: "0.75rem",
            animation: "fadeIn 0.3s ease-out",
          }}
        >
          Routing traffic around <span style={{ color: "#f43f5e", fontWeight: 700 }}>{routingAnim}</span> &rarr; remaining healthy models
        </div>
      )}

      {/* ---- VRAM Bar ---- */}
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono), monospace", color: "#a1a1aa", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            GPU VRAM Utilization
          </span>
          <span style={{ fontSize: "0.72rem", fontFamily: "var(--font-mono), monospace", color: vramPct > 85 ? "#f43f5e" : vramPct > 60 ? "#facc15" : "#4ade80" }}>
            {totalVram.toLocaleString()} / {TOTAL_VRAM.toLocaleString()} MB ({vramPct}%)
          </span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              width: `${vramPct}%`,
              borderRadius: 4,
              background: vramPct > 85 ? "linear-gradient(90deg, #f43f5e, #fb7185)" : vramPct > 60 ? `linear-gradient(90deg, #facc15, ${ACCENT})` : "linear-gradient(90deg, #4ade80, #34d399)",
              transition: "width 0.6s ease, background 0.6s ease",
            }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.25rem" }}>
          {models.map((m) => (
            <span
              key={m.name}
              style={{
                fontSize: "0.6rem",
                fontFamily: "var(--font-mono), monospace",
                color: m.circuit === "open" ? "#a1a1aa44" : "#a1a1aa",
                textDecoration: m.circuit === "open" ? "line-through" : "none",
                transition: "all 0.3s",
              }}
            >
              {m.name}: {m.circuit === "open" ? "--" : `${m.vram}MB`}
            </span>
          ))}
        </div>
      </div>

      {/* ---- Event Log ---- */}
      <div
        style={{
          background: "rgba(0,0,0,0.35)",
          borderRadius: 8,
          border: "1px solid rgba(255,255,255,0.06)",
          padding: "0.6rem 0.75rem",
          maxHeight: 120,
          overflowY: "auto",
          fontFamily: "var(--font-mono), monospace",
          fontSize: "0.68rem",
          lineHeight: 1.7,
          color: "#a1a1aa",
        }}
        className="scrollbar-hide"
      >
        {eventLog.length === 0 ? (
          <span style={{ color: "#a1a1aa55" }}>Event log &mdash; interact with controls above</span>
        ) : (
          eventLog.map((entry, i) => (
            <div key={`${entry}-${i}`} style={{ opacity: 1 - i * 0.07, animation: i === 0 ? "slideInRight 0.3s ease-out" : undefined }}>
              <span style={{ color: ACCENT, marginRight: 6 }}>&rsaquo;</span>
              {entry}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/* ================================================================== */
/*  ModelCard                                                         */
/* ================================================================== */

function ModelCard({ model: m, isRouting }: { model: ModelState; isRouting: boolean }) {
  const isFailed = m.status === "failed";
  const barPct = isFailed ? 0 : Math.min(100, (m.responseTime / 600) * 100);
  const thresholdPct = Math.min(100, (m.timeoutThreshold / 600) * 100);

  return (
    <div
      style={{
        background: isFailed ? "rgba(244,63,94,0.06)" : "rgba(255,255,255,0.02)",
        border: `1px solid ${isFailed ? "rgba(244,63,94,0.25)" : isRouting ? "rgba(244,63,94,0.5)" : "rgba(255,255,255,0.08)"}`,
        borderRadius: 10,
        padding: "0.7rem 0.6rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        transition: "all 0.4s ease",
        position: "relative",
        overflow: "hidden",
        animation: isRouting ? "pulse-highlight 1.2s ease-out" : undefined,
      }}
    >
      {/* Skipped-route overlay */}
      {isRouting && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(244,63,94,0.08)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2,
            borderRadius: 10,
          }}
        >
          <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f43f5e", opacity: 0.7, fontFamily: "var(--font-mono), monospace", letterSpacing: "0.1em" }}>SKIP</span>
        </div>
      )}

      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          {/* Status dot */}
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: statusColor(m.status),
              display: "inline-block",
              boxShadow: `0 0 6px ${statusColor(m.status)}66`,
              transition: "all 0.4s",
            }}
          />
          <span
            style={{
              fontSize: "0.78rem",
              fontFamily: "var(--font-mono), monospace",
              fontWeight: 600,
              color: isFailed ? "#a1a1aa66" : "#e4e4e7",
              textDecoration: isFailed ? "line-through" : "none",
              transition: "all 0.3s",
            }}
          >
            {m.name}
          </span>
        </div>
        {/* Circuit badge */}
        <span
          style={{
            fontSize: "0.58rem",
            fontFamily: "var(--font-mono), monospace",
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            padding: "0.1rem 0.4rem",
            borderRadius: 3,
            color: circuitBadgeColor(m.circuit),
            border: `1px solid ${circuitBadgeColor(m.circuit)}44`,
            background: `${circuitBadgeColor(m.circuit)}15`,
            transition: "all 0.4s",
          }}
        >
          {m.circuit === "half-open" ? "Half" : m.circuit === "closed" ? "Closed" : "Open"}
        </span>
      </div>

      {/* Response time + threshold bar */}
      <div style={{ position: "relative", height: 18 }}>
        {/* Track */}
        <div style={{ position: "absolute", top: 7, left: 0, right: 0, height: 4, borderRadius: 2, background: "rgba(255,255,255,0.06)" }} />
        {/* Response time fill */}
        <div
          style={{
            position: "absolute",
            top: 7,
            left: 0,
            width: `${barPct}%`,
            height: 4,
            borderRadius: 2,
            background: statusColor(m.status),
            transition: "width 0.5s ease, background 0.4s",
            boxShadow: `0 0 6px ${statusColor(m.status)}44`,
          }}
        />
        {/* Timeout threshold line */}
        <div
          style={{
            position: "absolute",
            top: 3,
            left: `${thresholdPct}%`,
            width: 1,
            height: 12,
            background: "#a1a1aa",
            transition: "left 0.5s ease",
          }}
        />
        {/* Threshold label */}
        <span
          style={{
            position: "absolute",
            top: -1,
            left: `calc(${thresholdPct}% + 3px)`,
            fontSize: "0.5rem",
            color: "#a1a1aa88",
            fontFamily: "var(--font-mono), monospace",
            transition: "left 0.5s ease",
            whiteSpace: "nowrap",
          }}
        >
          {m.timeoutThreshold}ms
        </span>
      </div>

      {/* Latency value */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontSize: "0.62rem", color: "#a1a1aa88", fontFamily: "var(--font-mono), monospace" }}>latency</span>
        <span
          style={{
            fontSize: "0.82rem",
            fontFamily: "var(--font-mono), monospace",
            fontWeight: 700,
            color: isFailed ? "#a1a1aa44" : statusColor(m.status),
            transition: "color 0.4s",
          }}
        >
          {isFailed ? "--" : `${m.responseTime}ms`}
        </span>
      </div>

      {/* Failure count */}
      {m.failures > 0 && (
        <div style={{ fontSize: "0.58rem", fontFamily: "var(--font-mono), monospace", color: "#f43f5e99", textAlign: "right" }}>
          {m.failures} failure{m.failures > 1 ? "s" : ""}
        </div>
      )}
    </div>
  );
}
