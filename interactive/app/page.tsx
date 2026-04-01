"use client";

import { useEffect } from "react";
import {
  SwarmModesWidget,
  SemanticVotingWidget,
  DebateWidget,
  ReasoningRouterWidget,
  ACPWidget,
  RAGPipelineWidget,
  CircuitBreakerWidget,
  ELORatingWidget,
  CongressSimWidget,
  PersonalityWidget,
} from "./components/widgets";

const chapters = [
  { id: "ch1", num: "01", title: "Swarm Modes", color: "text-ch1" },
  { id: "ch2", num: "02", title: "Voting", color: "text-ch2" },
  { id: "ch3", num: "03", title: "Debate", color: "text-ch3" },
  { id: "ch4", num: "04", title: "Reasoning", color: "text-ch4" },
  { id: "ch5", num: "05", title: "ACP", color: "text-ch5" },
  { id: "ch6", num: "06", title: "RAG", color: "text-ch6" },
  { id: "ch7", num: "07", title: "Resilience", color: "text-ch7" },
  { id: "ch8", num: "08", title: "Learning", color: "text-ch8" },
  { id: "ch9", num: "09", title: "Congress", color: "text-ch9" },
  { id: "ch10", num: "10", title: "Personality", color: "text-ch10" },
];

const tocItems = [
  { id: "ch1", num: "01", title: "Swarm Intelligence", desc: "4 modes for multi-model orchestration and parallel inference", color: "text-ch1", border: "border-orange-500/20 hover:border-orange-500/40" },
  { id: "ch2", num: "02", title: "Semantic Voting", desc: "LLM-judged clustering, weighted consensus, minority reports", color: "text-ch2", border: "border-cyan-500/20 hover:border-cyan-500/40" },
  { id: "ch3", num: "03", title: "Multi-Round Debate", desc: "Toulmin argumentation, devil's advocate, conviction tracking", color: "text-ch3", border: "border-green-500/20 hover:border-green-500/40" },
  { id: "ch4", num: "04", title: "Adaptive Reasoning", desc: "CoT, ReAct with tools, domain-aware routing", color: "text-ch4", border: "border-purple-500/20 hover:border-purple-500/40" },
  { id: "ch5", num: "05", title: "Agent Communication", desc: "Role-based agents, message bus, task handoff", color: "text-ch5", border: "border-pink-500/20 hover:border-pink-500/40" },
  { id: "ch6", num: "06", title: "RAG Pipeline", desc: "Document ingestion, vector search, source attribution", color: "text-ch6", border: "border-yellow-500/20 hover:border-yellow-500/40" },
  { id: "ch7", num: "07", title: "Resilience & Coordination", desc: "Circuit breakers, adaptive timeouts, graceful degradation", color: "text-ch7", border: "border-orange-400/20 hover:border-orange-400/40" },
  { id: "ch8", num: "08", title: "Learning & Adaptation", desc: "ELO ratings, dynamic weights, feedback loops", color: "text-ch8", border: "border-sky-500/20 hover:border-sky-500/40" },
  { id: "ch9", num: "09", title: "Congressional Simulation", desc: "6-phase legislative process with tick-based events", color: "text-ch9", border: "border-emerald-500/20 hover:border-emerald-500/40" },
  { id: "ch10", num: "10", title: "Personality & Emotion", desc: "Big Five traits, emotional voting, communication styles", color: "text-ch10", border: "border-rose-500/20 hover:border-rose-500/40" },
];

export default function Home() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) e.target.classList.add("visible");
        }),
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );
    document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen">
      {/* ---- Nav ---- */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-background/80 border-b border-border">
        <div className="max-w-5xl mx-auto px-4 py-2 flex items-center gap-1 overflow-x-auto scrollbar-hide">
          <a href="#top" className="nav-link font-bold text-foreground mr-2">
            AI<span className="text-muted-foreground font-normal">.congress</span>
          </a>
          {chapters.map((ch) => (
            <a key={ch.id} href={`#${ch.id}`} className="nav-link">
              <span className={ch.color}>{ch.num}</span>
              <span className="hidden md:inline ml-1">{ch.title}</span>
            </a>
          ))}
        </div>
      </nav>

      {/* ---- Hero ---- */}
      <header id="top" className="relative overflow-hidden py-24 md:py-32">
        <div className="hero-glow bg-orange-500 top-[-200px] left-[10%]" />
        <div className="hero-glow bg-cyan-500 top-[-100px] right-[15%]" />
        <div className="hero-glow bg-purple-500 bottom-[-200px] left-[40%]" />
        <div className="max-w-5xl mx-auto px-4 relative z-10">
          <p className="font-mono text-xs text-muted-foreground tracking-widest uppercase mb-4">
            Interactive Explorer
          </p>
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-tight mb-6">
            Where{" "}
            <span className="text-ch1">LLMs</span>{" "}
            Debate, Vote &{" "}
            <span className="text-ch2">Reach Consensus</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Ten chapters. Ten interactive widgets. One journey through{" "}
            <span className="text-ch1">swarm modes</span>,{" "}
            <span className="text-ch2">semantic voting</span>,{" "}
            <span className="text-ch3">multi-round debate</span>,{" "}
            <span className="text-ch4">adaptive reasoning</span>,{" "}
            <span className="text-ch5">agent communication</span>,{" "}
            <span className="text-ch6">RAG</span>,{" "}
            <span className="text-ch7">resilience</span>,{" "}
            <span className="text-ch8">learning</span>,{" "}
            <span className="text-ch9">congressional simulation</span>, and{" "}
            <span className="text-ch10">personality-driven AI</span>.
          </p>
          <p className="text-sm text-muted-foreground mt-4 font-mono">
            Every concept below is interactive. Click buttons, drag sliders, run simulations.
          </p>
        </div>
      </header>

      {/* ---- Table of Contents ---- */}
      <div className="max-w-5xl mx-auto px-4 mb-16">
        <div className="bg-card border border-border rounded-xl p-6">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-4">
            Chapters
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {tocItems.map((ch) => (
              <a
                key={ch.id}
                href={`#${ch.id}`}
                className={`block p-4 rounded-lg border ${ch.border} bg-card transition-all hover:bg-white/[0.02]`}
              >
                <span className={`font-mono text-xs ${ch.color}`}>{ch.num}</span>
                <h3 className="font-semibold text-sm mt-1">{ch.title}</h3>
                <p className="text-xs text-muted-foreground mt-1">{ch.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* ---- Main Content ---- */}
      <div className="max-w-5xl mx-auto px-4 prose-dark">

        {/* ===== Chapter 1: Swarm Modes ===== */}
        <section id="ch1" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch1 tracking-wider uppercase mb-2">01</p>
          <h2>Swarm Intelligence</h2>
          <p>
            Instead of trusting a single model, AI Congress sends every query to <strong>multiple LLMs simultaneously</strong> and
            aggregates their responses through weighted voting. This ensemble approach is more robust than any individual model — the
            swarm catches errors that individual models miss, and diverse perspectives lead to higher-quality answers.
          </p>
          <p>
            The system supports <span className="text-ch1">four distinct swarm modes</span>, each designed for different scenarios.
            Multi-model mode queries different architectures (phi3, mistral, llama3, qwen, deepseek) in parallel. Multi-request
            mode uses the same model at different temperatures for diversity. Hybrid mode selects top performers by weight. Personality
            mode assigns custom personas. All modes support real-time streaming with live vote breakdowns.
          </p>
          <SwarmModesWidget />
          <p>
            The swarm orchestrator manages concurrent execution with configurable semaphores (default: 10 parallel requests)
            and uses <strong>asyncio.gather()</strong> for maximum throughput. Each mode produces a set of candidate responses that
            flow into the voting engine for consensus determination.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 2: Semantic Voting ===== */}
        <section id="ch2" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch2 tracking-wider uppercase mb-2">02</p>
          <h2>Semantic Voting &amp; Consensus</h2>
          <p>
            Traditional voting counts identical strings. But LLMs express the same idea in wildly different words —
            &quot;The capital is Paris&quot; and &quot;France&apos;s capital city is Paris&quot; are semantically identical but string-different.
            AI Congress uses an <span className="text-ch2">LLM-as-judge</span> to cluster responses by meaning, not by text.
          </p>
          <p>
            The voting engine supports multiple algorithms: <strong>weighted majority vote</strong> (model performance weights),
            <strong>confidence-based voting</strong> (self-reported certainty), and <strong>temperature ensemble</strong>
            (lower temperature = higher weight). A contextual selector automatically picks the best algorithm for each query type.
            The system also generates <strong>minority reports</strong> — surfacing the strongest dissenting opinion even when
            consensus is reached.
          </p>
          <SemanticVotingWidget />
          <p>
            Confidence calibration ensures that when a model says it is 80% confident, it is actually correct about
            80% of the time. The calibrator fits predicted-vs-actual accuracy curves and adjusts future weight
            allocations accordingly — a self-correcting feedback loop.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 3: Multi-Round Debate ===== */}
        <section id="ch3" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch3 tracking-wider uppercase mb-2">03</p>
          <h2>Multi-Round Debate</h2>
          <p>
            Voting captures a snapshot — debate explores the reasoning space. AI Congress implements a
            <span className="text-ch3"> structured argumentation system</span> based on the Toulmin model: every argument
            must include a Claim, Evidence, Warrant, Qualifier, and Rebuttal. This forces models to construct rigorous,
            evidence-backed positions rather than asserting opinions.
          </p>
          <p>
            The debate runs up to 3 rounds with <strong>dynamic depth</strong> — simple questions with early consensus get fewer rounds,
            while contentious topics trigger the full protocol. A <strong>devil&apos;s advocate</strong> mechanism ensures one model
            always challenges the emerging consensus, preventing groupthink. Conviction scores track how strongly each model
            holds its position across rounds, with a bonus for consistency.
          </p>
          <DebateWidget />
          <p>
            Pressure prompts escalate commitment strategically. Round 1 is open-ended. Round 2 presents opposing views
            and asks &quot;Are you certain?&quot;. Round 3 introduces the devil&apos;s advocate. Round 4 demands final positions.
            An <strong>indecision detector</strong> identifies models that flip-flop and down-weights their final votes.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 4: Adaptive Reasoning ===== */}
        <section id="ch4" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch4 tracking-wider uppercase mb-2">04</p>
          <h2>Adaptive Reasoning</h2>
          <p>
            Not every question needs the same thinking process. A factual lookup (&quot;What is 2+2?&quot;) should be instant.
            A math word problem needs step-by-step reasoning. A current-events question needs real tools.
            AI Congress routes each query through an <span className="text-ch4">adaptive reasoning engine</span> that picks
            the right approach automatically.
          </p>
          <p>
            Three reasoning modes are available: <strong>Direct</strong> for simple factual queries, <strong>Chain-of-Thought (CoT)</strong>
            for complex multi-step reasoning, and <strong>ReAct</strong> for tool-augmented tasks. ReAct agents can invoke
            real tools — web search (DuckDuckGo, SearXNG) and calculators — and loop through
            Thought → Action → Observation cycles until they reach an answer. A query domain classifier (math, coding,
            science, general) further refines the routing.
          </p>
          <ReasoningRouterWidget />
          <p>
            A self-verification loop adds a final check: the system uses web search to fact-check its own answers
            and flags potential inaccuracies. Meta-cognitive monitoring tracks uncertainty — when the model
            &quot;knows it doesn&apos;t know,&quot; it signals for human review rather than hallucinating.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 5: Agent Communication Protocol ===== */}
        <section id="ch5" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch5 tracking-wider uppercase mb-2">05</p>
          <h2>Agent Communication Protocol</h2>
          <p>
            AI Congress doesn&apos;t just run models in parallel — it coordinates them as a team. The
            <span className="text-ch5"> Agent Communication Protocol (ACP)</span> defines 14 modules for agent discovery,
            message passing, role assignment, and task delegation. Think of it as an operating system for multi-agent collaboration.
          </p>
          <p>
            Five specialized roles drive the workflow: the <strong>Planner</strong> decomposes complex queries into sub-tasks,
            <strong>Workers</strong> research and draft responses, the <strong>Critic</strong> challenges weak arguments,
            the <strong>Judge</strong> evaluates quality, and the <strong>Synthesizer</strong> merges the best elements into
            a final answer. An OTP-style supervisor monitors agent health with heartbeats and implements retry-with-backoff
            for failed operations.
          </p>
          <ACPWidget />
          <p>
            The message bus enables pub/sub communication between agents. Hash-based anchoring lets models
            reference specific parts of other responses during cross-examination. An audit trail logs every
            decision for full transparency and post-hoc analysis.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 6: RAG Pipeline ===== */}
        <section id="ch6" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch6 tracking-wider uppercase mb-2">06</p>
          <h2>RAG Pipeline</h2>
          <p>
            LLMs only know what they were trained on. <span className="text-ch6">Retrieval-Augmented Generation (RAG)</span>
            bridges this gap by injecting relevant documents into the prompt at query time. AI Congress implements a
            full RAG pipeline: upload documents (PDF, DOCX, XLSX, Markdown, PPTX), chunk them into overlapping segments,
            embed them as vectors, and retrieve the most relevant chunks for each query.
          </p>
          <p>
            The chunking strategy uses 512-token windows with 50-token overlap to preserve context across boundaries.
            Embeddings are generated via sentence-transformers and stored in Oracle Vector DB for production or SQLite
            for local development. When a query arrives, it is embedded and matched against stored chunks using
            cosine similarity. The top-k results are injected as context, and the generated answer includes
            chunk-level source citations for full attribution.
          </p>
          <RAGPipelineWidget />
          <p>
            Multi-source fusion combines chunks from different documents, weighting by relevance score. The
            <strong> stare decisis</strong> (precedent) system stores past decisions in the vector database, allowing the
            system to cite its own prior reasoning — just like a court citing case law.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 7: Resilience & Coordination ===== */}
        <section id="ch7" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch7 tracking-wider uppercase mb-2">07</p>
          <h2>Resilience &amp; Coordination</h2>
          <p>
            When you depend on multiple LLMs, failures are inevitable — a model crashes, VRAM runs out, a response
            takes too long. AI Congress implements <span className="text-ch7">production-grade resilience patterns</span>
            borrowed from distributed systems: circuit breakers, adaptive timeouts, graceful degradation, and
            VRAM-aware concurrency control.
          </p>
          <p>
            The <strong>circuit breaker</strong> tracks failure rates per model. After consecutive failures, it &quot;opens&quot;
            the circuit and routes around the failing model. After a cooldown period, it enters a &quot;half-open&quot; state
            to test recovery. <strong>Adaptive timeouts</strong> use exponential moving averages of response times to set
            per-model timeout thresholds — fast models get tight deadlines, slow models get more slack. When enough
            models fail, the system gracefully degrades from full ensemble to simplified mode to single-model fallback.
          </p>
          <CircuitBreakerWidget />
          <p>
            The concurrency governor polls nvidia-smi to track GPU VRAM usage in real-time, throttling parallel
            requests when memory pressure is high. This prevents OOM crashes that would take down the entire system.
            Coalition formation groups similar responses pre-voting to reduce redundant computation.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 8: Learning & Adaptation ===== */}
        <section id="ch8" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch8 tracking-wider uppercase mb-2">08</p>
          <h2>Learning &amp; Adaptation</h2>
          <p>
            AI Congress doesn&apos;t just aggregate models — it learns which models perform best over time.
            The <span className="text-ch8">dynamic weight system</span> uses ELO ratings (like chess) and
            exponential moving averages to continuously update model weights based on voting outcomes and user
            feedback.
          </p>
          <p>
            When a model&apos;s response wins the vote, its ELO rating increases and its weight nudges up. When it loses,
            the opposite happens. The EMA formula (<code>w_new = alpha * win + (1 - alpha) * w_old</code>) provides
            smooth adaptation — recent performance matters more than ancient history, but the system doesn&apos;t
            overreact to a single bad result. Users can also give explicit thumbs-up/down feedback, which directly
            adjusts weights through the feedback loop.
          </p>
          <ELORatingWidget />
          <p>
            Prompt template evolution A/B tests different debate prompts and keeps the best performers. Personality
            persistence stores emotional state across sessions, so agents remember their &quot;mood&quot; from previous
            conversations. The entire learning system operates at zero additional cost — it reuses existing
            inference results to update its internal state.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 9: Congressional Simulation ===== */}
        <section id="ch9" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch9 tracking-wider uppercase mb-2">09</p>
          <h2>Congressional Simulation</h2>
          <p>
            The flagship mode: a <span className="text-ch9">full congressional simulation</span> where AI agents role-play
            as legislators debating policy proposals. The simulation runs through six authentic legislative phases —
            Introduction, Committee, Floor Debate, Amendments, Final Arguments, and Voting — with tick-based event
            streaming for real-time observation.
          </p>
          <p>
            Each agent adopts a personality from the US Congress dataset (47 real senators and representatives with
            party affiliations and policy positions), Hollywood celebrities, or YouTube personalities. Sentiment
            sparklines track each agent&apos;s emotional trajectory. Amendment proposals can modify the original bill,
            and agents vote on amendments before the final vote. A persuasion network graph shows which agents
            are influencing others.
          </p>
          <CongressSimWidget />
          <p>
            The Rust TUI provides the most immersive experience — phase-adaptive layouts morph as the simulation
            progresses, with live persuasion graphs, amendment diffs, filibuster alerts, and vote prediction gauges.
            Transcripts can be exported as Markdown or JSON for analysis.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Chapter 10: Personality & Emotion ===== */}
        <section id="ch10" className="scroll-mt-16 reveal">
          <p className="font-mono text-xs text-ch10 tracking-wider uppercase mb-2">10</p>
          <h2>Personality &amp; Emotional Voting</h2>
          <p>
            AI Congress doesn&apos;t treat all agents as identical reasoning machines. Each agent has a
            <span className="text-ch10"> Big Five personality profile</span> (Openness, Conscientiousness, Extraversion,
            Agreeableness, Neuroticism) that shapes how it communicates, deliberates, and votes. Highly agreeable
            agents are more willing to compromise; highly neurotic agents express more caution and uncertainty.
          </p>
          <p>
            Emotional state tracking persists across sessions — an agent that was &quot;frustrated&quot; in a previous debate
            carries that mood into the next one. Communication styles adapt based on personality: formal agents
            cite evidence and use careful language, while casual agents use direct, conversational phrasing.
            The <strong>emotional voting</strong> system weights each agent&apos;s vote by their emotional state —
            confident agents have more influence than anxious ones.
          </p>
          <PersonalityWidget />
          <p>
            Three pre-built personality sets are included: US Congress (senators and representatives with real
            political leanings), Hollywood (actors and directors with creative temperaments), and YouTubers
            (content creators with varied communication styles). Custom personality profiles can be created
            by adjusting the five trait sliders.
          </p>
          <div className="section-divider" />
        </section>

        {/* ===== Conclusion ===== */}
        <section className="reveal mb-16">
          <h2>The Full Picture</h2>
          <p>
            AI Congress combines these ten layers into a single system. A query enters through any interface
            (Web UI, CLI, TUI, API), gets routed to the appropriate reasoning mode, dispatched to a swarm of
            specialized agents who debate, vote, learn from outcomes, and produce a consensus response — all
            backed by RAG knowledge and protected by production-grade resilience patterns.
          </p>
          <div className="bg-card border border-border rounded-xl p-6 my-8">
            <h3 className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-4">Architecture at a Glance</h3>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {[
                { label: "Swarm Modes", value: "4", color: "#f97316" },
                { label: "Intelligence Improvements", value: "35", color: "#22d3ee" },
                { label: "ACP Modules", value: "14", color: "#f472b6" },
                { label: "Reasoning Modes", value: "3", color: "#a78bfa" },
                { label: "Simulation Phases", value: "6", color: "#34d399" },
              ].map((stat) => (
                <div key={stat.label} className="text-center p-3 rounded-lg border border-border">
                  <div className="text-2xl font-bold font-mono" style={{ color: stat.color }}>{stat.value}</div>
                  <div className="text-xs text-muted-foreground mt-1">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
          <p>
            The system runs on <strong>local Ollama models</strong> with zero API costs, stores its learning state in
            Oracle 26ai Free tier, and supports Docker deployment for production. Whether you are exploring multi-agent
            AI for research, building production ensemble systems, or just curious about how LLMs can collaborate —
            AI Congress is a complete, working implementation of these ideas.
          </p>
        </section>
      </div>

      {/* ---- Footer ---- */}
      <footer className="max-w-5xl mx-auto px-4 py-8 text-center text-xs text-muted-foreground border-t border-border">
        <p>
          AI Congress — Autonomous LLM Multi-Agent System{" "}
          <span className="mx-2">|</span>{" "}
          Built with FastAPI, Svelte, Rust TUI, and Ollama{" "}
          <span className="mx-2">|</span>{" "}
          Interactive explorer built with Next.js + React 19 + Tailwind CSS
        </p>
      </footer>
    </div>
  );
}
