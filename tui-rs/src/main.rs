mod api;
mod app;
mod screens;
mod session;
mod theme;
mod widgets;

use std::io;
use std::time::Duration;

use app::{ActiveScreen, App};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use crate::api::rest::ApiClient;
use crate::api::ws_chat::{WsChatClient, WsChatRequest};
use crate::api::ws_simulation::WsSimulationClient;
use crate::screens::*;

#[derive(Parser, Debug)]
#[command(name = "congress-tui", about = "AI Congress — Unified TUI")]
struct Args {
    /// Backend API URL
    #[arg(short, long, default_value = "http://localhost:8000")]
    server: String,

    /// Skip splash, go directly to simulation (legacy mode)
    #[arg(long)]
    simulation: bool,

    /// Topic for direct simulation launch
    #[arg(long)]
    topic: Option<String>,

    /// Number of agents for direct simulation
    #[arg(long, default_value_t = 10)]
    agents: u32,

    /// Number of ticks for direct simulation
    #[arg(long, default_value_t = 100)]
    ticks: u32,

    /// Model for direct simulation
    #[arg(long, default_value = "qwen3.5:9b")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    // Initialize terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let base_url = args.server.trim_end_matches('/').to_string();
    let api_client = ApiClient::new(&base_url);

    // Handle --simulation flag (legacy direct-launch mode)
    if args.simulation {
        let topic = args
            .topic
            .unwrap_or_else(|| "Should AI systems be regulated by federal law?".to_string());
        let ws_url = format!(
            "{}/ws/simulation",
            base_url
                .replace("http://", "ws://")
                .replace("https://", "wss://")
        );
        match WsSimulationClient::connect(&ws_url).await {
            Ok(mut ws) => {
                let _ = ws
                    .send_config(&topic, args.agents.min(10), args.ticks, &args.model)
                    .await;
                let mut sim =
                    simulation::SimulationScreen::new(topic, args.ticks, args.model);
                let result = run_simulation_loop(&mut terminal, &mut sim, &mut ws).await;
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                if let Err(e) = result {
                    eprintln!("Error: {}", e);
                }
                return Ok(());
            }
            Err(e) => {
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                eprintln!("Failed to connect: {}", e);
                return Ok(());
            }
        }
    }

    // Create app — starts on Splash screen
    let mut app = App::new(base_url.clone());

    // Fire initial health check and populate splash
    {
        let splash = match &mut app.screen {
            ActiveScreen::Splash(s) => s,
            _ => unreachable!(),
        };
        match api_client.health_check().await {
            Ok(true) => match api_client.list_models().await {
                Ok(models) => splash.set_connected(models.len()),
                Err(e) => splash.set_error(format!("{}", e)),
            },
            _ => splash.set_error("Backend not reachable".to_string()),
        }
    }

    // WebSocket clients — created on demand when entering chat/simulation
    let mut ws_chat: Option<WsChatClient> = None;
    let mut ws_sim: Option<WsSimulationClient> = None;

    // Main event loop
    let result = run_main_loop(
        &mut terminal,
        &mut app,
        &api_client,
        &mut ws_chat,
        &mut ws_sim,
        &base_url,
    )
    .await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {}", e);
    }

    Ok(())
}

async fn run_main_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    api_client: &ApiClient,
    ws_chat: &mut Option<WsChatClient>,
    ws_sim: &mut Option<WsSimulationClient>,
    base_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        if !app.running {
            return Ok(());
        }

        // Capture help state before the closure (closure borrows app.screen mutably)
        let show_help = app.show_help;
        let help_screen_name = match &app.screen {
            ActiveScreen::Splash(_) => "Splash",
            ActiveScreen::Models(_) => "Models",
            ActiveScreen::ModeSelect(_) => "Mode Select",
            ActiveScreen::ChatDashboard(_) => "Chat Dashboard",
            ActiveScreen::Simulation(_) => "Simulation",
            ActiveScreen::Results(_) => "Results",
        };
        let help_bindings: &[crate::widgets::help_overlay::KeyBinding] = match &app.screen {
            ActiveScreen::Splash(_) => crate::widgets::help_overlay::SPLASH_BINDINGS,
            ActiveScreen::Models(_) => crate::widgets::help_overlay::MODELS_BINDINGS,
            ActiveScreen::ModeSelect(_) => crate::widgets::help_overlay::MODE_SELECT_BINDINGS,
            ActiveScreen::ChatDashboard(_) => crate::widgets::help_overlay::CHAT_BINDINGS,
            ActiveScreen::Simulation(_) => crate::widgets::help_overlay::SIMULATION_BINDINGS,
            ActiveScreen::Results(_) => crate::widgets::help_overlay::RESULTS_BINDINGS,
        };

        // Render — use &mut app.screen so ModelsScreen.draw(&mut self) works
        terminal.draw(|f| {
            let area = f.area();
            match &mut app.screen {
                ActiveScreen::Splash(s) => s.draw(f, area),
                ActiveScreen::Models(s) => s.draw(f, area),
                ActiveScreen::ModeSelect(s) => s.draw(f, area),
                ActiveScreen::ChatDashboard(s) => s.draw(f, area),
                ActiveScreen::Simulation(s) => s.draw(f, area),
                ActiveScreen::Results(s) => s.draw(f, area),
            }
            if show_help {
                crate::widgets::help_overlay::draw_help_overlay(
                    f,
                    area,
                    help_screen_name,
                    help_bindings,
                );
            }
        })?;

        // Throughput tick for active sessions
        match &mut app.screen {
            ActiveScreen::ChatDashboard(s) => s.tick_throughput(),
            ActiveScreen::Simulation(s) => s.tick_throughput(),
            _ => {}
        }

        // Poll keyboard (~60 fps)
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                // Global: Ctrl+C always quits
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('c')
                {
                    app.running = false;
                    continue;
                }

                // Global: F1 toggles help overlay
                if key.code == KeyCode::F(1) {
                    app.show_help = !app.show_help;
                    continue;
                }

                // Dispatch to active screen
                let action = match &mut app.screen {
                    ActiveScreen::Splash(s) => s.handle_key(key),
                    ActiveScreen::Models(s) => s.handle_key(key),
                    ActiveScreen::ModeSelect(s) => s.handle_key(key),
                    ActiveScreen::ChatDashboard(s) => s.handle_key(key),
                    ActiveScreen::Simulation(s) => s.handle_key(key),
                    ActiveScreen::Results(s) => s.handle_key(key),
                };

                match action {
                    Action::None => {}
                    Action::Quit => {
                        app.running = false;
                    }
                    Action::Retry => {
                        // Re-check connection (splash screen)
                        if let ActiveScreen::Splash(s) = &mut app.screen {
                            match api_client.health_check().await {
                                Ok(true) => match api_client.list_models().await {
                                    Ok(models) => s.set_connected(models.len()),
                                    Err(e) => s.set_error(format!("{}", e)),
                                },
                                _ => s.set_error("Backend not reachable".to_string()),
                            }
                        }
                    }
                    Action::SwitchScreen(screen_id) => {
                        handle_screen_switch(
                            app, api_client, ws_chat, ws_sim, base_url, screen_id,
                        )
                        .await?;
                    }
                }
            }
        }

        // Drain WebSocket events for the active session
        match &mut app.screen {
            ActiveScreen::ChatDashboard(s) => {
                if let Some(ref mut ws) = ws_chat {
                    for _ in 0..500 {
                        let ws_result =
                            tokio::time::timeout(Duration::from_millis(1), ws.next_event()).await;
                        match ws_result {
                            Ok(Some(event)) => s.handle_ws_event(event),
                            Ok(None) => {
                                // WS closed — if chat is done, auto-switch to results
                                if s.is_complete() {
                                    let result = s.build_results();
                                    app.screen =
                                        ActiveScreen::Results(results::ResultsScreen::new(result));
                                }
                                break;
                            }
                            Err(_) => break, // no more events ready
                        }
                    }
                }
            }
            ActiveScreen::Simulation(s) => {
                if let Some(ref mut ws) = ws_sim {
                    for _ in 0..500 {
                        let ws_result =
                            tokio::time::timeout(Duration::from_millis(1), ws.next_event()).await;
                        match ws_result {
                            Ok(Some(event_value)) => {
                                if let Ok(ws_event) = serde_json::from_value::<
                                    simulation::WsEvent,
                                >(event_value)
                                {
                                    s.handle_event(ws_event);
                                }
                            }
                            Ok(None) => {
                                if s.is_running() {
                                    s.force_complete("Connection lost".to_string());
                                }
                                break;
                            }
                            Err(_) => break,
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

async fn handle_screen_switch(
    app: &mut App,
    api_client: &ApiClient,
    ws_chat: &mut Option<WsChatClient>,
    ws_sim: &mut Option<WsSimulationClient>,
    base_url: &str,
    screen_id: ScreenId,
) -> Result<(), Box<dyn std::error::Error>> {
    match screen_id {
        ScreenId::Splash => {
            app.screen = ActiveScreen::Splash(splash::SplashScreen::new());
        }
        ScreenId::Models => {
            let mut models_screen = models::ModelsScreen::new();
            match api_client.list_models().await {
                Ok(models) => models_screen.set_models(models),
                Err(e) => models_screen.set_error(format!("{}", e)),
            }
            app.screen = ActiveScreen::Models(models_screen);
        }
        ScreenId::ModeSelect { selected_models } => {
            app.screen =
                ActiveScreen::ModeSelect(mode_select::ModeSelectScreen::new(selected_models));
        }
        ScreenId::ChatDashboard(config) => {
            // WsChatClient::connect takes base_url, appends /ws/chat internally
            match WsChatClient::connect(base_url).await {
                Ok(mut client) => {
                    let req = WsChatRequest {
                        prompt: config.prompt.clone(),
                        models: config.models.clone(),
                        mode: Some(config.mode.clone()),
                        stream: Some(false),
                        temperature: Some(config.temperature),
                        voting_mode: Some(config.voting_mode.clone()),
                        inference_backend: Some(config.inference_backend.clone()),
                        personalities: None,
                        history: None,
                    };
                    let _ = client.send_chat(req).await;
                    *ws_chat = Some(client);
                }
                Err(e) => {
                    eprintln!("WS chat connect failed: {}", e);
                    return Ok(());
                }
            }
            app.screen = ActiveScreen::ChatDashboard(
                chat_dashboard::ChatDashboardScreen::new(config),
            );
        }
        ScreenId::Simulation(config) => {
            let ws_url = format!(
                "{}/ws/simulation",
                base_url
                    .replace("http://", "ws://")
                    .replace("https://", "wss://")
            );
            match WsSimulationClient::connect(&ws_url).await {
                Ok(mut client) => {
                    let _ = client
                        .send_config(&config.topic, config.agents, config.ticks, &config.model)
                        .await;
                    *ws_sim = Some(client);
                }
                Err(e) => {
                    eprintln!("WS simulation connect failed: {}", e);
                    return Ok(());
                }
            }
            app.screen = ActiveScreen::Simulation(simulation::SimulationScreen::new(
                config.topic,
                config.ticks,
                config.model,
            ));
        }
        ScreenId::Results(result) => {
            app.screen = ActiveScreen::Results(results::ResultsScreen::new(result));
        }
    }
    Ok(())
}

/// Legacy direct simulation loop (for --simulation flag backward compat).
async fn run_simulation_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    sim: &mut simulation::SimulationScreen,
    ws: &mut WsSimulationClient,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal.draw(|f| sim.draw(f, f.area()))?;
        sim.tick_throughput();

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        _ => {
                            let _ = sim.handle_key(key);
                        }
                    }
                }
            }
        }

        // Drain WS events
        for _ in 0..500 {
            let ws_result =
                tokio::time::timeout(Duration::from_millis(1), ws.next_event()).await;
            match ws_result {
                Ok(Some(event_value)) => {
                    if let Ok(ws_event) =
                        serde_json::from_value::<simulation::WsEvent>(event_value)
                    {
                        sim.handle_event(ws_event);
                    }
                }
                Ok(None) => {
                    if sim.is_running() {
                        sim.force_complete("Connection lost".to_string());
                    }
                    // Final render, wait for q
                    terminal.draw(|f| sim.draw(f, f.area()))?;
                    loop {
                        if let Event::Key(key) = event::read()? {
                            if key.kind == KeyEventKind::Press
                                && (key.code == KeyCode::Char('q')
                                    || key.code == KeyCode::Esc)
                            {
                                return Ok(());
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }
}
