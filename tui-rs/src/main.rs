mod app;
mod theme;
mod ui;
mod ws;

use std::io;
use std::time::Duration;

use app::App;
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ws::WsClient;

#[derive(Parser, Debug)]
#[command(name = "congress-tui", about = "Real-time TUI for AI Congress simulations")]
struct Args {
    /// WebSocket server URL
    #[arg(long, default_value = "ws://localhost:8000/ws/simulation")]
    server: String,

    /// Bill/topic for the simulation
    #[arg(long, default_value = "Should AI systems be regulated by federal law?")]
    topic: String,

    /// Number of agents (max 10)
    #[arg(long, default_value_t = 10)]
    agents: u32,

    /// Number of simulation ticks
    #[arg(long, default_value_t = 100)]
    ticks: u32,

    /// Ollama model to use
    #[arg(long, default_value = "qwen3.5:9b")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let agents = args.agents.min(10);

    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    // Connect to WebSocket
    let mut ws = match WsClient::connect(&args.server).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("Failed to connect to {}: {}", args.server, e);
            eprintln!("Make sure the AI Congress backend is running: python run_server.py");
            std::process::exit(1);
        }
    };

    // Send simulation config
    ws.send_config(&args.topic, agents, args.ticks, &args.model)
        .await?;

    // Initialize terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Create app state
    let mut app = App::new(args.topic.clone(), args.ticks, args.model.clone());

    // Main event loop
    let result = run_loop(&mut terminal, &mut app, &mut ws).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {}", e);
    }

    // Print final result if simulation completed
    if let Some(ref result_str) = app.simulation_result {
        println!("\n{}", result_str);
        println!(
            "Final vote: YEA {} / NAY {} / ABSTAIN {}",
            app.yea_count, app.nay_count, app.abstain_count
        );
    }

    Ok(())
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    ws: &mut WsClient,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Render
        terminal.draw(|f| ui::draw(f, app))?;

        // Tick throughput ring buffer
        app.tick_throughput();

        // Poll for terminal events (non-blocking, 30ms timeout)
        if event::poll(Duration::from_millis(30))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Tab => app.toggle_layout(),
                        KeyCode::Char('j') | KeyCode::Down => {
                            if app.layout_mode == app::LayoutMode::Focus {
                                app.next_agent();
                            } else {
                                app.scroll_feed_down();
                            }
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            if app.layout_mode == app::LayoutMode::Focus {
                                app.prev_agent();
                            } else {
                                app.scroll_feed_up();
                            }
                        }
                        KeyCode::PageDown => {
                            for _ in 0..10 {
                                app.scroll_feed_down();
                            }
                        }
                        KeyCode::PageUp => {
                            for _ in 0..10 {
                                app.scroll_feed_up();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Try to receive WebSocket events (non-blocking via tokio::select with timeout)
        let ws_result = tokio::time::timeout(Duration::from_millis(10), ws.next_event()).await;

        match ws_result {
            Ok(Some(event_value)) => {
                if let Ok(ws_event) = serde_json::from_value::<app::WsEvent>(event_value) {
                    app.handle_event(ws_event);
                }
            }
            Ok(None) => {
                // WebSocket closed
                if app.running {
                    app.running = false;
                    if app.simulation_result.is_none() {
                        app.simulation_result = Some("Connection lost".to_string());
                    }
                }
                // Do one final render then wait for 'q'
                terminal.draw(|f| ui::draw(f, app))?;
                loop {
                    if let Event::Key(key) = event::read()? {
                        if key.kind == KeyEventKind::Press
                            && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
                        {
                            return Ok(());
                        }
                    }
                }
            }
            Err(_) => {
                // Timeout — no WS event available, continue loop
            }
        }
    }
}
