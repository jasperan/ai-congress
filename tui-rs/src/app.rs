use crate::screens::*;

pub enum ActiveScreen {
    Splash(splash::SplashScreen),
    Models(models::ModelsScreen),
    ModeSelect(mode_select::ModeSelectScreen),
    ChatDashboard(chat_dashboard::ChatDashboardScreen),
    Simulation(simulation::SimulationScreen),
    Results(results::ResultsScreen),
}

pub struct App {
    pub screen: ActiveScreen,
    pub base_url: String,
    pub running: bool,
    pub show_help: bool,
}

impl App {
    pub fn new(base_url: String) -> Self {
        Self {
            screen: ActiveScreen::Splash(splash::SplashScreen::new()),
            base_url,
            running: true,
            show_help: false,
        }
    }
}
