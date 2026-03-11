pub mod config;
pub mod error;
mod app;
mod board_widget;
mod input;
mod runner;

pub use config::{PlayerConfig, TuiConfig};
pub use error::{Result, TuiError};

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

pub fn run(config: TuiConfig) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = app::App::new(config).run(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use board_widget::BoardView;
    use hex_agents::RandomAgent;
    use hex_core::{GameState, Move};
    use ratatui::{backend::TestBackend, widgets::Widget, Terminal};

    use crate::runner::GameRunner;

    #[test]
    fn test_board_render() {
        let state = GameState::new()
            .apply_move(Move { row: 0, col: 0 })
            .unwrap(); // Red plays (0,0)

        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                BoardView {
                    state: &state,
                    cursor: (5, 5),
                }
                .render(f.area(), f.buffer_mut());
            })
            .unwrap();

        // row=0, col=0 → x = area.x(0) + row(0) + col*3(0) + 1 = 1, y = 0
        let cell = terminal.backend().buffer().cell((1u16, 0u16)).unwrap();
        assert_eq!(cell.symbol(), "R");
    }

    #[test]
    fn test_cursor_render() {
        let state = GameState::new();
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                BoardView {
                    state: &state,
                    cursor: (0, 0),
                }
                .render(f.area(), f.buffer_mut());
            })
            .unwrap();

        // cursor at (0,0) → '*' at position x=1, y=0
        let cell = terminal.backend().buffer().cell((1u16, 0u16)).unwrap();
        assert_eq!(cell.symbol(), "*");
    }

    #[test]
    fn test_full_game_agents() {
        let mut runner = GameRunner::new(
            PlayerConfig::agent(RandomAgent),
            PlayerConfig::agent(RandomAgent),
        );
        loop {
            if runner.state.is_terminal() {
                break;
            }
            runner.kick_agent();
            loop {
                if let Some(mv) = runner.poll_agent() {
                    runner.apply(mv).unwrap();
                    break;
                }
            }
        }
        assert!(runner.state.winner().is_some());
    }
}
