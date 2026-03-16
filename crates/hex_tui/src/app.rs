use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use hex_core::{Move, SIZE};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use crate::{
    board_widget::BoardView,
    config::TuiConfig,
    error::Result,
    input::{key_to_action, AppAction},
    runner::GameRunner,
};

enum AppState {
    Playing,
    WaitingForAgent,
    GameOver(String),
}

pub struct App {
    runner: GameRunner,
    cursor: (u8, u8),
    app_state: AppState,
}

impl App {
    pub fn new(config: TuiConfig) -> Self {
        let mut app = App {
            runner: GameRunner::new(config.red, config.blue, config.swap),
            cursor: (5, 5),
            app_state: AppState::Playing,
        };
        if !app.runner.current_is_human() {
            app.runner.kick_agent();
            app.app_state = AppState::WaitingForAgent;
        }
        app
    }

    pub fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<()> {
        loop {
            terminal.draw(|f| self.draw(f))?;

            if let Some(mv) = self.runner.poll_agent() {
                self.apply_move(mv);
            }

            if event::poll(Duration::from_millis(10))?
                && let Event::Key(key) = event::read()?
            {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key_to_action(key) {
                    AppAction::Quit => return Ok(()),
                    AppAction::MoveCursor(dr, dc) => self.move_cursor(dr, dc),
                    AppAction::PlaceMove => {
                        if matches!(self.app_state, AppState::Playing)
                            && self.runner.current_is_human()
                        {
                            let mv = Move {
                                row: self.cursor.0,
                                col: self.cursor.1,
                            };
                            self.apply_move(mv);
                        }
                    }
                    AppAction::None => {}
                }
            }
        }
    }

    fn apply_move(&mut self, mv: Move) {
        if self.runner.apply(mv).is_err() {
            return;
        }
        if self.runner.state.is_terminal() {
            let winner = self
                .runner
                .state
                .winner()
                .map(|c| format!("{c:?}"))
                .unwrap_or_default();
            self.app_state = AppState::GameOver(format!("{winner} wins!"));
            return;
        }
        if self.runner.current_is_human() {
            self.app_state = AppState::Playing;
        } else {
            self.runner.kick_agent();
            self.app_state = AppState::WaitingForAgent;
        }
    }

    fn move_cursor(&mut self, dr: i8, dc: i8) {
        let r = (self.cursor.0 as i8 + dr).clamp(0, SIZE as i8 - 1) as u8;
        let c = (self.cursor.1 as i8 + dc).clamp(0, SIZE as i8 - 1) as u8;
        self.cursor = (r, c);
    }

    fn draw(&self, f: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(14), Constraint::Length(3)])
            .split(f.area());

        f.render_widget(
            BoardView {
                state: &self.runner.state,
                cursor: self.cursor,
            },
            chunks[0],
        );

        let status = match &self.app_state {
            AppState::Playing => format!(
                "{:?} to move | arrows/hjkl: move  enter/space: place  q: quit",
                self.runner.state.current_player()
            ),
            AppState::WaitingForAgent => "Agent is thinking\u{2026}  (q: quit)".into(),
            AppState::GameOver(msg) => format!("{msg}  (q: quit)"),
        };
        f.render_widget(
            Paragraph::new(status).block(Block::default().borders(Borders::ALL).title("Status")),
            chunks[1],
        );
    }
}
