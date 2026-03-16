use std::sync::{
    mpsc::{self, Receiver},
    Arc,
};
use std::thread;

use hex_agents::Agent;
use hex_core::{Cell, GameState, HexError, Move};

use crate::config::PlayerConfig;

pub struct GameRunner {
    pub state: GameState,
    pub red: PlayerConfig,
    pub blue: PlayerConfig,
    agent_rx: Option<Receiver<Move>>,
}

impl GameRunner {
    pub fn new(red: PlayerConfig, blue: PlayerConfig, swap: bool) -> Self {
        let state = if swap { GameState::new_with_swap() } else { GameState::new() };
        Self {
            state,
            red,
            blue,
            agent_rx: None,
        }
    }

    pub fn current_is_human(&self) -> bool {
        match self.state.current_player() {
            Cell::Red => self.red.is_human(),
            Cell::Blue => self.blue.is_human(),
            Cell::Empty => unreachable!(),
        }
    }

    /// Spawn agent thread if not already pending. Call once per agent turn.
    pub fn kick_agent(&mut self) {
        if self.agent_rx.is_some() {
            return;
        }
        let agent: Arc<dyn Agent> = match self.state.current_player() {
            Cell::Red => {
                if let PlayerConfig::Agent(a) = &self.red {
                    Arc::clone(a)
                } else {
                    return;
                }
            }
            Cell::Blue => {
                if let PlayerConfig::Agent(a) = &self.blue {
                    Arc::clone(a)
                } else {
                    return;
                }
            }
            Cell::Empty => unreachable!(),
        };
        let state = self.state.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(agent.select_move(&state));
        });
        self.agent_rx = Some(rx);
    }

    /// Poll for an agent move. Returns Some(Move) if ready.
    pub fn poll_agent(&mut self) -> Option<Move> {
        let mv = self.agent_rx.as_ref()?.try_recv().ok()?;
        self.agent_rx = None;
        Some(mv)
    }

    /// Apply a move. Returns Err on illegal move.
    pub fn apply(&mut self, mv: Move) -> Result<(), HexError> {
        self.state = self.state.apply_move(mv)?;
        Ok(())
    }
}
