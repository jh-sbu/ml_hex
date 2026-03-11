mod heuristic;
mod random;

pub use heuristic::HeuristicAgent;
pub use random::RandomAgent;

use hex_core::{GameState, Move};

pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    /// # Panics
    /// Panics if called on a terminal state.
    fn select_move(&self, state: &GameState) -> Move;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_core::Cell;

    pub(crate) fn play_game(red: &dyn Agent, blue: &dyn Agent) -> Cell {
        let mut state = GameState::new();
        loop {
            if state.is_terminal() {
                return state.winner().unwrap();
            }
            let mv = if state.current_player() == Cell::Red {
                red.select_move(&state)
            } else {
                blue.select_move(&state)
            };
            state = state.apply_move(mv).unwrap();
        }
    }

    #[test]
    fn test_heuristic_vs_random_winrate() {
        let heuristic = HeuristicAgent;
        let random = RandomAgent;
        let wins: u32 = (0..100)
            .map(|_| if play_game(&heuristic, &random) == Cell::Red { 1 } else { 0 })
            .sum();
        assert!(wins > 70, "HeuristicAgent won {wins}/100 games (expected >70)");
    }
}
