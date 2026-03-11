use hex_core::{GameState, Move};

use crate::Agent;

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "Random"
    }

    fn select_move(&self, state: &GameState) -> Move {
        use rand::prelude::IndexedRandom as _;
        let moves: Vec<Move> = state.legal_moves().collect();
        let mut rng = rand::rng();
        *moves
            .choose(&mut rng)
            .expect("no legal moves: state is terminal")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_core::GameState;
    use rand::Rng as _;

    #[test]
    fn test_random_legal() {
        let agent = RandomAgent;
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut state = GameState::new();
            let moves_to_apply = rng.random_range(0..=60usize);
            for _ in 0..moves_to_apply {
                if state.is_terminal() {
                    break;
                }
                let mv = agent.select_move(&state);
                state = state.apply_move(mv).expect("random agent returned illegal move");
            }
            if !state.is_terminal() {
                let mv = agent.select_move(&state);
                assert!(
                    state.apply_move(mv).is_ok(),
                    "RandomAgent returned an illegal move"
                );
            }
        }
    }
}
