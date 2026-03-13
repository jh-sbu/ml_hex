use burn::{
    prelude::*,
    tensor::activation,
};
use hex_core::{GameState, Move, CELLS, SIZE};

use crate::Agent;
use crate::model::{HexNet, InferBackend, encode_state};

// ---------------------------------------------------------------------------
// PpoAgent
// ---------------------------------------------------------------------------

/// `HexNet<NdArray>` contains `OnceCell` fields that are `Send` but not `Sync`.
/// Wrapping in `Mutex` satisfies the `Agent: Send + Sync` requirement.
pub struct PpoAgent {
    net: std::sync::Mutex<HexNet<InferBackend>>,
    greedy: bool,
}

impl PpoAgent {
    pub fn new(net: HexNet<InferBackend>, greedy: bool) -> Self {
        Self {
            net: std::sync::Mutex::new(net),
            greedy,
        }
    }

    /// Load from a BinFileRecorder checkpoint (same format as AlphaZeroAgent).
    ///
    /// # Panics
    /// Panics if the checkpoint file cannot be read.
    pub fn load(checkpoint_path: &str, greedy: bool) -> Self {
        use burn::record::{BinFileRecorder, FullPrecisionSettings};
        let device: <InferBackend as Backend>::Device = Default::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let net = HexNet::<InferBackend>::new(&device)
            .load_file(checkpoint_path, &recorder, &device)
            .unwrap_or_else(|e| {
                panic!("Failed to load checkpoint '{checkpoint_path}': {e}")
            });
        Self::new(net, greedy)
    }
}

impl Agent for PpoAgent {
    fn name(&self) -> &str {
        "PpoAgent"
    }

    fn select_move(&self, state: &GameState) -> Move {
        assert!(!state.is_terminal(), "select_move called on terminal state");

        let device: <InferBackend as Backend>::Device = Default::default();
        let net = self.net.lock().unwrap();

        let x = encode_state::<InferBackend>(state, &device)
            .unsqueeze_dim::<4>(0); // [1, 3, 11, 11]
        let (logits, _) = net.forward(x); // [1, 121]

        // Mask illegal moves to -1e9
        let legal_moves: Vec<Move> = state.legal_moves().collect();
        let mut mask_vals = vec![-1e9f32; CELLS];
        for mv in &legal_moves {
            mask_vals[mv.index()] = 0.0;
        }
        let mask_t = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(mask_vals, vec![1, CELLS]),
            &device,
        );
        let masked = logits + mask_t;

        let probs_data: Vec<f32> = activation::softmax(masked, 1)
            .into_data()
            .to_vec()
            .unwrap();

        let action_idx = if self.greedy {
            probs_data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            sample_action(&probs_data)
        };

        Move {
            row: (action_idx / SIZE) as u8,
            col: (action_idx % SIZE) as u8,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sample_action(probs: &[f32]) -> usize {
    use rand::Rng as _;
    let mut rng = rand::rng();
    let threshold: f32 = rng.random();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= threshold {
            return i;
        }
    }
    // Fallback: last non-zero index
    probs
        .iter()
        .enumerate()
        .rfind(|(_, p)| **p > 0.0)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_legal() {
        let device = Default::default();
        let net = HexNet::<InferBackend>::new(&device);
        let agent = PpoAgent::new(net, false);
        let state = GameState::new();
        let mv = agent.select_move(&state);
        let legal: Vec<Move> = state.legal_moves().collect();
        assert!(legal.contains(&mv), "PpoAgent returned illegal move {mv:?}");
    }

    #[test]
    fn test_ppo_greedy_legal() {
        let device = Default::default();
        let net = HexNet::<InferBackend>::new(&device);
        let agent = PpoAgent::new(net, true);
        let state = GameState::new();
        let mv = agent.select_move(&state);
        let legal: Vec<Move> = state.legal_moves().collect();
        assert!(legal.contains(&mv), "PpoAgent (greedy) returned illegal move {mv:?}");
    }
}
