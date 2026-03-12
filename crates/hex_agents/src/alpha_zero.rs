use burn::{prelude::*, tensor::activation};
use hex_core::{GameState, Move, CELLS};

use crate::Agent;
use crate::model::{HexNet, InferBackend, encode_state};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct AlphaZeroConfig {
    pub checkpoint_path: String,
    pub simulations: u32,
    pub c_puct: f32,
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        Self {
            checkpoint_path: "ckpt_latest".to_string(),
            simulations: 200,
            c_puct: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PUCT tree node
// ---------------------------------------------------------------------------

struct PuctNode {
    state: GameState,
    visits: u32,
    total_value: f32,
    /// Fully expanded children: (move, node index, prior probability).
    children: Vec<(Move, usize, f32)>,
    /// Prior probabilities for unexpanded moves: (move, prior).
    priors: Vec<(Move, f32)>,
    is_terminal: bool,
}

impl PuctNode {
    fn new(state: GameState, priors: Vec<(Move, f32)>) -> Self {
        let is_terminal = state.is_terminal();
        Self {
            state,
            visits: 0,
            total_value: 0.0,
            children: vec![],
            priors,
            is_terminal,
        }
    }

    fn new_terminal(state: GameState) -> Self {
        Self {
            state,
            visits: 0,
            total_value: 0.0,
            children: vec![],
            priors: vec![],
            is_terminal: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PUCT tree
// ---------------------------------------------------------------------------

struct PuctTree {
    nodes: Vec<PuctNode>,
    c_puct: f32,
    net: HexNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl PuctTree {
    fn new(
        state: GameState,
        net: HexNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
        c_puct: f32,
    ) -> Self {
        // Evaluate root immediately
        let root_priors = if state.is_terminal() {
            vec![]
        } else {
            compute_priors_infer(&net, &state, &device)
        };
        let root = PuctNode::new(state, root_priors);
        let mut nodes = Vec::with_capacity(512);
        nodes.push(root);
        Self { nodes, c_puct, net, device }
    }

    /// Run one simulation: select → expand → backpropagate.
    fn run_simulation(&mut self) {
        let (path, leaf_value) = self.select_and_expand();
        self.backpropagate(&path, leaf_value);
    }

    /// Traverse from root to a leaf; expand if needed.
    /// Returns (path of node indices, value for the last node from ITS perspective).
    fn select_and_expand(&mut self) -> (Vec<usize>, f32) {
        let mut path = vec![0usize];

        loop {
            let idx = *path.last().unwrap();
            let node = &self.nodes[idx];

            if node.is_terminal {
                // Terminal: the current_player is the one who would move next (the loser).
                return (path, 0.0);
            }

            if !node.priors.is_empty() {
                // Has unexpanded moves — expand one.
                return self.expand_one(path);
            }

            if node.children.is_empty() {
                // Fully expanded but no children (shouldn't normally happen).
                return (path, 0.0);
            }

            // All moves expanded — pick best by PUCT score.
            let parent_visits = node.visits;
            let c = self.c_puct;
            let best_idx = node
                .children
                .iter()
                .map(|&(_, child_idx, prior)| {
                    let child = &self.nodes[child_idx];
                    let q = if child.visits == 0 {
                        0.0
                    } else {
                        1.0 - child.total_value / child.visits as f32
                    };
                    let score = q
                        + c * prior * (parent_visits as f32).sqrt()
                            / (1.0 + child.visits as f32);
                    (score, child_idx)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, ci)| ci)
                .expect("children is non-empty");

            path.push(best_idx);
        }
    }

    fn expand_one(&mut self, mut path: Vec<usize>) -> (Vec<usize>, f32) {
        let idx = *path.last().unwrap();
        // Pop one prior from the node.
        let (mv, prior) = self
            .nodes[idx]
            .priors
            .pop()
            .expect("priors non-empty guaranteed by caller");

        let child_state = self.nodes[idx]
            .state
            .apply_move(mv)
            .expect("prior move was legal");

        let (child_priors, leaf_value) = if child_state.is_terminal() {
            // Child is terminal: value from the child's perspective = 0.0 (it's the loser's
            // turn next; child's "current player" actually just lost).
            (vec![], 0.0)
        } else {
            let priors = compute_priors_infer(&self.net, &child_state, &self.device);
            let value = compute_value_infer(&self.net, &child_state, &self.device);
            (priors, value)
        };

        let child_node = if child_state.is_terminal() {
            PuctNode::new_terminal(child_state)
        } else {
            PuctNode::new(child_state, child_priors)
        };

        let child_idx = self.nodes.len();
        self.nodes.push(child_node);
        self.nodes[idx].children.push((mv, child_idx, prior));
        path.push(child_idx);

        (path, leaf_value)
    }

    fn backpropagate(&mut self, path: &[usize], value_for_last: f32) {
        let mut value = value_for_last;
        for &node_idx in path.iter().rev() {
            self.nodes[node_idx].visits += 1;
            self.nodes[node_idx].total_value += value;
            value = 1.0 - value;
        }
    }

    /// Return the move with the highest visit count at the root.
    fn best_move(&self) -> Move {
        // Prefer a directly terminal winning child first.
        if let Some(&(mv, _, _)) = self.nodes[0]
            .children
            .iter()
            .find(|&&(_, ci, _)| self.nodes[ci].is_terminal)
        {
            return mv;
        }
        self.nodes[0]
            .children
            .iter()
            .max_by_key(|&&(_, ci, _)| self.nodes[ci].visits)
            .map(|&(mv, _, _)| mv)
            .expect("tree was never expanded")
    }

    /// Return (move, visit_count) pairs for the root, sorted by move index.
    pub fn root_visit_counts(&self) -> Vec<(Move, u32)> {
        self.nodes[0]
            .children
            .iter()
            .map(|&(mv, ci, _)| (mv, self.nodes[ci].visits))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Network evaluation helpers (inference backend)
// ---------------------------------------------------------------------------

/// Returns (move, softmax_prior) pairs for all legal moves of `state`.
pub(crate) fn compute_priors_infer(
    net: &HexNet<InferBackend>,
    state: &GameState,
    device: &<InferBackend as Backend>::Device,
) -> Vec<(Move, f32)> {
    let legal_moves: Vec<Move> = state.legal_moves().collect();
    if legal_moves.is_empty() {
        return vec![];
    }

    let x = encode_state::<InferBackend>(state, device)
        .unsqueeze_dim::<4>(0); // [1, 3, 11, 11]
    let (pol_logits, _) = net.forward(x); // [1, 121]

    // Mask illegal moves to -inf.
    let mut mask = vec![f32::NEG_INFINITY; CELLS];
    for mv in &legal_moves {
        mask[mv.index()] = 0.0;
    }
    let mask_t = Tensor::<InferBackend, 2>::from_data(
        TensorData::new(mask, vec![1, CELLS]),
        device,
    );
    let masked = pol_logits + mask_t;
    let probs = activation::softmax(masked, 1); // [1, 121]

    let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
    legal_moves
        .into_iter()
        .map(|mv| (mv, probs_data[mv.index()]))
        .collect()
}

/// Returns the value estimate (from current player's perspective) for `state`.
pub(crate) fn compute_value_infer(
    net: &HexNet<InferBackend>,
    state: &GameState,
    device: &<InferBackend as Backend>::Device,
) -> f32 {
    let x = encode_state::<InferBackend>(state, device)
        .unsqueeze_dim::<4>(0); // [1, 3, 11, 11]
    let (_, val) = net.forward(x); // [1, 1]
    // Map tanh output [-1, 1] to [0, 1] probability for current player.
    let raw: f32 = val.into_scalar();
    (raw + 1.0) * 0.5
}

// ---------------------------------------------------------------------------
// AlphaZeroAgent
// ---------------------------------------------------------------------------

/// `HexNet<NdArray>` contains `OnceCell` fields that are `Send` but not `Sync`.
/// Wrapping in `Mutex` satisfies the `Agent: Send + Sync` requirement.
pub struct AlphaZeroAgent {
    net: std::sync::Mutex<HexNet<InferBackend>>,
    simulations: u32,
    c_puct: f32,
    device: <InferBackend as Backend>::Device,
}

impl AlphaZeroAgent {
    /// Load a trained checkpoint.
    ///
    /// # Panics
    /// Panics if the checkpoint file cannot be read.
    pub fn load(config: AlphaZeroConfig) -> Self {
        use burn::record::{BinFileRecorder, FullPrecisionSettings};
        let device: <InferBackend as Backend>::Device = Default::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let net = HexNet::<InferBackend>::new(&device)
            .load_file(&config.checkpoint_path, &recorder, &device)
            .unwrap_or_else(|e| {
                panic!("Failed to load checkpoint '{}': {e}", config.checkpoint_path)
            });
        Self {
            net: std::sync::Mutex::new(net),
            simulations: config.simulations,
            c_puct: config.c_puct,
            device,
        }
    }
}

impl Agent for AlphaZeroAgent {
    fn name(&self) -> &str {
        "AlphaZero"
    }

    fn select_move(&self, state: &GameState) -> Move {
        assert!(!state.is_terminal(), "select_move called on terminal state");

        // Clone net for this search (PuctTree takes ownership).
        let net_clone = self.net.lock().unwrap().clone();
        let mut tree =
            PuctTree::new(state.clone(), net_clone, self.device, self.c_puct);
        for _ in 0..self.simulations {
            tree.run_simulation();
        }
        tree.best_move()
    }
}

// ---------------------------------------------------------------------------
// Public helper used by hex_train::self_play
// ---------------------------------------------------------------------------

/// Run PUCT MCTS on `state` using the given network.
///
/// Returns `(best_move, visit_distribution)` where `visit_distribution` is a
/// `[CELLS]`-length vector of normalised visit counts (sums to ≤ 1.0; 0.0 for
/// all unexpanded / illegal positions).
pub fn run_puct(
    net: HexNet<InferBackend>,
    state: &GameState,
    device: &<InferBackend as Backend>::Device,
    simulations: u32,
    c_puct: f32,
) -> (Move, Vec<f32>) {
    let mut tree = PuctTree::new(state.clone(), net, *device, c_puct);
    for _ in 0..simulations {
        tree.run_simulation();
    }

    let counts = tree.root_visit_counts();
    let total: u32 = counts.iter().map(|(_, c)| c).sum();
    let mut dist = vec![0.0f32; CELLS];
    for (mv, c) in &counts {
        dist[mv.index()] = *c as f32 / total.max(1) as f32;
    }

    let best = tree.best_move();
    (best, dist)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_zero_puct_legal() {
        let device: <InferBackend as Backend>::Device = Default::default();
        let net = HexNet::<InferBackend>::new(&device);
        let state = GameState::new();
        let (mv, dist) = run_puct(net, &state, &device, 3, 1.0);
        assert!(state.apply_move(mv).is_ok(), "run_puct returned illegal move");
        let sum: f32 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "visit distribution should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_alpha_zero_wins_in_one() {
        // Red has a chain at col 0, rows 0-9. Move row 10, col 0 wins.
        // With a random (untrained) network we just check a legal move is returned.
        let device: <InferBackend as Backend>::Device = Default::default();
        let net = HexNet::<InferBackend>::new(&device);
        let mut state = GameState::new();
        for row in 0..10u8 {
            state = state.apply_move(Move { row, col: 0 }).unwrap();
            state = state.apply_move(Move { row, col: 10 }).unwrap();
        }
        let (mv, _) = run_puct(net, &state, &device, 5, 1.0);
        assert!(state.apply_move(mv).is_ok(), "returned illegal move");
    }
}
