use hex_core::{GameState, Move};

use crate::Agent;

pub struct MctsConfig {
    pub rollout_budget: u32,
    pub exploration_constant: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            rollout_budget: 1000,
            exploration_constant: 1.4,
        }
    }
}

pub struct MctsAgent {
    config: MctsConfig,
}

impl MctsAgent {
    pub fn new(config: MctsConfig) -> Self {
        Self { config }
    }
}

impl Agent for MctsAgent {
    fn name(&self) -> &str {
        "MCTS"
    }

    fn select_move(&self, state: &GameState) -> Move {
        assert!(!state.is_terminal(), "select_move called on terminal state");
        let mut tree = MctsTree::new(state.clone(), &self.config);
        for _ in 0..self.config.rollout_budget {
            tree.run_one_simulation();
        }
        tree.best_move()
    }
}

struct MctsNode {
    state: GameState,
    visits: u32,
    total_value: f32,
    children: Vec<(Move, usize)>,
    unexplored: Vec<Move>,
    is_terminal: bool,
}

impl MctsNode {
    fn new(state: GameState) -> Self {
        let is_terminal = state.is_terminal();
        let unexplored: Vec<Move> = if is_terminal {
            vec![]
        } else {
            state.legal_moves().collect()
        };
        Self {
            state,
            visits: 0,
            total_value: 0.0,
            children: vec![],
            unexplored,
            is_terminal,
        }
    }
}

struct MctsTree {
    nodes: Vec<MctsNode>,
    exploration_constant: f32,
}

impl MctsTree {
    fn new(state: GameState, config: &MctsConfig) -> Self {
        let root = MctsNode::new(state);
        let mut nodes = Vec::with_capacity(config.rollout_budget as usize * 2);
        nodes.push(root);
        Self {
            nodes,
            exploration_constant: config.exploration_constant,
        }
    }

    /// Select a path from root to a node that can be expanded (has unexplored moves)
    /// or is terminal. Returns (path, can_expand).
    fn select(&self) -> (Vec<usize>, bool) {
        let mut path = vec![0usize];
        loop {
            let node_idx = *path.last().unwrap();
            let node = &self.nodes[node_idx];

            if node.is_terminal || !node.unexplored.is_empty() {
                let can_expand = !node.is_terminal && !node.unexplored.is_empty();
                return (path, can_expand);
            }

            // All children expanded — pick best by UCB1
            let parent_visits = node.visits;
            let c = self.exploration_constant;
            let best_child = node
                .children
                .iter()
                .map(|&(_, child_idx)| {
                    let child = &self.nodes[child_idx];
                    let score = if child.visits == 0 {
                        f32::INFINITY
                    } else {
                        (1.0 - child.total_value / child.visits as f32)
                            + c * ((parent_visits as f32).ln() / child.visits as f32).sqrt()
                    };
                    (score, child_idx)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, idx)| idx);

            match best_child {
                Some(idx) => path.push(idx),
                None => return (path, false),
            }
        }
    }

    /// Expand one unexplored move from the leaf node, return the new child index.
    fn expand(&mut self, leaf_idx: usize) -> usize {
        let mv = self.nodes[leaf_idx]
            .unexplored
            .pop()
            .expect("expand called with no unexplored moves");
        let child_state = self.nodes[leaf_idx]
            .state
            .apply_move(mv)
            .expect("unexplored move was illegal");
        let child_idx = self.nodes.len();
        self.nodes.push(MctsNode::new(child_state));
        self.nodes[leaf_idx].children.push((mv, child_idx));
        child_idx
    }

    /// Rollout from `state`. Returns 1.0 if `player_at_start` wins, 0.0 otherwise.
    ///
    /// Plays randomly, but always takes an immediate winning move when one exists.
    /// Uses `GameState::winning_move` (O(CELLS × 6) amortized, no cloning) to keep
    /// each rollout step cheap while dramatically reducing noise near terminal positions.
    fn simulate(state: &GameState) -> f32 {
        use rand::prelude::IndexedRandom as _;

        let player = state.current_player();
        let mut cur = state.clone();
        let mut rng = rand::rng();

        loop {
            if cur.is_terminal() {
                return if cur.winner().unwrap() == player { 1.0 } else { 0.0 };
            }
            let mv = if let Some(wm) = cur.winning_move() {
                wm
            } else {
                let moves: Vec<Move> = cur.legal_moves().collect();
                *moves.choose(&mut rng).expect("no legal moves but not terminal")
            };
            cur = cur.apply_move(mv).expect("legal move rejected");
        }
    }

    /// Backpropagate a value up the path, flipping perspective at each edge.
    fn backpropagate_path(&mut self, path: &[usize], value_for_last: f32) {
        let mut value = value_for_last;
        for &node_idx in path.iter().rev() {
            self.nodes[node_idx].visits += 1;
            self.nodes[node_idx].total_value += value;
            value = 1.0 - value;
        }
    }

    fn run_one_simulation(&mut self) {
        let (mut path, can_expand) = self.select();
        let leaf_idx = *path.last().unwrap();

        if self.nodes[leaf_idx].is_terminal {
            // Terminal node — the current_player() is the *next* mover (the loser).
            // Value for this node = 0.0 (the node's player lost).
            self.backpropagate_path(&path, 0.0);
            return;
        }

        if can_expand {
            let child_idx = self.expand(leaf_idx);
            path.push(child_idx);

            if self.nodes[child_idx].is_terminal {
                // Child is already terminal: child's current_player() is the loser → value = 0.0
                self.backpropagate_path(&path, 0.0);
            } else {
                let sim_value = Self::simulate(&self.nodes[child_idx].state.clone());
                self.backpropagate_path(&path, sim_value);
            }
        } else {
            // Fully expanded non-terminal with no children (shouldn't normally happen)
            let sim_value = Self::simulate(&self.nodes[leaf_idx].state.clone());
            self.backpropagate_path(&path, sim_value);
        }
    }

    /// Pick the best move for the root player.
    ///
    /// First checks for an immediate winning move (terminal child). Falls back
    /// to the most-visited child (robust standard MCTS criterion).
    fn best_move(&self) -> Move {
        // Immediate win: return it without needing visit counts.
        if let Some(&(mv, _)) = self.nodes[0]
            .children
            .iter()
            .find(|&&(_, child_idx)| self.nodes[child_idx].is_terminal)
        {
            return mv;
        }
        self.nodes[0]
            .children
            .iter()
            .max_by_key(|&&(_, child_idx)| self.nodes[child_idx].visits)
            .map(|&(mv, _)| mv)
            .expect("no children: tree was never expanded")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HeuristicAgent, RandomAgent};
    use hex_core::{Cell, GameState, Move};
    use rand::Rng as _;

    fn play_game(red: &dyn Agent, blue: &dyn Agent) -> Cell {
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
    fn test_mcts_legal() {
        let agent = MctsAgent::new(MctsConfig {
            rollout_budget: 50,
            exploration_constant: 1.4,
        });
        let mut rng = rand::rng();
        for _ in 0..20 {
            let mut state = GameState::new();
            let moves_to_apply = rng.random_range(0..=40usize);
            for _ in 0..moves_to_apply {
                if state.is_terminal() {
                    break;
                }
                let mv = RandomAgent.select_move(&state);
                state = state.apply_move(mv).unwrap();
            }
            if !state.is_terminal() {
                let mv = agent.select_move(&state);
                assert!(
                    state.apply_move(mv).is_ok(),
                    "MctsAgent returned an illegal move"
                );
            }
        }
    }

    #[test]
    fn test_mcts_wins_in_one() {
        // Red has a chain at col 0, rows 0–9. Blue is at col 10. Red to move.
        let mut state = GameState::new();
        for row in 0..10u8 {
            state = state.apply_move(Move { row, col: 0 }).unwrap();
            state = state.apply_move(Move { row, col: 10 }).unwrap();
        }
        let agent = MctsAgent::new(MctsConfig {
            rollout_budget: 200,
            exploration_constant: 1.4,
        });
        let mv = agent.select_move(&state);
        assert_eq!(mv, Move { row: 10, col: 0 }, "MCTS must see the winning move");
    }

    #[test]
    fn test_mcts_vs_random_winrate() {
        let mcts = MctsAgent::new(MctsConfig {
            rollout_budget: 500,
            exploration_constant: 1.4,
        });
        let random = RandomAgent;
        let wins: u32 = (0..20)
            .map(|_| if play_game(&mcts, &random) == Cell::Red { 1 } else { 0 })
            .sum();
        assert!(wins >= 17, "MctsAgent won {wins}/20 vs RandomAgent (expected >=17)");
    }

    #[test]
    fn test_mcts_vs_heuristic_winrate() {
        let mcts = MctsAgent::new(MctsConfig {
            rollout_budget: 1000,
            exploration_constant: 1.4,
        });
        let heuristic = HeuristicAgent;
        let wins: u32 = (0..20)
            .map(|_| if play_game(&mcts, &heuristic) == Cell::Red { 1 } else { 0 })
            .sum();
        assert!(wins >= 13, "MctsAgent won {wins}/20 vs HeuristicAgent (expected >=13)");
    }
}
