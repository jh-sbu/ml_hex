/// Configuration for the AlphaZero self-play training loop.
#[derive(Debug, Clone)]
pub struct AlphaZeroConfig {
    /// Number of training generations.
    pub generations: u32,
    /// Self-play games collected per generation.
    pub games_per_gen: u32,
    /// PUCT simulations per move during self-play.
    pub simulations: u32,
    /// Maximum replay buffer capacity (evicts oldest when full).
    pub replay_capacity: usize,
    /// Mini-batch size for gradient updates.
    pub batch_size: usize,
    /// Gradient steps per generation.
    pub gradient_steps: u32,
    /// Save a checkpoint every N generations.
    pub checkpoint_every: u32,
    /// Directory to write checkpoint files into.
    pub checkpoint_dir: String,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Moves played with temperature = 1 (sample from visit dist); greedy after.
    pub temperature_moves: u32,
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        Self {
            generations: 10,
            games_per_gen: 100,
            simulations: 200,
            replay_capacity: 10_000,
            batch_size: 256,
            gradient_steps: 100,
            checkpoint_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
            learning_rate: 1e-3,
            c_puct: 1.0,
            temperature_moves: 30,
        }
    }
}
