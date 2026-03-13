pub mod alpha_zero_train;
pub mod backend;
pub mod config;
pub mod ppo_train;
pub mod replay_buffer;
pub mod self_play;

pub use alpha_zero_train::train as train_alphazero;
pub use config::AlphaZeroConfig;
pub use ppo_train::{PpoConfig, train_ppo};
