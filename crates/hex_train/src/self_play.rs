use hex_core::{Cell, GameState};

use hex_agents::{
    alpha_zero::run_puct,
    model::{HexNet, state_to_vec},
};

use crate::{backend::{InferBackend, TrainBackend}, config::AlphaZeroConfig, replay_buffer::Experience};

/// Run one complete self-play game.
///
/// Both players share the same network (symmetric). Returns one [`Experience`]
/// per move, with `value_target` filled in after the game ends.
pub fn generate_episode(
    net: &HexNet<TrainBackend>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    config: &AlphaZeroConfig,
) -> Vec<Experience> {
    // Switch to inference mode to avoid gradient bookkeeping during self-play.
    use burn::module::AutodiffModule as _;
    let infer_net: HexNet<InferBackend> = net.valid();
    let infer_device: <InferBackend as burn::prelude::Backend>::Device = Default::default();

    let _ = device; // keep signature compatible; NdArray device is always CPU

    let mut state = GameState::new();
    // Accumulate (state_vec, visit_dist, player_at_that_move).
    let mut records: Vec<(Vec<f32>, Vec<f32>, Cell)> = Vec::new();

    let mut move_index: u32 = 0;

    loop {
        if state.is_terminal() {
            break;
        }
        // Clone net for PUCT (takes ownership).
        let net_copy = infer_net.clone();
        let (mv, visit_dist) = run_puct(
            net_copy,
            &state,
            &infer_device,
            config.simulations,
            config.c_puct,
        );

        let state_vec = state_to_vec(&state);
        let player = state.current_player();

        // Choose move: temperature=1 (sample) for early moves, greedy later.
        let chosen_mv = if move_index < config.temperature_moves {
            // Sample proportional to visit_dist (already computed in run_puct).
            sample_from_dist(&visit_dist, mv)
        } else {
            mv
        };

        records.push((state_vec, visit_dist, player));
        state = state.apply_move(chosen_mv).unwrap();
        move_index += 1;
    }

    let winner = state.winner().expect("loop exits on terminal");

    records
        .into_iter()
        .map(|(state_vec, visit_dist, player)| {
            let value_target = if player == winner { 1.0f32 } else { -1.0 };
            Experience::new(state_vec, visit_dist, value_target)
        })
        .collect()
}

/// Sample a move index from `dist`, falling back to `default_mv` if dist is all-zero.
fn sample_from_dist(dist: &[f32], default_mv: hex_core::Move) -> hex_core::Move {
    #[allow(unused_imports)]
    use rand::Rng as _;

    // Build (index, weight) pairs for non-zero entries.
    let weighted: Vec<(usize, f32)> = dist
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(i, w)| (i, *w))
        .collect();

    if weighted.is_empty() {
        return default_mv;
    }

    let mut rng = rand::rng();
    // Use weighted sampling: pick index proportional to weight.
    let total: f32 = weighted.iter().map(|(_, w)| w).sum();
    let threshold: f32 = rng.random();
    let threshold = threshold * total;
    let mut acc = 0.0f32;
    for (idx, w) in &weighted {
        acc += w;
        if acc >= threshold {
            let row = (*idx / hex_core::SIZE) as u8;
            let col = (*idx % hex_core::SIZE) as u8;
            return hex_core::Move { row, col };
        }
    }
    default_mv
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_agents::model::HexNet;

    #[test]
    fn test_experience_collection() {
        let device: <TrainBackend as burn::prelude::Backend>::Device = Default::default();
        let net = HexNet::<TrainBackend>::new(&device);
        let config = AlphaZeroConfig {
            simulations: 2,
            games_per_gen: 1,
            ..Default::default()
        };
        let exps = generate_episode(&net, &device, &config);
        assert!(!exps.is_empty(), "episode should produce experiences");
        for exp in &exps {
            let sum: f32 = exp.visit_dist.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "visit_dist should sum to ~1.0, got {sum}"
            );
        }
    }
}
