use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, activation},
};
use hex_core::{CELLS, SIZE, GameState};

use hex_agents::model::{HexNet, state_to_vec};

use crate::backend::{InferBackend, TrainBackend};

// ---------------------------------------------------------------------------
// PpoConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub generations: u32,
    pub episodes_per_gen: u32,
    pub ppo_epochs: u32,
    pub batch_size: usize,
    pub clip_eps: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    pub learning_rate: f64,
    pub checkpoint_every: u32,
    /// Directory to write checkpoint files into.
    pub checkpoint_dir: String,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            generations: 50,
            episodes_per_gen: 20,
            ppo_epochs: 4,
            batch_size: 256,
            clip_eps: 0.2,
            gamma: 1.0,
            gae_lambda: 0.95,
            entropy_coeff: 0.01,
            value_coeff: 0.5,
            learning_rate: 1e-3,
            checkpoint_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// PpoStep
// ---------------------------------------------------------------------------

struct PpoStep {
    state_tensor: Vec<f32>,
    legal_mask: Vec<bool>,
    action: usize,
    log_prob_old: f32,
    value_old: f32,
    reward: f32,
}

// ---------------------------------------------------------------------------
// Episode collection
// ---------------------------------------------------------------------------

fn collect_episode(
    net: &HexNet<InferBackend>,
    device: &<InferBackend as Backend>::Device,
) -> Vec<PpoStep> {
    let mut state = GameState::new();
    let mut steps: Vec<PpoStep> = Vec::new();

    loop {
        if state.is_terminal() {
            break;
        }

        let state_vec = state_to_vec(&state);
        let legal_moves: Vec<hex_core::Move> = state.legal_moves().collect();

        let mut legal_mask = vec![false; CELLS];
        for mv in &legal_moves {
            legal_mask[mv.index()] = true;
        }

        // Build [1, 3, 11, 11] input tensor
        let x = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(state_vec.clone(), vec![1, 3, SIZE, SIZE]),
            device,
        );
        let (logits, value) = net.forward(x); // [1, 121], [1, 1]

        // Apply mask
        let mask_vals: Vec<f32> = legal_mask
            .iter()
            .map(|&legal| if legal { 0.0f32 } else { -1e9f32 })
            .collect();
        let mask_t = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(mask_vals, vec![1, CELLS]),
            device,
        );
        let masked_logits = logits + mask_t;

        let log_probs_data: Vec<f32> = activation::log_softmax(masked_logits.clone(), 1)
            .into_data()
            .to_vec()
            .unwrap();
        let probs_data: Vec<f32> = activation::softmax(masked_logits, 1)
            .into_data()
            .to_vec()
            .unwrap();

        let value_old: f32 = value.into_scalar();

        let action = sample_action_from_probs(&probs_data);
        let log_prob_old = log_probs_data[action];

        steps.push(PpoStep {
            state_tensor: state_vec,
            legal_mask,
            action,
            log_prob_old,
            value_old,
            reward: 0.0,
        });

        let mv = hex_core::Move {
            row: (action / SIZE) as u8,
            col: (action % SIZE) as u8,
        };
        state = state.apply_move(mv).unwrap();
    }

    // The player who made the last move won — give them +1
    if let Some(last) = steps.last_mut() {
        last.reward = 1.0;
    }

    steps
}

fn sample_action_from_probs(probs: &[f32]) -> usize {
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
// GAE computation
// ---------------------------------------------------------------------------

/// Compute Generalized Advantage Estimation for one episode.
///
/// Returns `Vec<(advantage, return_)>` with one entry per step.
///
/// The negation of `next_value` accounts for the opponent's perspective:
/// `V(s_{t+1})` from the current player's POV = `-V(s_{t+1})` from the
/// opponent's POV (who is the current player at step t+1).
fn compute_gae(
    steps: &[PpoStep],
    gamma: f32,
    lambda: f32,
) -> Vec<(f32, f32)> {
    let n = steps.len();
    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];
    let mut gae = 0.0f32;
    let mut next_value = 0.0f32; // terminal state value = 0

    for t in (0..n).rev() {
        let delta =
            steps[t].reward + gamma * (-next_value) - steps[t].value_old;
        gae = delta + gamma * lambda * gae;
        advantages[t] = gae;
        returns[t] = gae + steps[t].value_old;
        next_value = steps[t].value_old;
    }

    advantages.into_iter().zip(returns).collect()
}

// ---------------------------------------------------------------------------
// PPO mini-batch update
// ---------------------------------------------------------------------------

fn ppo_update(
    net: HexNet<TrainBackend>,
    optim: &mut impl Optimizer<HexNet<TrainBackend>, TrainBackend>,
    steps: &[&PpoStep],
    gae_data: &[&(f32, f32)],
    config: &PpoConfig,
    device: &<TrainBackend as Backend>::Device,
) -> (HexNet<TrainBackend>, f32, f32) {
    let n = steps.len();

    // State input [n, 3, 11, 11]
    let state_data: Vec<f32> = steps
        .iter()
        .flat_map(|s| s.state_tensor.iter().copied())
        .collect();
    let x: Tensor<TrainBackend, 4> = Tensor::from_data(
        TensorData::new(state_data, vec![n, 3, SIZE, SIZE]),
        device,
    );

    // Action indices [n, 1] as Int
    let action_data: Vec<i64> = steps.iter().map(|s| s.action as i64).collect();
    let actions_t = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(action_data, vec![n, 1]),
        device,
    );

    // Old log probs [n]
    let log_probs_old_data: Vec<f32> =
        steps.iter().map(|s| s.log_prob_old).collect();
    let log_probs_old: Tensor<TrainBackend, 1> = Tensor::from_data(
        TensorData::new(log_probs_old_data, vec![n]),
        device,
    );

    // Normalized advantages [n] (computed on CPU to avoid Burn API variability)
    let raw_adv: Vec<f32> = gae_data.iter().map(|g| g.0).collect();
    let mean_adv: f32 = raw_adv.iter().sum::<f32>() / n as f32;
    let var_adv: f32 =
        raw_adv.iter().map(|a| (a - mean_adv).powi(2)).sum::<f32>() / n as f32;
    let std_adv = var_adv.sqrt() + 1e-8;
    let norm_adv: Vec<f32> = raw_adv.iter().map(|a| (a - mean_adv) / std_adv).collect();
    let adv: Tensor<TrainBackend, 1> = Tensor::from_data(
        TensorData::new(norm_adv, vec![n]),
        device,
    );

    // Return targets [n, 1]
    let ret_data: Vec<f32> = gae_data.iter().map(|g| g.1).collect();
    let returns_t: Tensor<TrainBackend, 2> = Tensor::from_data(
        TensorData::new(ret_data, vec![n, 1]),
        device,
    );

    // Legal mask [n, 121]
    let mask_data: Vec<f32> = steps
        .iter()
        .flat_map(|s| {
            s.legal_mask
                .iter()
                .map(|&legal| if legal { 0.0f32 } else { -1e9f32 })
        })
        .collect();
    let mask_t: Tensor<TrainBackend, 2> = Tensor::from_data(
        TensorData::new(mask_data, vec![n, CELLS]),
        device,
    );

    // Forward pass
    let (logits, new_values) = net.forward(x); // [n, 121], [n, 1]

    // Apply legal mask
    let masked_logits = logits + mask_t;

    // Log probs and gather at actions
    let log_probs_new_full =
        activation::log_softmax(masked_logits.clone(), 1); // [n, 121]
    let log_prob_new: Tensor<TrainBackend, 1> = log_probs_new_full
        .gather(1, actions_t)
        .reshape([n]); // [n, 1] → [n]

    // Ratio = exp(log_π_new - log_π_old)
    let ratio = (log_prob_new - log_probs_old).exp(); // [n]

    // PPO clipped policy loss: -mean(min(ratio*adv, clip(ratio)*adv))
    // Using: min(a,b) = a - relu(a-b)
    let unclipped = ratio.clone() * adv.clone();
    let clipped =
        ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv;
    let policy_loss =
        -(unclipped.clone() - activation::relu(unclipped - clipped)).mean();

    // Value loss: 0.5 * MSE
    let value_loss =
        MseLoss::new().forward(new_values, returns_t, Reduction::Mean);

    // Entropy: -sum(p * log_p)
    let probs_for_ent = activation::softmax(masked_logits.clone(), 1);
    let log_probs_for_ent = activation::log_softmax(masked_logits, 1);
    let entropy = -(probs_for_ent * log_probs_for_ent).sum_dim(1).mean();

    let total_loss = policy_loss.clone()
        + value_loss.clone().mul_scalar(config.value_coeff)
        - entropy.mul_scalar(config.entropy_coeff);

    let policy_loss_scalar = policy_loss.into_scalar();
    let value_loss_scalar = value_loss.into_scalar();

    let grads = total_loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &net);
    let net = optim.step(config.learning_rate, net, grads_params);

    (net, policy_loss_scalar, value_loss_scalar)
}

// ---------------------------------------------------------------------------
// train_ppo outer loop
// ---------------------------------------------------------------------------

pub fn train_ppo(config: PpoConfig) -> Result<(), Box<dyn std::error::Error>> {
    let device: <TrainBackend as Backend>::Device = Default::default();
    let infer_device: <InferBackend as Backend>::Device = Default::default();

    let mut net = HexNet::<TrainBackend>::new(&device);
    let mut optim = AdamConfig::new()
        .init::<TrainBackend, HexNet<TrainBackend>>();

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    for generation in 0..config.generations {
        // Switch to inference mode for episode collection
        use burn::module::AutodiffModule as _;
        let infer_net: HexNet<InferBackend> = net.valid();

        let mut all_steps: Vec<PpoStep> = Vec::new();
        let mut all_gae: Vec<(f32, f32)> = Vec::new();

        for _ in 0..config.episodes_per_gen {
            let steps = collect_episode(&infer_net, &infer_device);
            let gae = compute_gae(&steps, config.gamma, config.gae_lambda);
            all_steps.extend(steps);
            all_gae.extend(gae);
        }

        let n_total = all_steps.len();
        if n_total == 0 {
            println!("gen {generation}: no steps collected, skipping");
            continue;
        }

        let mut last_policy_loss = 0.0f32;
        let mut last_value_loss = 0.0f32;

        for _ in 0..config.ppo_epochs {
            let mut indices: Vec<usize> = (0..n_total).collect();
            shuffle_indices(&mut indices);

            for chunk in indices.chunks(config.batch_size) {
                let batch_steps: Vec<&PpoStep> =
                    chunk.iter().map(|&i| &all_steps[i]).collect();
                let batch_gae: Vec<&(f32, f32)> =
                    chunk.iter().map(|&i| &all_gae[i]).collect();

                let (updated_net, pl, vl) = ppo_update(
                    net,
                    &mut optim,
                    &batch_steps,
                    &batch_gae,
                    &config,
                    &device,
                );
                net = updated_net;
                last_policy_loss = pl;
                last_value_loss = vl;
            }
        }

        println!(
            "gen {generation}: policy_loss={last_policy_loss:.4} value_loss={last_value_loss:.4}"
        );

        if generation % config.checkpoint_every == 0 {
            net.clone()
                .save_file(format!("{}/ppo_ckpt_gen{generation}", config.checkpoint_dir), &recorder)?;
            net.clone().save_file(format!("{}/ppo_ckpt_latest", config.checkpoint_dir), &recorder)?;
        }
    }

    net.save_file(format!("{}/ppo_ckpt_latest", config.checkpoint_dir), &recorder)?;
    Ok(())
}

fn shuffle_indices(indices: &mut [usize]) {
    use rand::seq::SliceRandom as _;
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_step(reward: f32, value_old: f32) -> PpoStep {
        PpoStep {
            state_tensor: vec![],
            legal_mask: vec![],
            action: 0,
            log_prob_old: 0.0,
            value_old,
            reward,
        }
    }

    #[test]
    fn test_gae_computation() {
        // Two-step episode: step 0 reward=0, step 1 reward=+1.
        // gamma=1, lambda=0 → GAE = delta (no lookahead beyond one step)
        // step 1: delta = 1 + 1*(-0) - 0 = 1; adv=1, ret=1
        // step 0: delta = 0 + 1*(-0) - 0 = 0; adv=0, ret=0
        let steps = vec![
            make_dummy_step(0.0, 0.0),
            make_dummy_step(1.0, 0.0),
        ];
        let gae = compute_gae(&steps, 1.0, 0.0);
        assert_eq!(gae.len(), 2);
        assert!(
            (gae[0].0 - 0.0).abs() < 1e-5,
            "step 0 advantage should be 0, got {}",
            gae[0].0
        );
        assert!(
            (gae[1].0 - 1.0).abs() < 1e-5,
            "step 1 advantage should be 1, got {}",
            gae[1].0
        );
        assert!(
            (gae[0].1 - 0.0).abs() < 1e-5,
            "step 0 return should be 0, got {}",
            gae[0].1
        );
        assert!(
            (gae[1].1 - 1.0).abs() < 1e-5,
            "step 1 return should be 1, got {}",
            gae[1].1
        );
    }

    #[test]
    fn test_ppo_training_runs() {
        let config = PpoConfig {
            generations: 2,
            episodes_per_gen: 2,
            ppo_epochs: 1,
            batch_size: 8,
            checkpoint_every: 999, // no disk writes during test
            ..Default::default()
        };
        train_ppo(config).expect("PPO training should not panic");
    }
}
