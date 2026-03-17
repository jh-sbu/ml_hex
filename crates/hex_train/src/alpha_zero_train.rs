use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::activation,
};
use hex_core::{CELLS, SIZE};

use hex_agents::model::HexNet;

use crate::{
    backend::TrainBackend,
    config::AlphaZeroConfig,
    replay_buffer::{Experience, ReplayBuffer},
    self_play::generate_episode,
};

/// Run the full AlphaZero training loop.
///
/// Prints loss after each generation and saves checkpoints to disk.
pub fn train(config: AlphaZeroConfig) -> Result<(), Box<dyn std::error::Error>> {
    let device: <TrainBackend as Backend>::Device = Default::default();
    let mut net = HexNet::<TrainBackend>::new(&device);

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init::<TrainBackend, HexNet<TrainBackend>>();

    let mut replay = ReplayBuffer::new(config.replay_capacity);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    for generation in 0..config.generations {
        // ---- Self-play: collect experience ----
        for _ in 0..config.games_per_gen {
            let exps = generate_episode(&net, &device, &config);
            for exp in exps {
                replay.push(exp);
            }
        }

        if replay.is_empty() {
            println!("gen {generation}: replay buffer empty, skipping grad steps");
            continue;
        }

        // ---- Gradient updates ----
        let mut last_loss = 0.0f32;
        for _ in 0..config.gradient_steps {
            let batch: Vec<&Experience> = replay.sample_batch(config.batch_size);
            let n = batch.len();

            // Build input tensor [n, 3, 11, 11]
            let state_data: Vec<f32> =
                batch.iter().flat_map(|e| e.state_tensor.iter().copied()).collect();
            let x: Tensor<TrainBackend, 4> = Tensor::from_data(
                TensorData::new(state_data, vec![n, 3, SIZE, SIZE]),
                &device,
            );

            // Policy targets [n, CELLS]
            let pol_data: Vec<f32> =
                batch.iter().flat_map(|e| e.visit_dist.iter().copied()).collect();
            let policy_targets: Tensor<TrainBackend, 2> =
                Tensor::from_data(TensorData::new(pol_data, vec![n, CELLS]), &device);

            // Value targets [n, 1]
            let val_data: Vec<f32> = batch.iter().map(|e| e.value_target).collect();
            let value_targets: Tensor<TrainBackend, 2> =
                Tensor::from_data(TensorData::new(val_data, vec![n, 1]), &device);

            // Forward pass
            let (pol_logits, val) = net.forward(x);

            // Policy loss: cross-entropy with soft targets
            //   loss = -mean over batch of sum_a( p(a) * log_softmax(logits)[a] )
            let log_probs = activation::log_softmax(pol_logits, 1);
            let policy_loss = -(policy_targets * log_probs).sum_dim(1).mean();

            // Value loss: MSE
            let value_loss = MseLoss::new().forward(val, value_targets, Reduction::Mean);

            let loss = policy_loss + value_loss;

            // Read loss scalar before backward (clones the graph node).
            last_loss = loss.clone().into_scalar();

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            net = optim.step(config.learning_rate, net, grads_params);
        }

        println!("gen {generation}: loss = {last_loss:.4}");

        // ---- Checkpointing ----
        if (generation + 1) % config.checkpoint_every == 0 {
            net.clone()
                .save_file(format!("{}/ckpt_gen{generation}", config.checkpoint_dir), &recorder)?;
            net.clone().save_file(format!("{}/ckpt_latest", config.checkpoint_dir), &recorder)?;
        }
    }

    // Final checkpoint
    net.save_file(format!("{}/ckpt_latest", config.checkpoint_dir), &recorder)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_runs() {
        let config = AlphaZeroConfig {
            generations: 2,
            games_per_gen: 1,
            simulations: 2,
            replay_capacity: 500,
            batch_size: 8,
            gradient_steps: 1,
            checkpoint_every: 10, // don't write checkpoint files during test
            ..Default::default()
        };
        train(config).expect("training should not panic");
    }
}
