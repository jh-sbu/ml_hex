use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation,
};
use hex_core::{Cell, GameState, SIZE};

/// Inference-only backend: pure-CPU NdArray.
pub type InferBackend = burn::backend::NdArray<f32>;

// ---------------------------------------------------------------------------
// Residual block
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    relu: Relu,
}

impl<B: Backend> ResBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        let pad = PaddingConfig2d::Explicit(1, 1);
        Self {
            conv1: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(pad.clone())
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            conv2: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(pad)
                .init(device),
            bn2: BatchNormConfig::new(64).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let skip = x.clone();
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        activation::relu(x + skip)
    }
}

// ---------------------------------------------------------------------------
// HexNet
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HexNet<B: Backend> {
    // Stem
    stem_conv: Conv2d<B>,
    stem_bn: BatchNorm<B>,
    stem_relu: Relu,
    // 4 residual blocks
    res_blocks: Vec<ResBlock<B>>,
    // Policy head
    pol_conv: Conv2d<B>, // 64→2, k=1
    pol_bn: BatchNorm<B>,
    pol_relu: Relu,
    pol_fc: Linear<B>, // 2*11*11=242 → 121
    // Value head
    val_conv: Conv2d<B>, // 64→1, k=1
    val_bn: BatchNorm<B>,
    val_relu: Relu,
    val_fc1: Linear<B>, // 121 → 64
    val_relu2: Relu,
    val_fc2: Linear<B>, // 64 → 1
}

impl<B: Backend> HexNet<B> {
    pub fn new(device: &B::Device) -> Self {
        let stem_pad = PaddingConfig2d::Explicit(1, 1);
        Self {
            stem_conv: Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(stem_pad)
                .init(device),
            stem_bn: BatchNormConfig::new(64).init(device),
            stem_relu: Relu::new(),
            res_blocks: (0..4).map(|_| ResBlock::new(device)).collect(),
            pol_conv: Conv2dConfig::new([64, 2], [1, 1]).init(device),
            pol_bn: BatchNormConfig::new(2).init(device),
            pol_relu: Relu::new(),
            pol_fc: LinearConfig::new(2 * SIZE * SIZE, SIZE * SIZE).init(device),
            val_conv: Conv2dConfig::new([64, 1], [1, 1]).init(device),
            val_bn: BatchNormConfig::new(1).init(device),
            val_relu: Relu::new(),
            val_fc1: LinearConfig::new(SIZE * SIZE, 64).init(device),
            val_relu2: Relu::new(),
            val_fc2: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` – board tensor of shape `[batch, 3, 11, 11]`
    ///
    /// # Returns
    /// `(policy_logits [batch, 121], value [batch, 1])`
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Stem
        let x = self.stem_conv.forward(x);
        let x = self.stem_bn.forward(x);
        let x = self.stem_relu.forward(x);

        // Residual blocks
        let mut x = x;
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        // Policy head
        let p = self.pol_conv.forward(x.clone());
        let p = self.pol_bn.forward(p);
        let p = self.pol_relu.forward(p);
        let [batch, _, _, _] = p.dims();
        let p = p.reshape([batch, 2 * SIZE * SIZE]);
        let p = self.pol_fc.forward(p);

        // Value head
        let v = self.val_conv.forward(x);
        let v = self.val_bn.forward(v);
        let v = self.val_relu.forward(v);
        let [batch, _, _, _] = v.dims();
        let v = v.reshape([batch, SIZE * SIZE]);
        let v = self.val_fc1.forward(v);
        let v = self.val_relu2.forward(v);
        let v = self.val_fc2.forward(v);
        let v = activation::tanh(v);

        (p, v)
    }
}

// ---------------------------------------------------------------------------
// State encoding helpers
// ---------------------------------------------------------------------------

/// Encode a `GameState` as a `[3, 11, 11]` float tensor.
///
/// - Channel 0: Red cells (1.0 where Red, else 0.0)
/// - Channel 1: Blue cells (1.0 where Blue, else 0.0)
/// - Channel 2: Current player (all 1.0 = Red to move, all 0.0 = Blue to move)
pub fn encode_state<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 3> {
    let data = state_to_vec(state);
    Tensor::<B, 1>::from_data(TensorData::new(data, vec![3 * SIZE * SIZE]), device)
        .reshape([3, SIZE, SIZE])
}

/// Encode a `GameState` as a flat `Vec<f32>` of length 363 (3 × 11 × 11).
///
/// Same channel layout as [`encode_state`].
pub fn state_to_vec(state: &GameState) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * SIZE * SIZE];
    let player_val = if state.current_player() == Cell::Red {
        1.0f32
    } else {
        0.0f32
    };
    for r in 0..SIZE {
        for c in 0..SIZE {
            let idx = r * SIZE + c;
            match state.cell_at(r, c) {
                Cell::Red => data[idx] = 1.0,
                Cell::Blue => data[SIZE * SIZE + idx] = 1.0,
                Cell::Empty => {}
            }
            data[2 * SIZE * SIZE + idx] = player_val;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_net_forward_shape() {
        let device = Default::default();
        let net = HexNet::<InferBackend>::new(&device);

        // Build a batch of 2 random state tensors [2, 3, 11, 11]
        let x = Tensor::<InferBackend, 4>::zeros([2, 3, SIZE, SIZE], &device);
        let (pol, val) = net.forward(x);

        assert_eq!(pol.dims(), [2, SIZE * SIZE], "policy shape mismatch");
        assert_eq!(val.dims(), [2, 1], "value shape mismatch");
    }

    #[test]
    fn test_encode_state_channels() {
        let device = Default::default();
        let state = GameState::new()
            .apply_move(hex_core::Move { row: 0, col: 0 })
            .unwrap();

        let tensor = encode_state::<InferBackend>(&state, &device);
        let data = tensor.into_data();
        let vals: Vec<f32> = data.to_vec().unwrap();

        // ch0: Red cell at (0,0) → index 0
        assert_eq!(vals[0], 1.0, "ch0[0][0] should be 1.0 for Red");
        // ch1: no Blue cells → index SIZE*SIZE + 0 = 121
        assert_eq!(vals[SIZE * SIZE], 0.0, "ch1[0][0] should be 0.0");
        // After Red's move, it's Blue's turn → ch2 should be all 0.0
        assert_eq!(vals[2 * SIZE * SIZE], 0.0, "ch2 should be 0.0 (Blue to move)");
    }
}
