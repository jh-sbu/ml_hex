/// CPU inference backend (no gradient tracking).
pub type InferBackend = burn::backend::NdArray<f32>;

/// Training backend — wraps `InferBackend` with automatic differentiation.
pub type TrainBackend = burn::backend::Autodiff<InferBackend>;
