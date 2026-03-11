#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TuiError>;
