use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HexError {
    #[error("cell ({row}, {col}) is already occupied")]
    CellOccupied { row: u8, col: u8 },
    #[error("move ({row}, {col}) is out of bounds")]
    OutOfBounds { row: u8, col: u8 },
    #[error("swap is not available in this game")]
    SwapNotAvailable,
    #[error("game is already over")]
    GameOver,
}
