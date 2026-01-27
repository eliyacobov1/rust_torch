use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum TorchError {
    #[error("{op}: shapes not broadcastable: {lhs:?} vs {rhs:?}")]
    BroadcastMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    #[error("{op}: dim {dim} out of range for rank {rank}")]
    InvalidDim {
        op: &'static str,
        dim: isize,
        rank: usize,
    },
    #[error("{op}: {msg}")]
    InvalidArgument { op: &'static str, msg: String },
}

pub type Result<T> = std::result::Result<T, TorchError>;
