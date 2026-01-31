use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum TorchError {
    #[error("{op}: shapes not broadcastable: {lhs:?} vs {rhs:?}")]
    BroadcastMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    #[error("{op}: non-contiguous layout not supported (shape={shape:?}, strides={strides:?})")]
    NonContiguous {
        op: &'static str,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    #[error("{op}: invalid layout (shape={shape:?}, strides={strides:?}): {msg}")]
    InvalidLayout {
        op: &'static str,
        shape: Vec<usize>,
        strides: Vec<usize>,
        msg: String,
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
