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
    #[error("{op}: overlapping layout (shape={shape:?}, strides={strides:?}): {msg}")]
    OverlappingLayout {
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
    #[error("checkpoint {op}: {msg}")]
    CheckpointIo { op: &'static str, msg: String },
    #[error("checkpoint {op}: {msg}")]
    CheckpointFormat { op: &'static str, msg: String },
    #[error("checkpoint {op}: dtype mismatch for {name} (expected {expected}, got {actual})")]
    CheckpointDtypeMismatch {
        op: &'static str,
        name: String,
        expected: String,
        actual: String,
    },
    #[error("checkpoint {op}: shape mismatch for {name} (expected {expected:?}, got {actual:?})")]
    CheckpointShapeMismatch {
        op: &'static str,
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("checkpoint {op}: layout mismatch for {name} (expected {expected:?}, got {actual:?})")]
    CheckpointLayoutMismatch {
        op: &'static str,
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("autograd {op}: {msg}")]
    Autograd { op: &'static str, msg: String },
    #[error("experiment {op}: {msg}")]
    Experiment { op: &'static str, msg: String },
    #[error("data {op}: {msg}")]
    Data { op: &'static str, msg: String },
    #[error("optimizer {op}: {msg}")]
    Optimizer { op: &'static str, msg: String },
    #[error("training {op}: {msg}")]
    Training { op: &'static str, msg: String },
}

pub type Result<T> = std::result::Result<T, TorchError>;
