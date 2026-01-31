use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};
use crate::tensor::Tensor;

const CHECKPOINT_MAGIC: [u8; 4] = *b"RTCH";
const CHECKPOINT_VERSION: u32 = 1;
const DTYPE_F32: &str = "f32";

pub type StateDict = BTreeMap<String, Tensor>;

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointMetadata {
    version: u32,
    tensors: Vec<TensorMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TensorMetadata {
    name: String,
    shape: Vec<usize>,
    strides: Vec<usize>,
    numel: usize,
    offset_bytes: usize,
    len_bytes: usize,
    dtype: String,
    requires_grad: bool,
}

pub fn save_state_dict<P: AsRef<Path>>(path: P, state: &StateDict) -> Result<()> {
    let mut data_bytes = Vec::new();
    let mut tensors = Vec::with_capacity(state.len());
    let mut offset_bytes = 0usize;

    for (name, tensor) in state.iter() {
        tensor.validate_layout("save_state_dict")?;
        let storage = tensor.storage();
        let len_bytes = storage.data.len() * std::mem::size_of::<f32>();
        let mut encoded = Vec::with_capacity(len_bytes);
        for value in storage.data.iter() {
            encoded.extend_from_slice(&value.to_le_bytes());
        }
        data_bytes.extend_from_slice(&encoded);

        tensors.push(TensorMetadata {
            name: name.clone(),
            shape: tensor.shape().to_vec(),
            strides: tensor.strides().to_vec(),
            numel: tensor.numel(),
            offset_bytes,
            len_bytes,
            dtype: DTYPE_F32.to_string(),
            requires_grad: tensor.requires_grad(),
        });

        offset_bytes = offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(|| TorchError::CheckpointFormat {
                op: "save_state_dict",
                msg: "checkpoint data size overflow".to_string(),
            })?;
    }

    let metadata = CheckpointMetadata {
        version: CHECKPOINT_VERSION,
        tensors,
    };
    let metadata_bytes = serde_json::to_vec(&metadata).map_err(|err| TorchError::CheckpointFormat {
        op: "save_state_dict",
        msg: format!("metadata serialization failed: {err}"),
    })?;
    let metadata_len = metadata_bytes.len() as u64;

    let mut file = File::create(path).map_err(|err| TorchError::CheckpointIo {
        op: "save_state_dict",
        msg: err.to_string(),
    })?;
    file.write_all(&CHECKPOINT_MAGIC)
        .map_err(|err| TorchError::CheckpointIo {
            op: "save_state_dict",
            msg: err.to_string(),
        })?;
    file.write_all(&CHECKPOINT_VERSION.to_le_bytes())
        .map_err(|err| TorchError::CheckpointIo {
            op: "save_state_dict",
            msg: err.to_string(),
        })?;
    file.write_all(&metadata_len.to_le_bytes())
        .map_err(|err| TorchError::CheckpointIo {
            op: "save_state_dict",
            msg: err.to_string(),
        })?;
    file.write_all(&metadata_bytes)
        .map_err(|err| TorchError::CheckpointIo {
            op: "save_state_dict",
            msg: err.to_string(),
        })?;
    file.write_all(&data_bytes)
        .map_err(|err| TorchError::CheckpointIo {
            op: "save_state_dict",
            msg: err.to_string(),
        })?;
    Ok(())
}

pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let mut file = File::open(path).map_err(|err| TorchError::CheckpointIo {
        op: "load_state_dict",
        msg: err.to_string(),
    })?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|err| TorchError::CheckpointIo {
            op: "load_state_dict",
            msg: err.to_string(),
        })?;
    if magic != CHECKPOINT_MAGIC {
        return Err(TorchError::CheckpointFormat {
            op: "load_state_dict",
            msg: "invalid checkpoint magic".to_string(),
        });
    }

    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)
        .map_err(|err| TorchError::CheckpointIo {
            op: "load_state_dict",
            msg: err.to_string(),
        })?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CHECKPOINT_VERSION {
        return Err(TorchError::CheckpointFormat {
            op: "load_state_dict",
            msg: format!(
                "unsupported checkpoint version {version} (expected {CHECKPOINT_VERSION})"
            ),
        });
    }

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|err| TorchError::CheckpointIo {
            op: "load_state_dict",
            msg: err.to_string(),
        })?;
    let metadata_len = u64::from_le_bytes(len_bytes) as usize;
    let mut metadata_buf = vec![0u8; metadata_len];
    file.read_exact(&mut metadata_buf)
        .map_err(|err| TorchError::CheckpointIo {
            op: "load_state_dict",
            msg: err.to_string(),
        })?;
    let metadata: CheckpointMetadata =
        serde_json::from_slice(&metadata_buf).map_err(|err| TorchError::CheckpointFormat {
            op: "load_state_dict",
            msg: format!("metadata parse failed: {err}"),
        })?;
    if metadata.version != CHECKPOINT_VERSION {
        return Err(TorchError::CheckpointFormat {
            op: "load_state_dict",
            msg: format!(
                "metadata version {} does not match header {CHECKPOINT_VERSION}",
                metadata.version
            ),
        });
    }

    let mut data_bytes = Vec::new();
    file.read_to_end(&mut data_bytes)
        .map_err(|err| TorchError::CheckpointIo {
            op: "load_state_dict",
            msg: err.to_string(),
        })?;

    let mut state = StateDict::new();
    for tensor_meta in metadata.tensors {
        if tensor_meta.dtype != DTYPE_F32 {
            return Err(TorchError::CheckpointDtypeMismatch {
                op: "load_state_dict",
                name: tensor_meta.name,
                expected: DTYPE_F32.to_string(),
                actual: tensor_meta.dtype,
            });
        }
        let end = tensor_meta
            .offset_bytes
            .checked_add(tensor_meta.len_bytes)
            .ok_or_else(|| TorchError::CheckpointFormat {
                op: "load_state_dict",
                msg: "checkpoint tensor range overflow".to_string(),
            })?;
        if end > data_bytes.len() {
            return Err(TorchError::CheckpointFormat {
                op: "load_state_dict",
                msg: format!(
                    "tensor {} out of bounds (end {end} > {})",
                    tensor_meta.name,
                    data_bytes.len()
                ),
            });
        }
        if tensor_meta.len_bytes % std::mem::size_of::<f32>() != 0 {
            return Err(TorchError::CheckpointFormat {
                op: "load_state_dict",
                msg: format!("tensor {} byte length is not aligned", tensor_meta.name),
            });
        }

        let numel = tensor_meta.len_bytes / std::mem::size_of::<f32>();
        if numel != tensor_meta.numel {
            return Err(TorchError::CheckpointShapeMismatch {
                op: "load_state_dict",
                name: tensor_meta.name,
                expected: vec![tensor_meta.numel],
                actual: vec![numel],
            });
        }
        let expected_numel: usize = tensor_meta.shape.iter().product();
        if expected_numel != tensor_meta.numel {
            return Err(TorchError::CheckpointShapeMismatch {
                op: "load_state_dict",
                name: tensor_meta.name,
                expected: tensor_meta.shape.clone(),
                actual: vec![tensor_meta.numel],
            });
        }

        let slice = &data_bytes[tensor_meta.offset_bytes..end];
        let mut values = Vec::with_capacity(numel);
        for chunk in slice.chunks_exact(std::mem::size_of::<f32>()) {
            let array: [u8; 4] = chunk
                .try_into()
                .map_err(|_| TorchError::CheckpointFormat {
                    op: "load_state_dict",
                    msg: format!("tensor {} chunk decode failed", tensor_meta.name),
                })?;
            values.push(f32::from_le_bytes(array));
        }

        let tensor = Tensor::from_vec_f32(values, &tensor_meta.shape, None, tensor_meta.requires_grad);
        if tensor.strides() != tensor_meta.strides.as_slice() {
            return Err(TorchError::CheckpointLayoutMismatch {
                op: "load_state_dict",
                name: tensor_meta.name,
                expected: tensor.strides().to_vec(),
                actual: tensor_meta.strides,
            });
        }
        state.insert(tensor_meta.name, tensor);
    }

    Ok(state)
}
