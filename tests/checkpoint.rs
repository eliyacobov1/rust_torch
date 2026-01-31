use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use rand::Rng;

use rustorch::checkpoint::{load_state_dict, save_state_dict, StateDict};
use rustorch::tensor::Tensor;

fn temp_checkpoint_path() -> PathBuf {
    let mut rng = rand::thread_rng();
    let suffix: u64 = rng.gen();
    std::env::temp_dir().join(format!("rustorch_checkpoint_{suffix}.bin"))
}

#[test]
fn save_and_load_state_dict_roundtrip() {
    let mut state: StateDict = BTreeMap::new();
    state.insert(
        "linear.weight".to_string(),
        Tensor::from_vec_f32(vec![1.0, -2.0, 3.5, 4.0], &[2, 2], None, true),
    );
    state.insert(
        "linear.bias".to_string(),
        Tensor::from_vec_f32(vec![0.5, -0.5], &[2], None, true),
    );

    let path = temp_checkpoint_path();
    save_state_dict(&path, &state).expect("save_state_dict should succeed");
    let loaded = load_state_dict(&path).expect("load_state_dict should succeed");

    let weight = loaded.get("linear.weight").expect("missing weight");
    let bias = loaded.get("linear.bias").expect("missing bias");

    assert_eq!(weight.shape(), &[2, 2]);
    assert_eq!(bias.shape(), &[2]);
    assert_eq!(weight.storage().data, vec![1.0, -2.0, 3.5, 4.0]);
    assert_eq!(bias.storage().data, vec![0.5, -0.5]);

    fs::remove_file(path).ok();
}
