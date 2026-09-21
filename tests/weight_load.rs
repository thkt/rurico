#![cfg(feature = "test-mlx")]

use mlx_rs::{
    module::{ModuleParameters, ModuleParametersExt},
    random,
};
use rurico::{
    modernbert::{Config, ModernBert},
    sandbox::require_unsandboxed_mlx_runtime,
};

// A small native MLX round trip exercises successful metadata validation and
// assignment. Fixed official header fixtures independently check the schema.
#[test]
fn checked_load_restores_all_saved_parameters_after_reinitialization() {
    require_unsandboxed_mlx_runtime();
    let mut config: Config = serde_json::from_str(include_str!(
        "fixtures/modernbert_configs/ruri-v3-310m.json"
    ))
    .unwrap();
    config.vocab_size = 8;
    config.hidden_size = 4;
    config.intermediate_size = 6;
    config.num_attention_heads = 2;
    config.num_hidden_layers = 2;

    random::seed(11).unwrap();
    let original = ModernBert::new(&config).unwrap();
    let file = tempfile::Builder::new()
        .suffix(".safetensors")
        .tempfile()
        .unwrap();
    original.save_safetensors(file.path()).unwrap();
    let expected = original.parameters().flatten();

    random::seed(22).unwrap();
    let loaded = ModernBert::load(file.path(), &config).unwrap();
    let actual = loaded.parameters().flatten();
    assert_eq!(actual.len(), expected.len());
    for (key, reference) in expected {
        let actual = actual.get(&key).unwrap();
        assert_eq!(actual.shape(), reference.shape(), "{key}");
        assert_eq!(actual.dtype(), reference.dtype(), "{key}");
        assert_eq!(
            actual.as_slice::<f32>(),
            reference.as_slice::<f32>(),
            "{key}"
        );
    }
}
