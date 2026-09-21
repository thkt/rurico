//! Check the checkpoint contract before allocating MLX parameters.
//!
//! Only metadata is read: the tensor payload remains on disk until MLX loads
//! it. Names are exact (no prefix stripping, aliases or tied-weight fallback).

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[cfg(feature = "mlx")]
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::{ModuleParameters, ModuleParametersExt},
};
use serde::Deserialize;
use serde::de::{self, MapAccess, Visitor};
use serde_json::Value;

use super::Config;

#[derive(Clone, Copy, Debug)]
pub(crate) enum WeightKind {
    Embed,
    Reranker,
}

pub(crate) fn expected_shapes(config: &Config, kind: WeightKind) -> BTreeMap<String, Vec<usize>> {
    let prefix = match kind {
        WeightKind::Embed => "",
        WeightKind::Reranker => "model.",
    };
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let mut shapes = BTreeMap::new();
    shapes.insert(
        format!("{prefix}embeddings.tok_embeddings.weight"),
        vec![config.vocab_size, h],
    );
    shapes.insert(format!("{prefix}embeddings.norm.weight"), vec![h]);
    shapes.insert(format!("{prefix}final_norm.weight"), vec![h]);
    for i in 0..config.num_hidden_layers {
        let layer = format!("{prefix}layers.{i}");
        // The first encoder layer uses Identity, not an unloaded LayerNorm.
        if i != 0 {
            shapes.insert(format!("{layer}.attn_norm.weight"), vec![h]);
        }
        shapes.insert(format!("{layer}.mlp_norm.weight"), vec![h]);
        shapes.insert(format!("{layer}.attn.Wqkv.weight"), vec![3 * h, h]);
        shapes.insert(format!("{layer}.attn.Wo.weight"), vec![h, h]);
        shapes.insert(format!("{layer}.mlp.Wi.weight"), vec![2 * inter, h]);
        shapes.insert(format!("{layer}.mlp.Wo.weight"), vec![h, inter]);
    }
    if matches!(kind, WeightKind::Reranker) {
        shapes.insert("head.dense.weight".into(), vec![h, h]);
        shapes.insert("head.norm.weight".into(), vec![h]);
        shapes.insert("classifier.weight".into(), vec![1, h]);
        shapes.insert("classifier.bias".into(), vec![1]);
    }
    shapes
}

/// Diagnostic categories follow `weights:`; details include the key.
pub(crate) fn validate(path: &Path, config: &Config, kind: WeightKind) -> Result<(), String> {
    config
        .validate()
        .map_err(|e| format!("invalid config: {e}"))?;
    let (header, payload_len) = read_header(path).map_err(|e| format!("weights: header: {e}"))?;
    validate_header(&header, payload_len, config, kind)
}

/// Preflight on CPU, then load the payload once and require exact assignment.
/// Checking the actual MLX map also catches naming drift in our module tree,
/// and a checkpoint replaced between preflight and the backend's open.
#[cfg(feature = "mlx")]
pub(crate) fn load<M: ModuleParameters>(
    path: &Path,
    config: &Config,
    kind: WeightKind,
    build: impl FnOnce(&Config) -> Result<M, Exception>,
) -> Result<M, Exception> {
    validate(path, config, kind).map_err(Exception::custom)?;
    let mut model = build(config)?;
    let mut loaded = Array::load_safetensors(path)
        .map_err(|e| Exception::custom(format!("SafeTensors load error: {e}")))?;
    for (key, param) in model.parameters_mut().flatten() {
        let value = loaded
            .remove(key.as_ref())
            .ok_or_else(|| Exception::custom(format!("weights: missing key: {key}")))?;
        if value.shape() != param.shape() {
            return Err(Exception::custom(format!(
                "weights: shape: {key}: expected {:?}, got {:?}",
                param.shape(),
                value.shape()
            )));
        }
        if value.dtype() != Dtype::Float32 {
            return Err(Exception::custom(format!(
                "weights: dtype: {key}: expected F32, got {:?}",
                value.dtype()
            )));
        }
        *param = value;
    }
    if let Some(key) = loaded.keys().next() {
        return Err(Exception::custom(format!("weights: unknown key: {key}")));
    }
    model.eval()?;
    Ok(model)
}

// Unlike serde_json's normal map deserializer, reject duplicate tensor names
// rather than letting different readers select different definitions.
struct Header(BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header object with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Header, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, Value>()? {
                    if entries.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate key {key}")));
                    }
                }
                Ok(Header(entries))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn read_header(path: &Path) -> Result<(Header, u64), String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let file_len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut len_bytes = [0; 8];
    file.read_exact(&mut len_bytes).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_bytes);
    // Same upper bound as artifact kind inspection; never allocate payload size.
    if header_len > 100 * 1024 * 1024 {
        return Err("header exceeds 100 MiB limit".into());
    }
    let payload_len = file_len
        .checked_sub(8 + header_len)
        .ok_or("truncated header")?;
    let mut bytes = vec![0; usize::try_from(header_len).expect("bounded header")];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let header = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok((header, payload_len))
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn validate_header(
    header: &Header,
    payload_len: u64,
    config: &Config,
    kind: WeightKind,
) -> Result<(), String> {
    let mut expected = expected_shapes(config, kind);
    let mut ranges = Vec::new();
    for (key, value) in &header.0 {
        if key == "__metadata__" {
            // MLX writes null when no metadata is supplied. Safetensors uses
            // Option<HashMap<String, String>> for this field as well.
            Option::<BTreeMap<String, String>>::deserialize(value)
                .map_err(|e| format!("weights: header: invalid metadata: {e}"))?;
            continue;
        }
        let shape = expected
            .remove(key)
            .ok_or_else(|| format!("weights: unknown key: {key}"))?;
        let tensor =
            TensorInfo::deserialize(value).map_err(|e| format!("weights: header: {key}: {e}"))?;
        if tensor.shape != shape {
            return Err(format!(
                "weights: shape: {key}: expected {shape:?}, got {:?}",
                tensor.shape
            ));
        }
        if tensor.dtype != "F32" {
            return Err(format!(
                "weights: dtype: {key}: expected F32, got {}",
                tensor.dtype
            ));
        }
        let bytes = shape
            .iter()
            .try_fold(4_u64, |n, &dim| n.checked_mul(dim as u64))
            .ok_or_else(|| format!("weights: offsets: {key}: tensor size overflow"))?;
        let [start, end] = tensor.data_offsets;
        if end.checked_sub(start) != Some(bytes) || end > payload_len {
            return Err(format!(
                "weights: offsets: {key}: invalid range {start}..{end} for {bytes} bytes (payload {payload_len})"
            ));
        }
        ranges.push((start, end, key));
    }
    if let Some(key) = expected.keys().next() {
        return Err(format!("weights: missing key: {key}"));
    }
    ranges.sort_unstable();
    let mut position = 0;
    for (start, end, key) in ranges {
        if start != position {
            return Err(format!(
                "weights: offsets: {key}: gap or overlap at {position}"
            ));
        }
        position = end;
    }
    if position != payload_len {
        return Err("weights: offsets: trailing payload bytes".into());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
