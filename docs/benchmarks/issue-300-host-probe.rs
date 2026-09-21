use std::time::Instant;
use rurico::embed::{Embed, Embedder, ModelId, cached_artifacts};
use rurico::reranker::{Reranker, RerankerModelId, cached_artifacts as reranker_artifacts};

fn memory() -> serde_json::Value {
    serde_json::json!({
        "active": mlx_rs::memory::active_memory().unwrap(),
        "cache": mlx_rs::memory::cache_memory().unwrap(),
        "peak": mlx_rs::memory::peak_memory().unwrap(),
    })
}

fn main() {
    rurico::sandbox::exit_if_seatbelt("issue-300-load-comparison");
    let args: Vec<String> = std::env::args().collect();
    let kind = &args[1];
    let seed: u64 = args[2].parse().unwrap();
    mlx_rs::random::seed(seed).unwrap();
    let query = "東京の人口";
    let docs = ["東京は日本の都市です。", "京都も日本の都市です。", "東京都の人口統計を調べます。", "猫は窓辺で眠っています。"];
    let result = if kind == "embed" {
        let artifacts = cached_artifacts(ModelId::DEFAULT).unwrap().unwrap();
        mlx_rs::memory::reset_peak_memory().unwrap();
        let start = Instant::now();
        let model = Embedder::new(&artifacts).unwrap();
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let loaded_memory = memory();
        let query_output = model.embed_query(query).unwrap();
        let batch = model.embed_documents_batch(&docs).unwrap();
        let batch: Vec<Vec<Vec<f32>>> = batch.iter().map(|d| d.chunks().to_vec()).collect();
        serde_json::json!({"kind":kind,"seed":seed,"load_ms":load_ms,"loaded_memory":loaded_memory,"query":query_output,"batch":batch})
    } else {
        assert_eq!(kind, "reranker");
        let artifacts = reranker_artifacts(RerankerModelId::default()).unwrap().unwrap();
        mlx_rs::memory::reset_peak_memory().unwrap();
        let start = Instant::now();
        let model = Reranker::new(&artifacts).unwrap();
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let loaded_memory = memory();
        let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (query,*d)).collect();
        let scores = model.score_batch(&pairs).unwrap();
        let repeat_scores = model.score_batch(&pairs).unwrap();
        let mut ranking: Vec<usize> = (0..scores.len()).collect();
        ranking.sort_by(|&a,&b| scores[b].total_cmp(&scores[a]));
        serde_json::json!({"kind":kind,"seed":seed,"load_ms":load_ms,"loaded_memory":loaded_memory,"scores":scores,"repeat_scores":repeat_scores,"ranking":ranking})
    };
    println!("{result}");
}
