use rurico::embed::{Embed, Embedder, ModelId, cached_artifacts};
use rurico::reranker::{Reranker, RerankerModelId, cached_artifacts as reranker_artifacts};
fn main() {
    rurico::sandbox::exit_if_seatbelt("cleanup-output-comparison");
    mlx_rs::random::seed(42).unwrap();
    let embedder = Embedder::new(&cached_artifacts(ModelId::DEFAULT).unwrap().unwrap()).unwrap();
    let query = embedder.embed_query("東京の人口").unwrap();
    let batch = embedder.embed_documents_batch(&["東京は日本の都市です。", "京都も日本の都市です。"]).unwrap();
    let batch: Vec<Vec<Vec<f32>>> = batch.iter().map(|d| d.chunks().to_vec()).collect();
    drop(embedder);
    mlx_rs::random::seed(42).unwrap();
    let reranker = Reranker::new(&reranker_artifacts(RerankerModelId::default()).unwrap().unwrap()).unwrap();
    let scores = reranker.score_batch(&[("東京の人口", "東京は日本の都市です。"), ("東京の人口", "京都も日本の都市です。")]).unwrap();
    println!("{}", serde_json::json!({"seed":42,"query":query,"batch":batch,"scores":scores}));
}
