# retrieval契約の回帰検知

[Issue #362](https://github.com/thkt/rurico/issues/362) の範囲は既存の順位・source集約・wire互換性の検証強化である。
開始版は `24725a72be44300afc24186b82b14bcb3f5f9d3d`。2026-10-08にホストでremote mainを取得し、開始版と一致した。
実装担当のDNS制限による未確認を補ったもので、開始版や合意範囲を変更していない。
Issue固定版から開始版まで、参照されたretrievalとquery_normalizeの対象sourceに差分はない。

## 方法と結果

本番関数を直接呼び、固定の小さな入力と手計算の期待hit・score・source mapを比較する。
Identity/Dedupeの降順出力は整列済みStage 2入力を前提とし、任意入力では入力順を保持する。
MaxChunk/TopKAverageのparent出力は `chunk_id = None` で、score同点のparentはdoc_idで整列する。
MaxChunkの同点chunkから選ぶsource mapは、入力で最初の最大値を持つhitのものである。
公開JSONは変更せず、literal入力と期待wire出力をserde往復とは別に確認する。

既存MLX-only構成とlocked依存で、次を実行する。

```sh
cargo test --locked --lib --features test-support,test-mlx,smoke retrieval::tests::
cargo test --locked --lib --features test-support,test-mlx,smoke storage::query_normalize::tests::
```

開始版・変更版・最終版はそれぞれretrieval 50件、query normalization 21件が成功した。
重複したTopKAverageの平均・source平均・parent化を統合し、元の数値・順位・source契約を保持した。
#298の極値・overflow・欠落sourceの別算術経路は削除していない。
各版1回の実行で、変更版retrievalの表示時間は0.01秒、開始版・最終版retrievalと各版query normalizationは0.00秒だった。
単発で表示精度も粗いため、速度改善の根拠にはしない。

次の12変異を1つずつ適用し、対応する値比較またはliteral JSON読込みの失敗を確認した。
wire変更では、過去literalに存在するdoc_id・source名・source_weightsが読めなくなる。
ほかの変異は順位・score・source map・期待hitの不一致で失敗する。
ビルド・環境の失敗を検知成功としていない。各変異を復元してから次へ進み、最終版も上記71件が成功した。

| 一時変異 | 失敗を確認したテスト |
| --- | --- |
| `config_wire_renamed` | `hybrid_search_config_literal_wire_compatibility` |
| `hit_wire_renamed` | `merged_hit_literal_wire_compatibility` |
| `max_only_source_map_wrong` | `max_chunk_keeps_max_scoring_hits_source_scores` |
| `max_sort_removed` | `max_chunk_unique_input_only_resorts` |
| `max_wrong_source_map` | `max_chunk_keeps_max_scoring_hits_source_scores` |
| `merge_sort_removed` | `weighted_rrf_fuses_chunks_separately_when_chunk_id_differs` |
| `recency_always_empty` | `weighted_rrf_recency_nan_half_life_keeps_finite_score` |
| `source_wire_renamed` | `merged_hit_literal_wire_compatibility` |
| `topk_missing_source_denominator` | `topk_average_averages_top_k_per_doc_id` |
| `topk_unselected_source_key` | `topk_average_averages_top_k_per_doc_id` |
| `topk_wrong_selection` | `topk_average_averages_top_k_per_doc_id` |
| `weights_ignored` | `weighted_rrf_fts_heavy_weight_reorders` |

再確認時は隔離コピーを使い、1変異ずつ適用して該当テスト名で上記cargo testを絞る。
整列の削除、重みを1へ固定、選ばれたsource mapの取り違え、非採用chunkのsource混入、
欠落sourceを除いた除数への変更、wire名変更、recency結果を常に空にする変更を、それぞれ検出する。
失敗理由を確認し、復元後の成功も確認する。恒常的なmutation基盤は追加していない。

#313の統合版では、上表の実行結果を履歴として保持する。
再実行時のNaN half-lifeのfilterは `weighted_rrf_recency_nan_half_life_keeps_unboosted_hits` を使う。
inf age／half-lifeの対応filterは `weighted_rrf_recency_inf_half_life_inf_age_keeps_unboosted_hits` である。
どちらも有限性だけでなく全hit・score・source mapを固定値へ照合する。
重複例の統合と現行版での検出力確認は [#313の統合記録](../issue-313/report.md#最新mainとの統合と今回のfindings)を参照する。

測定後、触ったファイルのテスト名の再述・装飾見出しを削減した。
公開APIの契約、数値例、異なる失敗条件の理由は保持した。
空行・コメントを除く全行が測定版と一致することをホストで照合したため、記録は現在版にも適用できる。
通常check・独立評価・同じheadのCIは、この対象限定検証の後に実施する。

Cargo.lock SHA-256は `743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。モデル取得・実推論は行っていない。
実モデル数値・検索品質・性能・consumer実行経路は今回確認していない。
