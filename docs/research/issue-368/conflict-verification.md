# PR #375の競合解消と検証

main `01dd1c69c633a7f2eeccd188d566ec25d0f94e93` と、公開head
`2f3de3907055675b276b45ae3e73db3a28e87d2a` を統合した。
CONTRIBUTINGで同じ位置へ追加されたvector検証と文書分割検証の両方を保持した。
mainの追加検証・文書・filelock 4.0.8を保持し、Cargo.lockはmainと一致する。
storageの処理・入力・期待値は変えず、操作を言い換えるコメント2箇所を削除した。

## 実SQLでの回帰検出

統合後の未commit変更を含む一時コピーで、既存storageテスト3件を実行した。
正常版と復元版は3件とも成功した。次の変異はそれぞれ終了コード101となり、
対応する値比較のassertionが失敗した。

- 挿入側の各f32の4 bytesを逆転し、保存されたlittle-endian bytesの比較で検出した。
- 取得した近傍を逆順にして、ID・順序・距離の比較で検出した。
- 元のKNN SQLをmaterialized CTEに入れ、外側のSELECTでrowid順へ並べ直し、
  近傍取得を成功させたうえで上位3件の順序比較で検出した。

vec0のKNN queryへ直接`ORDER BY rowid`を指定した試行は、SQLiteが拒否した。
その失敗は順位assertionの検出実績へ数えず、上記の外側整列で確認し直した。
初回runnerはassertionのpanicをstdoutだけで探したため確認に失敗した。
stderrも確認するrunnerへ直し、旧試行の生ログは保全した。
全変異を復元し、今回のcheckoutを変異させていない。

[版・hash・各試行の結果](conflict-verification.json)にコマンドとログhashを保存した。
sourceとCargo.lock、Rust 1.99.0、Xcode 27、macOS 27のホスト記録も保持している。
時間にはコンパイルが含まれ、速度改善・反復安定性は判断していない。
実モデルの意味表現・検索品質、consumer pipelineとmigrationは未確認である。
標準checkと新しい独立評価、更新headのCIはこの後に行い、旧成功を流用しない。
