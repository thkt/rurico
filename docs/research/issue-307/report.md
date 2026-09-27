# Issue #307 公式数値比較とamici検索品質の調査結果

2026-09-27にApple M3で実測した。同じtokenとshapeを与えたFP32比較は、embeddingの27行、rerankerの16行すべてで測定前の診断基準を満たした。公式SentenceTransformerの公開wrapperとの比較では、末尾空白の除去によるembedding差を3入力で確認した。amiciの60文書・168クエリは既存baselineの許容範囲内だった。

これは今回の入力・モデル・環境での観測であり、全入力の同値性や性能改善を示さない。通常の推論処理、製品の許容差、保存済みfixtureは変更していない。後続の最適化では、この結果を比較の出発点として使える。

## 対象と測定条件

要求の正本は [Issue #307](https://github.com/thkt/rurico/issues/307)（更新時刻2026-09-27T09:11:44Z）。製品コードとCargo.lockの開始版は `d0639cc815d03f44dc7c80abebe40e9157deaca1`。追加した観測コードは既存の`smoke` featureと`cfg(test)`に限定した。

[ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md) は開始版と同じblob `1e2403400b586d3a09907eb12998f4b6c2f2fa81`。検索評価はamiciが所有するという決定を適用し、評価コードをruricoへ移植しなかった。ADR冒頭のbaseline一致の期待と今回の実測を区別する。#304のfixture形式も変更していない。

| 条件 | 固定した値 |
| --- | --- |
| ホスト | Apple M3、24 GiB、Mac15,3、macOS 27.0、arm64 |
| Rust / Metal | Rust・Cargo 1.98.1、Xcode 27.0 (27A266a)、Metal 32023.921 |
| rurico側 | mlx-rs 0.32.0、mlx-sys 0.6.0、MLX 0.32.2、FP32、追加Rust flagsなし |
| 公式参照 | Python 3.12.13、Torch 2.8.0、Transformers 4.56.2、Sentence Transformers 5.1.1 |
| 参照の演算条件 | CPU、FP32、eager attention、eval、4 threads、seed 0、deterministic algorithms、reference_compile=False |
| embedding | `cl-nagoya/ruri-v3-310m`、revision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` |
| reranker | `cl-nagoya/ruri-v3-reranker-310m`、revision `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3` |

固定版の[embeddingカード](https://huggingface.co/cl-nagoya/ruri-v3-310m/blob/18b60fb8c2b9df296fb4212bb7d23ef94e579cd3/README.md)・[rerankerカード](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m/blob/bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3/README.md)と実ファイルを確認した。embeddingはTransformer＋mean pooling、promptを含める設定、768次元、最大8192 token。カードが示すcosine比較に合わせ、出力を明示的にL2正規化した。rerankerはCrossEncoderのheadとsigmoidを使用した。モデルの学習時ライブラリ版と今回の参照実行版は同一ではない。

重みの全tensorがF32であること、重み・tokenizer・config・カードのHF revision/ETagと内容を照合した。開始・終了時にsource、実行ファイル、モデルの不変性を確認した。測定時の全Python依存は[保存済み環境のpython_packages](results/numerical-rechecked/environment.json)（現在のrequirementsとは区別）、内容hashは[models.json](results/numerical-rechecked/models.json)、sourceと環境は[environment.json](results/numerical-rechecked/environment.json)に記録した。再測定時の`reference.py`のSHA-256は`a4cded073a48ebd3fd63261321515d86c911f4b4b6833889fa1434dbbc13ccad`で、後述するrerankerの重複推論修正前の版である。保存済み数値はその版の証拠として保持した。その後、変更した公式reranker参照だけを現行コードで追加実測し、生出力・比較結果の完全一致を確認した（[追加実測](results/reranker-reference-refresh/complete.json)）。再測定後には報告書・手順を更新し、結果ファイルを追加した。初回の数値実測と検索実測は、それぞれのmanifestが示す版に対応する。

## 同一tokenでの数値比較

[公開合成入力](inputs.json)を使い、独立したtokenization、公式wrapper出力、ruricoのtoken列をそのまま与えた公式model出力を別々に記録した。表の行数には同じ入力の実長・bucket・混在batch・chunkが含まれ、独立した試行回数ではない。

| 比較 | 対象 | 測定前の診断基準 | 観測した最大差・最小cosine | 結果 |
| --- | --- | --- | --- | --- |
| 正規化embedding | 10入力、23 batch、27行 | cosine ≥ 0.99999 かつ最大絶対差 ≤ 1e-5 | cosine最小 0.9999999999932134、絶対差最大 7.152557373046875e-07 | 全行が基準内 |
| reranker | 6入力、12 batch、16行 | logit絶対差 ≤ 1e-4 かつscore絶対差 ≤ 2.5e-5 | logit最大 1.4901161193847656e-05、score最大 2.2649765014648438e-06 | 全行が基準内 |

正本は[embedding比較](results/numerical-rechecked/comparison-embedding.json)と[reranker比較](results/numerical-rechecked/comparison-reranker.json)。基準を測定後に変更していない。

embeddingのraw tokenは10/10入力で一致した。prefix込みのraw長は、短文8、日本語13、コード36、local境界126/130、上限境界8190/8194、query/topic/空prefixが13/12/9。最大8192へのtruncate、長短混在、paddingの影響を含む。padding条件はembedding 12単位、reranker 6単位で確認した。実長とbucket長が等しいembedding 2単位・reranker 1単位は同一観測を参照し、追加推論を行わない。異なるshapeの比較は両実装とも基準内だった。

修正後の公開文書batchは返却件数の検査を通過し、7文書・8 chunkを返した。上限を超える文書は既存plannerにより8192 tokenと2056 tokenのchunkになり、記録内の順序と個別演算との差を確認した。これはruricoのchunkを公式modelへ再生した比較であり、公式wrapperに同じchunking機能があるという意味ではない。

embeddingのhidden probeは有効tokenの先頭・中央・末尾の3位置だけを観測した。最大絶対差は0.00095367431640625で、正規化後のembedding用基準は適用しない。全位置・全layerのhidden stateや、poolingだけの誤差を証明するものではない。

rerankerのraw/model tokenは6/6入力で一致し、8198→8192 tokenのtruncateを含む。順位は2クエリに対する6条件（実長2、bucket 2、混在1、公開API 1）すべてで一致した。公式logit差が2e-4を超える候補対に逆転はない。近接候補には同一文の`library`と`library-copy`があり、同点と返却順序を保持した。異なる文が僅差になる一般的な順位安定性までは確認していない。

## 公開wrapperとの相違は末尾空白

次の3入力は同一tokenのmodel比較を通過する一方、通常のSentenceTransformer経由では基準外になった。

| 入力 | rurico / 公式wrapperのtoken数 | cosine | 最大絶対差 |
| --- | ---: | ---: | ---: |
| local-before | 126 / 125 | 0.9962851332783761 | 0.010189130902290344 |
| local-after | 130 / 129 | 0.9968912973800986 | 0.009299302473664284 |
| limit-before | 8190 / 8189 | 0.9999817322436276 | 0.0007517458871006966 |

3件ともraw token列は一致し、wrapper後だけ末尾空白に当たるtoken 271がEOSの直前から1個消える。残るtoken列は一致する。ruricoの公開APIと直接演算との差は0だった。

Sentence Transformers 5.1.1の[Transformer.tokenize](https://github.com/huggingface/sentence-transformers/blob/v5.1.1/sentence_transformers/models/Transformer.py#L312)は入力文字列全体に`str(s).strip()`を適用する。固定tagのソースとインストール済みファイルは改行を正規化して一致した（SHA-256 `03c1ebeb82fe2b9a64a7f70ea71c8c780883498cdb1a2179c7486097d952afb2`）。raw列の一致、除去位置、同一tokenでの一致を合わせ、この3件の差をwrapperの前処理差と判断した。`limit-after`では除去される末尾がtruncate範囲外になるため、両者の8192 token列は一致した。

今回はruricoにstripを追加しない。prefix付加後の文字列全体に対するstripと、利用者が本文だけに行うtrimは常に同じではない。変更すれば保存済みembeddingや再indexに影響し得るため、[Issue #315](https://github.com/thkt/rurico/issues/315)の前処理契約で対象と互換性を決める。

## amiciでの検索品質

[amici固定版](https://github.com/thkt/amici/tree/547f9ee2ed734a2eab316fdbd62f194849875ee4)の公開archiveを新規取得し、通常/devのrurico依存2箇所を開始版d063へ変え、必要なlock解決だけを行った。[manifest差分](results/search/amici-dependency.patch)・[lock差分](results/search/amici-lock.patch)・[解決済みpackage](results/search/resolved-packages.json)を保存した。評価コード・公開fixture・既存baselineは変更していない。

既存eval_harnessをdev buildし、`capture-baseline`で別ファイルへ測定結果を保存した。続く`verify-baseline`は終了0だった。fixtureは60文書・168クエリ、8分類各21クエリ。identity aggregation、RRF/source weights、normalizationは既存baselineと同一。95% CIは既存のbootstrap（1000 resamples、seed 42）を使った。

| 指標 | 今回の値 | 95% CI | 既存baselineとの差 |
| --- | ---: | --- | ---: |
| Hit@1 | 1.000000 | [1.000000, 1.000000] | 0 |
| Hit@3 | 1.000000 | [1.000000, 1.000000] | 0 |
| Recall@5 | 0.629960 | [0.598512, 0.663591] | 0 |
| Recall@10 | 0.741468 | [0.711210, 0.774405] | 0 |
| MRR@10 | 1.000000 | [1.000000, 1.000000] | 0 |
| nDCG@10 | 0.892977 | [0.876098, 0.906292] | +0.000261945249 |

全桁と分類別結果は[今回のsnapshot](results/search/amici-measured.json)、既存値・差・実行結果は[search-comparison.json](results/search/search-comparison.json)。既存許容差はRecall@5が0.01、他が0.001で、nDCGの差もその範囲内。旧版を同じホストで新たに測定した因果比較ではなく、改善とは判断しない。

再実行手順を確定する前にも、同じamici公開archiveと未変更のrurico d063 archiveを一時Cargo path patchで組み合わせて測定した。[最初の観測](results/search-manual/quality-comparison.json)は今回のglobal各指標・CIと一致し、既存baseline検証・identity・single-docの既存テストも成功した。二つの依存解決方法と実行記録は区別して保存し、標本を合算していない。

amici snapshotの`mlx_rs_version=0.25`はharness内の定数で、実際の解決版を表さない。raw結果を改変せず、実際のmlx-rs 0.32.0等はlock/package一覧で確認した。model revisionの汎用ラベルも実際のモデルhashと区別する。この結果はamiciのreference compositionに限る。recall等の利用側アプリ全体の品質、実データでの品質、速度改善は未検証。

## 再現方法と成果物

[README](README.md)に新規実行と比較器の再集計手順を示す。`results/numerical-rechecked`が文書件数・shape観測共有の修正後、rerankerの重複推論修正前の数値実測、`results/numerical`が初回の数値実測、`results/search`が検索実測であり、それぞれ成功した一回の実行で、`complete.json`に元ファイルのSHA-256を記録した。数値実測ごとの大きな4個の生JSONをgzip圧縮して保存し、[packaging.json](results/packaging.json)に圧縮前後のhashを対応付けた。展開すると元のJSONをbyte単位で復元できる。比較器はモデルを再実行せず、その生出力から同じ比較JSONを生成できる。

ログ、モデル重み、実行binary、venv、archive、個人pathは公開物に含めない。準備時の初回実行は`reference_compile`の渡し先誤りで公式モデル読込みに失敗し、失敗記録を別ディレクトリに保持した。設定の渡し先と必要なSentencePiece/protobuf依存・tokenizer.model取得を修正して、新しい実行先で全工程を完了した。部分的な出力を成功結果として再利用していない。

準備時の比較器6テストは、cosineだけで正規化差を見逃す、sigmoid飽和でlogit差を隠す、近接候補や公開APIの順位を落とす、非有限値・欠損を成功扱いする、異なる検索条件を比較する誤報を防ぐ。cosineのみ・logit条件無視・公開順位無視の3改変で対応テストが失敗することも準備時に確認した。既存のplanner・pooling・重みロード・fixture検証は保持している。標準check・独立評価・同じheadのCIの結果はPRの検証記録に置き、このモデル実測と区別する。

## 記録処理の修正と再測定

初回の数値実測は[初回比較](results/numerical/comparison-embedding.json)と元の生出力・manifestに保持した。embedding 25 batch・29行、reranker 13 batch・17行で同一tokenの診断基準を満たしたが、公開文書batchの記録ではzipが余剰返却を捨てる可能性があった。その記録だけでは返却総数の上限を証明できない。これは記録処理の欠落であり、実際の製品出力に余剰が見つかったわけではない。

文書数の過剰・不足をzip前に拒否するよう修正し、同じrowの実長とbucket長が等しい場合は一度の観測から両条件を評価するようにした。異なるshape、別row、混在batch、公開API、実装間の観測は保持する。共有先はschema 2の`conditions`へ記録し、比較器は旧schema 1も読める。反復判定のなかった同じ条件の追加推論を除いたため、反復時だけの不安定さを観測する機会は減る。反復安定性や所要時間・メモリの削減量は今回の保証に含めない。

モデルmanifest作成では、HF ETagとhashを記録済みのJSONについて、再計算したhashを捨てる処理を除去した。追加wrapper JSONのhash、最初のrevision/ETag照合、推論後の内容hash再検証は維持した。重複hashを検出するテストは修正前に失敗し、修正後に成功した。

修正後は固定Python 3.12.13環境の8テストと、記録境界・batch選択のRust 2テストを実行して成功した。共有条件の検査を外す改変でも追加テストが失敗する。これらはモデルを使わない検証であり、別途Metalと公式CPU参照を新しい出力先で再実行した。再測定では文書返却の過剰・不足がないことも確認でき、上の実測表はこの修正後の結果を示す。旧実測の書換えや投影を新しい実測として扱っていない。

検索評価の本体・入力・rurico d063は修正していないため、amiciの値は既に完了した検索実測を参照する。hash計算の修正はPythonテストと数値比較入口の再実行で確認した。標準check・文書を含む独立評価・同じheadのCIはPRの検証記録で別途確認する。

## 公式reranker参照の追加実測

初回と再測定では、各入力にIdentityとSigmoidを指定した`predict`を1回ずつ呼んでいた。[固定版CrossEncoder.predict](https://github.com/huggingface/sentence-transformers/blob/v5.1.1/sentence_transformers/cross_encoder/CrossEncoder.py)はactivation前に毎回forwardを行う。そこで、1回の公式predict内で生logitを保存してTorch sigmoidを返す`RecordingSigmoid`へ統合した。公式wrapperのtokenization・truncation、別shapeの比較、両数値の診断は保持する。反復安定性の判定は元からなく、削除した推論にだけ生じる一過性失敗の観測機会は減る。時間・メモリ効果は未測定。

最初の局所修正は関数をactivationへ渡していたため、実モデルが登録済みのSigmoid子moduleを持つ場合にTypeErrorで失敗した。軽量テストにも同じ初期状態を持たせると、この失敗を再現できた。activationを`torch.nn.Module`として実装し直し、固定Python環境の全10テストが成功した。追加2テストは公式CrossEncoder.predictとTorchを使い、モデル・tokenizerを模擬する。異なる符号とsigmoid飽和を含む3入力についてforward回数、logit・score・IDの対応、tokenization引数、forwardとactivationの例外伝播を確認する。実重みや長文の検証の代用にはしない。

その後、修正版の`reference.capture("reranker", ...)`を実際の固定モデルと6入力で再実行した。Metal側は`numerical-rechecked`で取得済みの生出力を使い、Rust/Cargoのsourceが同じことと生出力hashを確認した。変更のないembeddingと検索pipelineは再実行していない。今回の実行範囲とruntime source・環境は[environment.json](results/reranker-reference-refresh/environment.json)へ記録した。

新しい公式参照JSONと比較JSONは、保存済みの再測定結果に完全一致した。native 6入力と同一tokenの12 batch・16行を含み、診断基準・順位の結果も変わらない。[完了記録](results/reranker-reference-refresh/complete.json)・[比較結果](results/reranker-reference-refresh/comparison-reranker.json)・新しい参照生出力を保存した。runtime sourceとモデル内容は前後で再検証した。この追加実測は公式参照だけの再実行であり、Metalの再実行とは扱わない。初回の型エラーの記録も別の実行先に保持した。

全体を再実行する場合は[README](README.md#ホストで実行する)の数値比較コマンドを使う。保存出力の再集計では、Rust出力を`numerical-rechecked/rurico-reranker.json.gz`、公式参照を`reranker-reference-refresh/reference-reranker.json.gz`から展開して比較できる。製品側の推論と、既に確認した検索構成は変更していない。

## 後続で判断すること

[Issue #308](https://github.com/thkt/rurico/issues/308)・[Issue #309](https://github.com/thkt/rurico/issues/309)では入力、前処理、モデルrevision、参照条件を固定し、変更前後の数値と実機性能を比較する。今回の観測だけで精度やarchitecture変更を採用しない。wrapperに合わせる前処理変更は#315で互換性を判断し、演算最適化と同時に混ぜない。

追加のCPU backend、恒常的な品質CI、製品の新しい許容差、amiciへの修正は本調査で導入していない。
