# Issue #316: query枠と長文documentの文脈比較

[Issue #316](https://github.com/thkt/rurico/issues/316)と合意記録を要求の正本とする。
[報告](report.md)は観測・推奨案・未確認事項の入口。
[局所plan観測](plan-results.json)は固定tokenizerでの文脈・境界確認であり、モデル品質・性能の測定ではない。
製品のtruncation、公開API、consumer、padding、演算は変更しない。
研究用binary `context_spike` は既存の `smoke` featureでだけビルドする。
通常の `cargo fetch --locked` / `bash scripts/check.sh` とcaptureなしの契約を維持する。
checkはこの比較のモデル測定を実行しない。

## 比較する方式

全方式・全queryで、独立テスト担当が用意した
[同じ12候補](../../../tests/fixtures/long_context/generate.py)を入力順どおり使う。
日本語/コードの長短4query、根拠の先頭/中央/末尾、短い正常対照、誤答・無関係を含む。
relevanceは全文の回答内容に基づくラベルであり、期待するモデル順位ではない。
文字数からtoken overflowを推定しない。固定tokenizerで実token数を記録する。

| 方式 | queryとdocumentの選択 | 集約 |
| --- | --- | --- |
| baseline | 現行のpair encode後に製品の`truncate_pair` | 単一score |
| query-prefix | query末尾を最大1024 token保持し、残り枠にdocument先頭 | 単一score |
| chunk-neighbors | 同じquery枠。512 token chunkからqueryとのtoken ID一致数最大のchunkと前後各1 chunk | 単一score |
| windows | 同じquery枠。残り枠を上限にdocument全体を128 token overlapで掃引 | window sigmoid scoreのmaxと算術meanを別々に評価 |

1024/512/128は今回の比較候補のパラメータであり、採用済みの品質条件や資源上限ではない。
長query末尾に質問を置くfixtureの条件を明示してquery末尾保持を試す。
query先頭に重要条件がある実利用へ一般化しない。
chunkの選択はモデルscoreやrelevanceラベルを使わない軽量な代理候補。
一致数同点は先頭chunk、文書score同点は元の候補indexが小さい順。
maxは一つの強いwindowを拾い、meanは全windowの平均的な関連性を測る。
overlap内の情報は複数回寄与し、meanはwindow数と無関係な文脈に影響される。
同じforward出力から両集約を計算し、集約ごとに再推論しない。

候補pairは固定tokenizerのpostprocessorで特殊tokenと境界を付け直す。
query/documentを個別encodeするため、pair encodeとの一致はホストのtoken記録で照合する。
baselineの境界消失は現行動作の観測として記録し、候補で同じ損失を強制しない。
全方式とも既存bucket `[128,512,2048,8192]`、singleton forward、既存の有限性検査とcleanupを使う。
製品の複数文書batchとの性能差は、この比較から判断しない。

## ホストで実行する

Apple Silicon、対応するXcode/Metal、固定F32 reranker snapshot
`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3` を使う。
モデル不足時は[既存のprepare-cache手順](../../../CONTRIBUTING.md#mlx_smoke-smoke-テスト)を使う。
runner/binaryはdownload、ブラウザー、サーバーを起動しない。
親 [#296](https://github.com/thkt/rurico/issues/296) の本文・updatedAt・hashを取得し、
実行・保存条件を確認してから測定する。実装セッションの取得失敗後、ホストで取得・照合した版は[報告](report.md)に示す。

同じcheckoutと未commit差分をreleaseでビルドし、実際のbinaryへ紐づける。

```sh
CARGO_TARGET_DIR=/tmp/rurico-316-host-build cargo build --locked --release --features smoke --bin context_spike --message-format=json > /tmp/rurico-316-build.jsonl
```

`compiler-artifact`の `target.name=context_spike` のexecutableを使う。
開始commitは `1d81211bf7571598f0a5c98663703c21f13d056b`。
測定前にHEAD、差分、Cargo.lock、binary、Rust/Cargo/Xcode/Metal、機種・RAM・OS、
電源・熱・背景負荷、モデル3ファイルのhashとF32 headerを記録する。
次のcontextの値は書式例であり、実際の環境観測とbuild argvへ置き換える。

```json
{
  "machine": "実機の機種とchip",
  "ram_bytes": 25769803776,
  "os": "実際のOS版",
  "rust": "rustc --versionの実出力",
  "cargo": "cargo --versionの実出力",
  "xcode": "xcodebuild -versionの実出力",
  "metal": "xcrun metal --versionの実出力",
  "build": "上記ビルドargvとprofile、build.jsonlとbinaryへの対応",
  "load_conditions": "電源、熱、背景負荷、他GPU測定の有無",
  "reranker_snapshot": "/absolute/HF/snapshots/bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3",
  "parent_issue_evidence": "親Issueの取得版・hash・実行と保存条件を記録した証拠への参照"
}
```

```sh
python3 docs/research/issue-316/run.py /absolute/path/to/context_spike /tmp/rurico-316-context.json /tmp/rurico-316-plan-NEW --mode plan --seconds 1800
python3 docs/research/issue-316/run.py /absolute/path/to/context_spike /tmp/rurico-316-context.json /tmp/rurico-316-model-NEW --mode model --seconds 1800
```

1800秒は例示した実験予算であり、製品のhard limitではない。
実際の実行枠に収まる値を指定し、合意済み上限を変えない。
同じ固定環境を親runner、fixture生成の子Python、binary、buildへ渡す。
Pythonの実体と版はrunnerが記録する。Rust等のcontext値は担当AIが生ログへ照合する。
actor runtimeの版やbinary絶対パスだけで子ツールの版を証明しない。
`plan`は同じtoken選択経路をモデルforwardなしで通す。品質・速度・peakの実測へ読み替えない。

生ログはcheckoutと旧runの両方の外の新規ディレクトリへ保存する。
runnerは入力、source差分と試作source、source/lock/binary/model/contextの前後hash、
strategy別JSONL、stdout/stderr/status、OS観測、summaryと完了manifestを保存する。
失敗・予算超過では部分証拠を保持して停止する。`complete`は収集完了であり受入ではない。
既存#318のprocess group回収とRSS/pressure/swap観測を再利用し、同Issueの停止閾値は転用しない。
強制終了やホスト停止で最終記録・回収が完了する保証はない。

## 結果を照合する

JSONLの実token列、sequence ID、byte offset、特殊token mask、選択document token範囲から、
query/document境界、EOS、8192上限、根拠の残存を実際の文字列と照合する。
fixtureの根拠spanはUnicode文字位置であり、binaryはbyte位置へ変換する。
queryのどの条件を失ったかも確認し、`evidence_retained`だけを品質保証にしない。
baselineの末尾置換位置は元Encodingとの差分から判別する。
候補でquery/documentのsequence、BOS/EOS、特殊token数、観測列長を失う出力はrunnerも拒否する。
短文の正常対照では元のpair IDs/maskとの一致も照合する。

summaryのRecall/nDCG@1/5/10、全候補順位（位置別根拠を含む）、forward数、wallとpeakを
JSONL・monitorの原記録へ照合する。synthetic指標はrerankだけの比較であり、検索pipelineのRecallではない。
全queryで同候補集合、同順序、同入力hashが維持されていることを確認する。
modelはstrategy別processでロードし、queryごとに初回1回とwarm3回を記録する。
後のqueryは同processの既存cacheを共有するため、`cold`はそのqueryの初回呼出しを表す。
方式の順序は固定であり、OS cacheや熱・順序効果を除いた因果的な速度差は保証しない。
wallはtoken化・選択・推論・観測構築を含み、JSONL書込みとloadを含まない。
Metal peakは呼出し前にresetするがweight/cache込み、RSSはprocess全体の約0.1秒間隔観測最大。
両者をGPU全量や瞬間peakへ読み替えない。

#307の[公式比較・amici品質](../issue-307/report.md)は対象版・fixtureを固定した完了済みの根拠として使う。
amici `547f9ee2ed734a2eab316fdbd62f194849875ee4` のfixture・baselineは変更しない。
候補revの継続評価には[既存のmodel lane](../issue-307/validation-366.md)を使い、synthetic入力で置き換えない。
未commit試作の実行と公開commitを対象にするmodel laneは別の証拠であり、相互に成功を流用しない。

測定後、担当AIが原記録と公開可能な要約を照合して[報告](report.md)へ実測を追記する。
不具合があれば修正・再測定し、変更文書を含む同じ差分の独立評価へ戻す。
方式・品質と応答時間の許容範囲・用途は結果を見てユーザーが判断する。

## 方式を分けて収集する

全方式が一つの測定期限に収まらない場合、元の未完了記録を保存し、方式単位で新しい保存先へ収集する。
今回のwindow別測定は次の外部driverと同じ処理だった。対象tuple以外のcollectorを変えず、
同じ固定fixtureの全query・全trialを実行する。SIGTERMでは既存のfinallyによる子group回収へ戻す。
所有process・メモリ・期限の外側監視も維持し、測定中に他のビルド/GPU作業を重ねない。

```python
import importlib.util
import signal
import sys
from pathlib import Path

collector = Path(sys.argv.pop(1)).resolve(strict=True)
spec = importlib.util.spec_from_file_location("context316_collector", collector)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.STRATEGIES = ("windows",)

def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}; incomplete evidence retained")

signal.signal(signal.SIGTERM, interrupted)
module.main()
```

上記をcheckout外のdriverへ保存し、引数はcollectorの後に既存のbinary/context/新出力/`--mode model --seconds 1800`を渡す。
各保存先の入力・source/lock/model/binary/contextのhash、全trialのraw token列、完了statusとgroup不在を照合して統合する。
元の失敗した全体runを成功に変更せず、途中のwindow出力は別測定の全16回へ混ぜない。
[model-results.json](model-results.json)の完了範囲とhash、[報告](report.md)の測定方法を参照する。
