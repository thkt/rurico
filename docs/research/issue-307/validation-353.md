# Issue #353 Python参照環境の更新記録

2026-09-28 JST、47依存の固定一覧、wrapper/API対応、Python CIとホスト実行手順を準備した。
レビューR1-1で不足していた実測sourceとの対応を調査し、固定実モデルの両比較を再集計した。
測定版と現行版の推論実装・入力の同一性を確認できたため、下記の範囲で実測を再利用する。
R1-2の否定テストも修正し、対象検査を無効化すると失敗することを確認した。
[ホストの追加証拠](results/numerical-353/host-python-validation.json)により、新規環境への通常installと、
R1-2修正後の依存整合・厳密な版検査・全Pythonテスト15件の成功を確認した。
review-2は修正後sourceの標準check成功と実測の再利用条件を確認し、本文への証拠反映漏れをR2-1として返した。
今回はその記述を更新する。変更後の標準check・最終独立評価・同じheadのCIはホストで更新するため、
この記録はIssue全体の受入完了を示さない。

## 合意と参照版

要求は[Issue #353](https://github.com/thkt/rurico/issues/353)（更新時刻`2026-09-27T17:36:34Z`）。
開始commitと作業headはいずれも`99510691a564182d4af13485988a686d5f0a6893`で、変更は未commit。
#349の旧版固定・自動更新停止は今回の合意で見直し、過去の記録は保持する。
参照した原本は開始commitの次の版と一致し、過去観測へ新環境の成功を追記していない。

| 参照 | 開始版のGit blob | 適用範囲 |
| --- | --- | --- |
| [旧README](https://github.com/thkt/rurico/blob/99510691a564182d4af13485988a686d5f0a6893/docs/research/issue-307/README.md) | `451589fcac20bc8f86666c31f4c1f6905a6cb888` | 旧運用から現行運用へ更新する対象。旧Torch固定とrequirementsの不一致は今回解消 |
| [report.md](report.md) | `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 旧数値観測の条件・制約。新環境の数値結果には使わない |
| [旧参照環境](results/reranker-reference-refresh/environment.json) | `6c314ca41cce646b12ca1f554a8307bcfa10fc6f` | Torch 2.8.0 / Transformers 4.56.2 / ST 5.1.1の当時のsource・環境 |
| [ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md) | `1e2403400b586d3a09907eb12998f4b6c2f2fa81` | accepted。検索評価の所有先はamici。baseline一致の期待を今回の実測へ読み替えない |

製品Rust、入力、比較器の許容差、モデル/tokenizer revision、amiciの検索pipelineは変更していない。
#307の旧依存環境の出力は再利用していない。下記は#353の新依存環境でホストが
`run.py numerical`の全比較を実行した観測である。
同入口の`start_commit=d0639cc...`は#307の製品基点で、今回の開始commitではない。
実行対象の識別には`head`と全source hashを併用する。

## 依存とAPIの照合

実装時に全47名のPyPI projectページのRelease filesを読み直した。
[requirements](requirements.txt)の正規化名に対応する`https://pypi.org/project/<name>/`を出典とする。
45依存は採用版と公開最新版が一致し、残る2依存は下表の互換性制約で選んだ。
シェルのPyPI取得はDNSエラーとなったため、公開版の再照合はWeb取得、Requires-DistとAPIは
Issueの候補導入に使ったPython 3.12.13 / macOS arm64環境の配布ソースを読み取った。
既存環境へinstall・upgradeしていない。pip 26.2.1は導入ツールとして別記する。

| 依存 | 今回の採用版 | 確認した制約・公開版 |
| --- | --- | --- |
| [torch](https://pypi.org/project/torch/) | 2.14.0 | macOS arm64 / CPython 3.12 wheelの公開を確認 |
| [transformers](https://pypi.org/project/transformers/) | 5.17.0 | hub `>=1.5.0,<2.0`、tokenizers `>=0.23.1,<0.24.0` |
| [sentence-transformers](https://pypi.org/project/sentence-transformers/) | 6.1.0 | Transformers `>=5.0.0,<6.0.0`、hub `>=1.3.0,<2.0.0` |
| [tokenizers](https://pypi.org/project/tokenizers/) | 0.23.2 | 上記の範囲内 |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/) | 1.33.0 | 最新2.0.0は上記2依存が非対応。いずれかの更新時に両方を再確認 |
| [sympy](https://pypi.org/project/sympy/) | 1.14.0 | mpmath `>=1.1.0,<1.4` |
| [mpmath](https://pypi.org/project/mpmath/) | 1.3.0 | 最新1.4.1はSymPyの範囲外。SymPy更新時に再確認 |
| [protobuf](https://pypi.org/project/protobuf/) | 7.36.2 | SentencePiece 0.2.2とともに全導入版検査へ含める |

開始版requirementsの35依存はすべて保持し、12の推移依存を追加した。
開始版とのPEP 440比較でダウングレード0を確認し、主要4依存の版もIssueの目標と一致した。
解決成功だけを根拠に主要依存を旧版へ戻していない。requirementsを唯一の版固定とし、
全runtime依存の版と集合を検査する。正常な依存解決で増えた依存も一覧へ追記しなければ失敗する。

ST 6.1.0の配布ソースでは、CrossEncoderはmodule列を使う実装へ変わり、
`model`はその列からモデルを返すpropertyになった。旧テストの直接代入は`None`を返して失敗した。
また、poolingの複数boolean属性は`pooling_mode`へ変わり、`tokenize`は`preprocess`の非推奨aliasになった。
参照コードを現APIへ合わせ、rerankerの記録も実際のwrapper前処理を通すようにした。
生tokenizationは別に保持し、wrapperの前処理差を製品側へ適用していない。

Transformers 5.17.0のModernBERTには`reference_compile`属性と専用compiled helperがなく、
旧属性の検査はローカルfixtureでも失敗した。削除済み設定を渡す代わりに、
コンパイル済みmoduleを拒否し、Torch 2.14.0の`torch.compiler.set_stance("force_eager")`で
全参照推論を囲んだ。Torchの同APIは`torch.compile`指示を無効にする。
FP32、CPU、eager attention、eval、4 threads、seed 0、deterministic algorithms、
8192入力上限、mean pooling・prompt包含・L2正規化、同一forwardのlogit/sigmoidを維持する。
これらのAPI確認は、固定310m実モデルの数値一致の証明ではない。

確認した配布sourceのSHA-256（ホストの個人pathは省略）:

| 配布版内のpath | SHA-256 |
| --- | --- |
| ST 6.1.0 `sentence_transformers/base/model.py` | `60b8aaeac7e02bb67071b026461dedbf2ceef9820de21d0a5c60db621107fbef` |
| ST 6.1.0 `sentence_transformers/base/modules/transformer.py` | `895603634b3b63325cb7e5eaa2e8e2cdbc708b3b0e31a5c8b277ecd711a24be4` |
| ST 6.1.0 `sentence_transformers/base/modality.py`（R1-1調査で追加照合） | `e7defe33c33caf56e4279423a76585c6468622115a3aa58c019c578ed74fb74e` |
| ST 6.1.0 `sentence_transformers/cross_encoder/model.py` | `c8688911838df2acd48550e80dce2743c1f31dd85187f98d688517af1ce575e8` |
| ST 6.1.0 `sentence_transformers/sentence_transformer/modules/pooling.py` | `b6e96e0eac50ebc33d6310ef3b8d35b7cd2b7e0e521809552bba34c05c13d43d` |
| Transformers 5.17.0 `transformers/models/modernbert/modeling_modernbert.py` | `bac540e0d23974f55d0b066820fba55ec8243191acbe4c51b438ddc677a69670` |
| Transformers 5.17.0 `transformers/models/modernbert/configuration_modernbert.py` | `6bddc1471bde497acf90e786585df6bf2a4a43c913c572a1ba2e6b3315b2b648` |
| Torch 2.14.0 `torch/compiler/__init__.py` | `9843c22a125c5da699a3abef2b08feb0fbde937ba8b7f1499d174a8c8a4b52a1` |

## CI・Renovateと軽量検証

既存CIの`test`ジョブにPython setupと新規venvの通常install・pip check・版検査・unittestを追加した。
Rustの検証step、coverage/security/zizmor、setup/check/capture契約は維持した。
setup-python v7.0.0のcommit `5fda3b95a4ea91299a34e894583c3862153e4b97`を
[公式tag](https://github.com/actions/setup-python/releases/tag/v7.0.0)と照合して固定した。

Renovateの共有presetは[9f898d1のdefault.json](https://github.com/thkt/renovate-config/blob/9f898d1644ad6b4a1d35ff5facf637d3b488d36d/default.json)を参照した。
共有のCargo・Actions rule、`ignorePaths=["**/vendor/**"]`、7日のrelease ageは変更しない。
ローカルruleの対象は研究requirementsだけで、更新を同一グループへまとめ、自動マージを無効にする。
[公式設定仕様](https://docs.renovatebot.com/configuration-options/)に沿ってpath・manager・group・版制約を照合した。
運用中RenovateでのPR生成と設定validatorによる解決済みpreset全体の検証は未実施。

候補環境で実行した検証:

- `pip check`: 成功。今回の新規installではなく、Issueで作成済みの候補環境を読み取った結果。
- `reference.environment()`: Python 3.12.13、隔離環境、47依存の版・集合の一致を確認。
- `python -m unittest discover -s docs/research/issue-307 -p 'test_*.py'`: 15件成功。
  `PYTHONDONTWRITEBYTECODE=1`、`HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_PROGRESS_BARS=1`で実行。
  モデル取得なし。合成ModernBERTはhidden size 8・1 layerで、固定実モデルではない。
- `actionlint .github/workflows/ci.yml`、`zizmor --offline .github/workflows/ci.yml`、`git diff --check`: 成功。
  zizmorの既存suppressionは変更していない。オンラインのCI成功を意味しない。

旧10テストの比較器・manifest検査は維持した。rerankerの2テストはST 6の実constructorと
小さいローカルcheckpointを使うように改め、符号・飽和を含むlogit/scoreとID・token列の対応、
入力ごとに1 forward、forward/activation例外の無再試行を引き続き検出する。
ST 5固有の「activationが登録済み子module」という内部状態はST 6では廃止されたため再現しない。
新APIの公開経路を検証し、activation自体は引き続きTorch Moduleで渡す。

追加検証は、requirements更新と別テーブルの不一致、版違い・依存欠落・未記載依存の見逃し、
重複/非固定pin、Python版・隔離条件の逸脱、旧pooling設定の読込み失敗、正規化欠落、
コンパイル済みモデルによる演算条件の変更を対象にする。
既存の数値比較テストだけでは依存一覧やwrapper読込みを検査できないため追加した。
取得やGPUに依存せず、小さなfixtureで検出できる範囲に留めた。全モデル・長文の保証は増やしていない。
テスト削除・統合はなく、関連する旧失敗条件は保持した。テスト時間や性能の改善は主張しない。

## R1-1: 測定sourceの同定と再利用範囲

レビュー対象は`1521992f5aa30d1d35198599c5247d6c90792947bb07165b8f3053ca1c11be5b`。
初回レビューに先行する修正記録はなく、初回実装の報告は実測をホスト待ちとしていた。
レビューが発見した実測の測定元checkoutを読み取り、`reference.py`の実体が
[測定環境](results/numerical-353/environment.json)内のSHA-256
`714f1620040fc4ce4305195ddd551a56080a3fc46983c1bfbd3af04cdd3094e6`と一致することを確認した。
現行版は`6a8d7677330bc77ac30a0bc4d64df1c66737970432935b65523cb825704e6d49`。
両者の[差分](results/numerical-353/reference-update.patch)を保存した。現行版にこの差分を逆適用すると
測定版を復元できる。参照コード自体は今回の修正で変更していない。

差は`requirements()`の版表記の読取りだけだった。旧版の数字とドットだけの正規表現から、
PEP 440のpost releaseを許しpre-releaseを拒否する検査へ変わっている。
同関数以外の全module要素はAST一致し、`capture()`と`environment()`の本体は変更がない。
実際の47 pinに対する両版の`requirements()`の戻り値、候補環境での両版の`environment()`の
戻り値、保存済み全導入版（pipを含む48件）が一致した。実測のpinにはpost/pre-releaseはない。
したがって、この版検査変更は今回の入力と導入版で推論条件・出力を変えない。

測定環境の全source hashを現行ファイルと比較した。Rust source・Cargo.lock・入力・requirements・
`run.py`・`compare.py`は一致した。不一致は上記`reference.py`と、CI、CONTRIBUTING、README、
`test_reference.py`、Renovate設定だけで、後者5ファイルは`run.py numerical`から実行されない。
現行版で追加された環境テストと本記録も推論には使われない。CIやテストの成功はこの同一性から
推定しない。今後、推論・入力・依存・モデル条件が変われば、この再利用判断を適用できない。

[完了記録](results/numerical-353/complete.json)が指す全10生成物のhashをホストの実体と照合し、
現行比較器で両方の生出力を再集計して、保存されたcomparisonとのJSON値一致を確認した。
保存したのは環境・モデル・build・完了記録、差分、両comparisonだけである。
大きな4生出力・binary・モデル・privateログは公開物に含めず、ホストの元出力
`rurico-353-numerical-20260928`に保持する。
`complete.json`は非同梱の生出力も識別するため、公開ファイルだけでは再集計できない。
原本のhashと今回保存した小さなJSONのhashは同一で、原本を書き換えていない。

## 新依存環境の固定実モデル観測

実行時刻は2026-09-28 02:54:44–03:00:33 JST。機種はApple M3 / Mac15,3、macOS 27.0、
Python 3.12.13、Rust 1.98.1、Xcode 27.0 / Metal 32023.921。
依存と全sourceは[environment.json](results/numerical-353/environment.json)、固定モデル/tokenizerの
revision・重み/config/tokenizer hashは[models.json](results/numerical-353/models.json)にある。
embedding revisionは`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`、rerankerは
`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`。両モデルともCPU / FP32 / eager / eval、
force_eager、4 threads、seed 0、deterministic algorithmsの記録を生出力から確認した。

[build.json](results/numerical-353/build.json)のbinary SHA-256は
`7e5108a80b6e0c442cc723cd4494238054c0f2f3167efefd527f1295958daead`。
既存の`cargo test --locked --lib --features smoke --no-run --message-format=json`でbuildし、
各ignored testを`--exact --ignored --nocapture --test-threads=1`で順に実行している。
`run.py`の完了前検査はsource・binary・model内容の不変性を確認する。
[ホストの負荷隔離記録](results/numerical-353/host-python-validation.json)では、標準Rust check開始前に
開発controllerを停止し、固定したcheckoutでnumericalを実行して、subprocess終了後に再開した。
測定中のprocess確認でも並行するCargo/rustc buildや他のruricoモデルテストは見つからず、
source・binary・model cacheの変更も行っていないと記録されている。
負荷隔離の確認根拠はこのホストの実行管理記録であり、完了記録だけから推定したものではない。
連続したprocess観測を独立再現した証拠ではなく、process確認だけで短い活動や未知のGPU利用を
すべて検出できるとはしない。

| 同一token比較 | batch / 行 | 観測値 | 既存診断基準 |
| --- | --- | --- | --- |
| [embedding](results/numerical-353/comparison-embedding.json) | 23 / 27 | 最小cosine 0.9999999999929701、最大絶対差 7.152557373046875e-7 | cosine ≥ 0.99999、最大絶対差 ≤ 1e-5、全行基準内 |
| [reranker](results/numerical-353/comparison-reranker.json) | 12 / 16 | 最大logit差 1.1563301086425781e-5、最大score差 1.7881393432617188e-6 | logit ≤ 1e-4、score ≤ 2.5e-5、全行基準内 |

raw tokenとwrapper後tokenはembedding 10/10入力、reranker 6/6入力で一致した。
embeddingのpadding 12単位、reranker 6単位も両実装とも基準内。そのうち2単位と1単位は
実長とbucket長が等しい同一観測の共有であり、独立したpadding検証ではない。
reranker順位6条件で不一致なし。同文候補`library` / `library-copy`の同点を近接候補として保持し、
logit差2e-4超の候補対に逆転はなかった。異なる文が僅差になる一般的な順位安定性は未検証。

embeddingの公開wrapper比較も全10入力が基準内で、公開API対直接演算の差は0だった。
旧ST 5.1.1の[report.md](report.md#公開wrapperとの相違は末尾空白)で末尾token 271が除去された
3入力は、今回は126 / 130 / 8190 tokenのまま一致している。ST 6.1.0の
`Transformer.preprocess` → `InputFormatter.parse_inputs` → processorの文字列経路には、
旧`Transformer.tokenize`の`str(s).strip()`がない。配布sourceと今回のtoken列はこの前処理差の
解消に整合する。製品側にstripを追加した結果ではなく、旧観測を新環境に読み替えてもいない。
rerankerの公開wrapper対製品score最大差は1.7881393432617188e-6、公開API対直接演算の差は0。
生logitとsigmoidは同じforwardから記録されている。

embedding hidden probe最大差は0.000885009765625。ただし観測は各行の3位置だけで、
正規化embeddingの基準をhiddenへ適用しない。残る微小な数値差のkernel・演算順序別の寄与は未分離。
全layer・全入力、BF16・量子化、速度・検索品質の改善はこの比較から主張しない。

## R1-2: 否定テストの修正とホスト検証

安定版限定の検査が消えても、旧fixtureは導入版との不一致で失敗した。
必須ライブラリの検査が消えても、空requirementsに元の依存が残るため集合不一致で失敗した。
`test_environment.py`で前者は要求版と導入版を同じpre-releaseへ揃え、後者はpipだけの環境へ揃え、
それぞれ目的のエラー文を確認するようにした。追加・削除・統合はせず、既存3テスト内を修正した。
正常な更新版/post release、重複・非固定pin、版ずれ・欠落・未記載依存、Python版・隔離条件の
既存の検出条件は保持する。数値比較テストでは守れない入力境界であり、モデル取得やGPUを使わない
小さなfixtureなので維持する価値がある。速度改善・新しい実モデル保証は主張しない。

修正後に`PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s docs/research/issue-307 -p test_environment.py -v`
を候補環境で実行し、3件成功した。`reference.require`をメモリ上で差し替え、安定版限定と
必須ライブラリの検査を別々に無効化すると、いずれも対応するassertが`ValueError not raised`で失敗した。
他のテストは成功し、別のエラーによる検出ではない。無効化はソースへ保存していない。
修正後のテストSHA-256は`84349f535050f276e1d4e8c7126c111f222593161bbf6e47eb8717df98bd13d0`。

候補の新規通常install・pip check成功はIssueとホストの候補install記録にある。
その`compatible.txt`のhashは`911701ce00baa55060ca1ba2786ea22f90d314dd9348bf891a78df4311ed2e31`で、
実体の47 pinを正規化して現行requirementsと全件一致することを今回確認した。
installログの全47依存の導入完了も確認した。これは既存候補環境の証拠であり、後述の新規環境での
検証とは区別する。個人pathを含むinstallログは公開しない。

さらに、修正前の現行requirementsを使った[ホストの新規Python検証記録](results/python-353.json)を確認した。
2026-09-28 02:57:38–02:58:29 JST、Python 3.12.13を指定した`uv venv --seed`で別の新規環境を
作り、`python -m pip install -r docs/research/issue-307/requirements.txt`、`pip check`、
厳密な版検査、全Pythonテスト15件が終了0だった。ログも導入完了・整合成功・15件成功に一致した。
この記録のrequirements hashは現行と同一で、導入版・集合は実モデル環境とも一致した。
記録されたPython sourceのうち現在と異なるのは今回修正した`test_environment.py`だけである。
公開版は個人pathをplaceholderへ置換し、非公開ログを除外した。原記録と各ログのhash、
source hash、終了値、導入版を残し、修正前の実行であることを明示した。
測定に使った候補環境と、この新規install検証環境は同じ依存版の別環境である。

[修正後のホスト検証記録](results/numerical-353/host-python-validation.json)は
2026-09-28 03:14:29 JST時点の追加証拠である。上記の新規環境へのinstall後に変わったのは
`test_environment.py`のfixtureだけで、requirementsとruntimeコードは変わっていない。
同じ環境でR1-2修正後に`pip check`、`reference.environment()`、全Pythonテストを再実行し、
すべて終了0、15件成功だった。記録の8 Python source hashとrequirements hashは現行ファイルに一致し、
installログと修正後3検査のログも記録hashに一致することを照合した。
修正前の[python-353.json](results/python-353.json)は当時の結果として保持し、修正後の成功へ読み替えない。

review-2（対象`e3c43df0f84074dde2ab32016d161785b9265c885ab9c95c203c6770f8377038`）は、
R1-1・R1-2を修正済みとし、修正後sourceの標準check成功と上記追加証拠を確認した。
R2-1の原因は、この追加証拠が本文へ反映されず、完了済みの検証を待機扱いにしていたことだった。
本文から証拠へリンクし、新規install、修正後検査、負荷隔離の根拠と限界を反映した。
依存・推論実装・入力に変更はなく、この文書修正のための再install・再測定は不要である。
変更後の標準check・本記録と追加証拠を含む最終独立評価・同じheadの既存CIはホストで更新する。
許容差・演算条件・撮影不要の契約は変更しない。

初回実装の軽量検証時のcheckout内source SHA-256（R1-2修正後のテストhashは上記）:

| path | SHA-256 |
| --- | --- |
| `requirements.txt` | `b303c0bc5817f621d80e1521a61024493cc15a95130e891c15b5ddb643e16251` |
| `reference.py` | `6a8d7677330bc77ac30a0bc4d64df1c66737970432935b65523cb825704e6d49` |
| `test_reference.py` | `ea8a8c170beab07342171cb2f0553193995000e1ebff1898153df2b6e6b1a1e2` |
| `test_environment.py` | `d1c39e979dc8e8df2c41f916ca1ef36e3a854dda2924139ea0b30bde37c172a5` |
| `inputs.json` | `ef7943a637a6cfd4a39af83ef16592313a226fa25bc429451879746c49371f37` |
| `compare.py` | `a9ee60d66954b4a74e0d22830ee2a858e20dfb03814425b83bde723d02b540b4` |
| `run.py` | `f392efc6ce7356da6979669f3f22facf587b0344ebcae4b75c9bd2f65cdc61a3` |

初回実装の15件成功はその時点の記録として保持する。R1-2修正後の15件成功は上記の追加証拠で
確認済みであり、今回の文書修正後に更新する標準check・最終独立評価・CIとは区別する。
