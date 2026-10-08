# モデル種別の拒否検証

2026-10-08、[Issue #361](https://github.com/thkt/rurico/issues/361) の受入検証を実施した。
開始commitは `24725a72be44300afc24186b82b14bcb3f5f9d3d`。
ホストで着手時と検証後にremote mainを取得し、同じSHAであることを確認した。
Issue固定版から開始版まで、artifacts本体・テスト・weight loader・weight_loadに差分はない。

## 方法と結果

[CONTRIBUTINGの手順](../../../CONTRIBUTING.md#モデル種別の拒否条件issue-361) に従い、
開始版と未commit差分を含む隔離コピーで本番helperとcandidate経路を実行した。
3種類のheadは本番のprefix定義から独立した固定期待値で確認する。
kindのprefix検査と、config由来の全必須重み・shape・dtype・data range検査を区別する。
既存fixtureを使い、モデルの取得・推論は行っていない。

| 版 | 実行対象 | 結果 | 終了コード |
| --- | --- | --- | --- |
| 通常版 | artifacts全27件 | 27件成功、421件対象外 | 0 |
| embed head拒否分岐を削除 | helper・candidateのembed拒否 | 2件とも拒否期待のunwrap_errで失敗 | 100 |
| 正常版からnorm必須prefixだけを削除 | helper・candidateのreranker欠落拒否 | normだけを欠くケースで2件ともunwrap_errが失敗 | 100 |
| 正常版へ復元 | artifacts全27件 | 27件成功、421件対象外 | 0 |

両変異は対象が誤受理されてOkとなったため失敗した。ビルド・環境失敗を検知成功としていない。
head削除時は固定ケースの先頭classifier、norm削除時は他のheadが揃うnorm欠落を検知した。
同じfeatures `test-support,test-mlx,smoke`、ci profile、`--run-ignored default` で実行した。
全体checkや実モデル成功の代用ではなく、対象filter以外は未実行である。
各版1回の確認で、速度・維持費・不安定さの改善量は測定していない。

## 対象版と環境

Apple Silicon arm64、macOS 27.0.1（26A434）、Xcode 27.0（27A266a）、
Metal 32023.921、Rust 1.99.0、nextest 0.9.146。
locked依存とMLX-only構成を保持した。Cargo.lock SHA-256は `743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。
測定したテストのSHA-256は `0021e48bcc9b820e24819ce36a410cb299f3a0365363e529be606c459d57a435`。
測定後にcargo fmtで空行だけを整理し、現在のSHA-256は `15ddee60c102c7ee1eafb3c1d4fd7c98974fab31849a0206e7bb42004553f739`。
空行を除いた全行と本番artifacts.rsは測定した復元版と一致する。

既存コメントの装飾見出し、テスト名の再述、コードで明らかなfixture説明を削除した。
独立した期待prefixの理由、誤拒否を防ぐHF key形式、部分削除後の継続理由は保持した。
コメントと空行の整理による実行内容の変更はない。

通常check・独立評価・同じheadのCIは、この対象限定検証の後に実施する。
実モデル・公式数値・検索品質・性能・consumer実行経路は今回測定していない。
