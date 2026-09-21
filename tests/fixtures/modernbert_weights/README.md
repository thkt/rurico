# 公式 safetensors ヘッダー

2026-09-21 に固定 revision のローカル HF キャッシュから、先頭8バイトが示す JSON 領域を
抽出し、末尾の空白 padding のみ改行へ置換した。JSON 本文は実ヘッダーとバイト単位で一致し、
tensor 本体は含まない。config は隣の
[modernbert_configs](../modernbert_configs/README.md)を再利用する。

| ファイル | 取得元 | tensor 数（metadata 除外） |
| --- | --- | --- |
| `ruri-v3-310m.header.json` | [18b60fb の model.safetensors](https://huggingface.co/cl-nagoya/ruri-v3-310m/blob/18b60fb8c2b9df296fb4212bb7d23ef94e579cd3/model.safetensors) | 152 |
| `ruri-v3-reranker-310m.header.json` | [bb46934 の model.safetensors](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m/blob/bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3/model.safetensors) | 156 |

両方とも F32。embedding は直下、reranker backbone は `model.` 配下で、名前変換は不要。
両者の先頭 encoder 層には `attn_norm` がなく、各 LayerNorm の bias もない。
reranker の head は `head.dense.weight` と `head.norm.weight`、最終層は
`classifier.weight` と `classifier.bias` を持つ。共有重みの別名や永続 RoPE buffer は含まれない。
[Transformers 4.48.3 の定義](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/modernbert/modeling_modernbert.py)
でも先頭層は Identity、RoPE buffer は非永続、PredictionHead と最終 classifier の bias は別設定である。
masked-LM decoder の共有重みは、この2つの architecture の対象外。

採用範囲と bias 修正の互換性は [Issue #300](https://github.com/thkt/rurico/issues/300) を正本とする。
テストはこのヘッダーと sparse なゼロ埋め本体を組み合わせて受理を検査するため、
公式重みの数値や load 成功の証明にはならない。実モデル検証は
[CONTRIBUTING](../../../CONTRIBUTING.md#重みの読込み検証) のホスト手順で行う。
revision を更新する場合は新しい実ヘッダーと config を照合してから更新し、期待 shape から
合成した JSON で公式 fixture を置き換えない。
