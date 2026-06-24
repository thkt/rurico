---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Adopt subprocess probe exit-code IPC wire-format contract

## Context and Problem Statement

`src/model_probe.rs:10-26` は、MLX モデルのロード可否を検証する probe subprocess (現在のバイナリを re-exec する) と親プロセスの間で交わす **exit code の IPC wire format** を表形式で定義する。`0 = Available`, `1 = ModelLoadFailed`, `3-6 = SetupRejected` の各原因, `7-8 = SubprocessFailed (IO)`, code `2` は意図的に未使用、signal kill は `BackendUnavailable`。さらに `SetupReason` enum (`model_probe.rs:76-91`) は `#[repr(i32)]` で各 variant の discriminant を `PROBE_EXIT_*` 定数に 1:1 で結びつける。この対応は親子別プロセスの ABI 契約であり、Rust の型システムは「親の解釈表」と「子の exit code」と「`SetupReason` discriminant」の三者一致を強制できない。`#[non_exhaustive]` な enum に variant を追加する際、対応する exit code を割り当て、親の解釈を更新し、code 2 を未使用に保つ保守規則は doc コメントにしか存在しない。census audit (`docs/audit/2026-06-24-020141-adr-gaps.md` candidate C8) が nominate した。

なお、この exit code 経路の**セキュリティ的側面** (probe exit code を filesystem oracle として悪用する経路、`DYLD_*` 環境変数の遮断) は `docs/THREAT_MODEL.md` の SEC-003 / SEC-004 が扱う。本 ADR はそれを重複させず、wire format の保守契約に限定する。

## Decision Drivers

- 親子別プロセスの exit code 契約は型で強制できず、片方だけ変えると probe の解釈が無言でずれる
- `SetupReason` discriminant と `PROBE_EXIT_*` 定数の 1:1 対応は `#[repr(i32)]` 依存で、新 variant 追加時の同期忘れを lint が拾えない
- code 2 を未使用に保つ規則は明文の根拠なしには「歯抜け」に見え、将来の貢献者が埋めてしまう risk
- downstream バイナリ (rurico を host する側) はこの wire format に依存して probe を呼ぶため、互換破壊は cross-repo に波及する

## Considered Options

- exit-code wire format と `SetupReason` 保守規則を IPC 契約として ADR で pin する (status quo を明文化)
- exit code を単純な成功/失敗 2 値に縮約し、原因は stderr テキストのみで伝える
- probe 結果を構造化形式 (JSON / 長さ付きバイナリ) で stdout に出力し exit code は使わない

## Decision Outcome

Chosen option: **"exit-code wire format と保守規則を IPC 契約として pin する"**。exit code は signal kill と正常終了を区別でき (`BackendUnavailable` 判定)、親が child stdout を full parse せずに分類できる軽量さがあり、`#[repr(i32)]` による discriminant 一致が既に実装の骨格になっている。2 値縮約は SetupRejected の原因区別 (env 不足 / canonicalize 失敗 / cache 外パス) を失い、構造化出力への移行は probe の最小依存 (stdout は ACK トークンのみ) という設計前提を重くする。

### Consequences

- Good, because exit code → `ProbeError` の対応、code 2 予約、kill 判定が単一の参照点を持つ
- Good, because `SetupReason` variant 追加の保守手順 (exit code 割当 + 親解釈更新 + code 2 維持) が明文化される
- Bad, because exit code 空間は単一バイト範囲に限られ、原因種別が増えると枯渇しうる
- Bad, because 三者一致は型で強制できず、新 variant の同期忘れは code review に委ねる

## Confirmation

- `src/model_probe.rs:10-26` の exit code 表が parent 解釈の source of truth
- `src/model_probe.rs:47-74` の `PROBE_EXIT_*` 定数群が child 側の exit code を定義
- `src/model_probe.rs:76-91` の `SetupReason` `#[repr(i32)]` discriminant が各定数に 1:1 で結びつく
- code 2 が未使用である事実は `model_probe.rs:23` の doc コメントで明示
- `SetupReason` に variant を追加する PR では「対応 exit code 割当 / 親解釈表更新 / code 2 維持」が code review checklist

## Pros and Cons of the Options

### Pin exit-code wire format as IPC contract

exit code 表・`PROBE_EXIT_*` 定数・`SetupReason` discriminant の三者を契約の source of truth とする。

- Good, because signal kill と正常終了を区別でき、親は stdout を full parse 不要
- Good, because SetupRejected の原因区別が保たれる
- Bad, because exit code 空間が有限で原因増加に弱い

### Collapse to binary success/failure exit code

成功/失敗の 2 値のみ、原因は stderr テキストで伝える。

- Good, because wire format が最小化される
- Bad, because SetupRejected の原因区別が失われ、親が stderr を parse する依存が生まれる
- Bad, because stderr テキストは安定契約にしづらく downstream が脆くなる

### Structured stdout output, no exit code semantics

JSON 等を stdout に出し exit code は使わない。

- Good, because 任意の構造化情報を渡せる
- Bad, because probe の stdout は ACK トークンのみという最小依存設計を重くする
- Bad, because signal kill 時は stdout が空で、結局 exit/signal 判定が要る

## More Information

### Implementation Guidelines

- `SetupReason` に variant を追加する場合: (a) 新しい `PROBE_EXIT_*` 定数を割り当て、(b) `model_probe.rs:10-26` の解釈表に行を追加し、(c) 親側の分類ロジックを更新し、(d) code 2 は未使用のまま残す
- exit code の数値や意味を変える / code 2 を割り当てる変更は downstream バイナリの互換を破るため本 ADR を supersede する
- wire format の互換変更は downstream の追従責任 (各リポで個別対応) を伴う

### Related

- `docs/THREAT_MODEL.md` SEC-003 (probe exit code を filesystem oracle として扱う経路) / SEC-004 (`DYLD_*` 遮断) — 本 wire format のセキュリティ側面
- ADR 0007 (library logging boundary) — probe stderr メッセージの扱いと隣接
