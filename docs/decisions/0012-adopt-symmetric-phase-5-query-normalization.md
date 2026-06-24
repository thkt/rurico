---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Adopt symmetric Phase 5 query normalization

## Context and Problem Statement

`src/storage/query_normalize.rs` (Phase 5, Issue #69) は FTS5 の token stream に対する正規化 (NFKC → ASCII lowercase → whitespace collapse) を定義する。module doc は「index する text と query する text の**両側**に同じ正規化を適用しなければならず、片側だけに適用すると index 側が未正規化形を保持して match を silent に取りこぼす」と表明する。これが第 1 の incomplete-contract。第 2 に、`pre_phase_5_disabled()` (`query_normalize.rs:71`) は serde-default factory で、Phase 5 以前に採取された eval baseline ファイルが該当フィールドを欠くため、missing field を runtime `Default` (全 ON) ではなく全 OFF に解決する。これは「過去の baseline は採取時の数値で round-trip する」という invariant を守るためで、runtime default と取り違えると baseline の比較基準が無言でずれる。census audit (`docs/audit/2026-06-24-020141-adr-gaps.md` candidate C11) が nominate した。両 invariant とも型では強制できない。

## Decision Drivers

- index 側と query 側の非対称な正規化適用は FTS5 match を silent に取りこぼす (検索 recall の劣化)
- serde-default が runtime `Default` (全 ON) に解決されると、過去 baseline が採取時と異なる正規化で再評価され、metric 比較が無効になる
- 「両側に適用」「missing field は全 OFF」のどちらも型・lint で守れず、新規 caller / baseline 追加で破られうる
- 正規化ステップ (NFKC のみ採用、ひらがな↔カタカナは非マッピング、Unicode case folding は保留) は実測トレードオフの結果で、変更には根拠の参照点が要る

## Considered Options

- 両側対称適用 + baseline 全 OFF default を cross-cutting invariant として ADR で pin する (status quo を明文化)
- index 側のみ正規化し query 側は raw のまま受ける
- serde-default を runtime `Default` (全 ON) に統一し、過去 baseline を再採取する

## Decision Outcome

Chosen option: **"両側対称適用 + baseline 全 OFF default を invariant として pin する"**。正規化は index 時と query 時の FTS5 token stream を一致させて初めて recall を保つため対称性が本質で、過去 baseline の round-trip は eval の連続性 (ADR 0006 で amici へ移譲した eval harness) に必要。runtime default への統一は既存 baseline を全て無効化し、index 片側適用は recall を構造的に壊す。

### Consequences

- Good, because 「両側適用」と「baseline は全 OFF」の 2 invariant が単一の参照点を持つ
- Good, because 正規化ステップの採否トレードオフ (NFKC 採用 / かな非マッピング / case folding 保留) が文書化される
- Bad, because 正規化ステップを追加する変更は index 再構築 (既存 index の token stream がずれる) を伴い、影響が cross-cutting
- Bad, because 両 invariant とも型で強制できず、新規 caller が片側だけ正規化する / 新 baseline が field を明示しない regression は code review に委ねる

## Confirmation

- `src/storage/query_normalize.rs:1-19` の module doc が「両側適用」契約の source of truth
- `src/storage/query_normalize.rs:68-73` の `pre_phase_5_disabled()` doc が「missing field → 全 OFF」を表明し、`amici::eval::baseline::BaselineSnapshot` の serde-default に紐づく
- `normalize_for_fts` の idempotent 性 (`query_normalize.rs:75-79` doc) により、既正規化入力への二重適用は不動点を壊さない
- index 経路と query 経路の双方が同じ `QueryNormalizationConfig` で `normalize_for_fts` を呼ぶことが review checklist

## Pros and Cons of the Options

### Pin symmetric application + baseline-OFF default

index/query 両側が同一 config で正規化し、欠落 field は全 OFF に解決する。

- Good, because recall (対称性) と eval 連続性 (baseline round-trip) を同時に守る
- Good, because 正規化ステップのトレードオフが ADR で説明可能
- Bad, because 正規化変更時に index 再構築が要る

### Index-side only normalization

index する text のみ正規化し query は raw で受ける。

- Good, because query path の正規化コストが消える
- Bad, because query が未正規化形 (full-width 等) のとき index の正規化形と match せず recall が構造的に壊れる

### Unify serde-default to runtime Default

欠落 field を全 ON に解決し、過去 baseline を再採取する。

- Good, because default が 1 つになり概念が単純
- Bad, because Phase 5 以前の全 baseline が採取時と異なる正規化で再評価され、過去比較が無効になる
- Bad, because 再採取コストと、再現不能な過去環境の baseline 喪失リスク

## More Information

### Implementation Guidelines

- FTS5 に text を index する / FTS5 に query を発行する新規 caller は、必ず同一の `QueryNormalizationConfig` で `normalize_for_fts` を通す
- 正規化ステップ (NFKC / lowercase / whitespace) を追加・変更する場合は本 ADR を supersede し、(a) recall への影響、(b) index 再構築の要否、(c) 既存 baseline との互換を測定して新 ADR に記録する
- 新しい eval baseline スキーマで正規化 config を必須化する場合、`pre_phase_5_disabled` の後方互換契約を破るため supersede 必須

### Related

- ADR 0001 (typed FTS query contract) — FTS5 sanitization の前段
- ADR 0006 (eval harness migration to amici) — baseline round-trip が依存する eval 基盤
