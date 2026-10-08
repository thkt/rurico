# コメント整理後の版照合

比較元は公開head `b24ea787dcd59d33e04aa9b8255f3310f2070aa1`、比較先は以下のbyte hashで特定する未commitの4 source。R4両manifestの `current/` は比較元の4 sourceと一致した。R4のmanifest・raw・表・probe・tests・Clippyの原資料は変更していない。過去のcheck・独立評価・対象54件の成功は比較元の実行事実であり、整理後の成功ではない。

| source | 整理前 SHA-256（R4測定版） | 整理後 SHA-256 | 非コメント行 SHA-256（両版共通） |
| --- | --- | --- | --- |
| `src/storage/query_normalize.rs` | `e8dc6383a34a0e845c8cf5ede269deec3e6fa872c5085893fb85141d46b4f4a8` | `7cbd526169b6627fb29807a448018883dae6fb7f425603ecd034b792503a7075` | `6a4c12086f7eab3234826e262e2f599045f264a01f2a3230a2b66b1d77181802` |
| `src/storage/query_normalize/tests.rs` | `f383c60e62ec1555ad60ed73b2b0cd1a25c4c29295a5948f704725b8c5003d7a` | `c436697b3829cb1c1c07693980bf52a39110bd7f741b0deff44c4aed8b21add0` | `ee8d33eeba9afb2ebc191fd82b09279116b7aab8b27c30416472d12be47fb9ba` |
| `src/storage/search.rs` | `71eef25c7bb9be95e2835b8a60518ac628ec6ad13f9be8aa9ba3b7d4c6acb3e3` | `3541d1c05733a7df19b962345ee53e7bd6d45bf5b9e25d75d127cc73f7ce30c9` | `7af2892ec4c4abfe1cef4a0e09cfa5b41e4fb6ae3c1ffb3db1c8d1bca5c29803` |
| `src/storage/search/tests.rs` | `947f93aa15284660917dbe5496bc54ac138d66eeee8fa69950fa98479022599f` | `08be73df9d6d49d679ecdf81e7783e61406ff69b095740986c664409e12b048c` | `cb354702bc3d28665e68ed2cda540afba70db0bd4a838dd1127167fcfc1ebf89` |

Cargo.lock SHA-256: `743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。R4 probe SHA-256: `db558484541aebbf31503f3b8e8068841ebf5e453307185bdd5115e4060ae021`。双方は両manifestと一致し、依存版・probeは変更していない。

照合は各ファイルの先頭空白を除いた行が `//` で始まる場合だけ、その行全体を除外してbyte比較した。空行・それ以外の空白、文字列、API、assertion、数値条件を含む残りの行は完全一致した。`git diff --unified=0` の追加・削除行もすべて独立した行コメント/doc commentであることを確認した。差分を元の文字列境界と照合し、文字列内の行を除外していないことを確認した。対象には位置依存の `line!` / `column!` や自sourceの `include_str!` / `include_bytes!` はない。この照合は実行部分の同一性を示すが、doc commentのbyte同一性、コンパイラーのdebug情報やbinary hashの一致、現行版のテスト成功を証明しない。

再照合はcheckoutルートで次を実行する。ハッシュは上表へも照合する。この手順は今回の4 sourceに限定し、一般的なRust lexerやコメント削除器として使わない。

```sh
python3 - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
base = "b24ea787dcd59d33e04aa9b8255f3310f2070aa1"
manifest = json.loads(Path("docs/benchmarks/issue-312/repair-4/start/manifest.json").read_text())
for path in ("src/storage/query_normalize.rs", "src/storage/query_normalize/tests.rs",
             "src/storage/search.rs", "src/storage/search/tests.rs"):
    old = subprocess.check_output(["git", "show", base + ":" + path])
    new = Path(path).read_bytes()
    def without_comments(data):
        return b"".join(line for line in data.splitlines(keepends=True)
                        if not line.lstrip().startswith(b"//"))
    assert hashlib.sha256(old).hexdigest() == manifest["sources"]["current/" + path]
    assert without_comments(old) == without_comments(new), path
    print(path, hashlib.sha256(new).hexdigest(),
          hashlib.sha256(without_comments(new)).hexdigest())
PY
```

R4の数値は変更前のbyte版で観測されたCPU合成入力の結果として保持する。現在版には実行部分・入力・依存が同じという限定で処理説明と観測を適用し、再測定済みや新しい性能効果とは書かない。背景負荷未確認、時間増加・分布の重なり、公開String化を含む時間、SQLite内allocationが対象外という限界は元のREADMEとprobeのままである。現在版の標準checkと変更文書を含む独立評価はホストが新たに実行する。通常checkにない実モデル/検索品質の再測定をこのコメント変更の受入条件には加えない。

次の評価のassessmentsとhandoffには、公開String API・検索意味/エラー契約、query内だけの再利用、照会/割当削減と時間増加条件、R4の歴史的byte版と上記限定、現行check/CIの未確認を残す。既存添付への公開リンクも本文内に保持する対象として、次の資料を引き継ぐ。

- [R4 probe](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/probe.rs)
- [再実行手順](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/run.py)
- [schema拒否の修正前ログ](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-2/schema-red.txt)
- [R3比較元manifest](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-3/start/manifest.json)
- [開始版比較 manifest.json](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/start/manifest.json)
- [開始版比較 raw.jsonl](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/start/raw.jsonl)
- [開始版比較 table.md](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/start/table.md)
- [開始版比較 tests.txt](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/start/tests.txt)
- [前回版比較 manifest.json](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/prior/manifest.json)
- [前回版比較 raw.jsonl](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/prior/raw.jsonl)
- [前回版比較 table.md](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/prior/table.md)
- [前回版比較 tests.txt](https://github.com/thkt/rurico/blob/b24ea787dcd59d33e04aa9b8255f3310f2070aa1/docs/benchmarks/issue-312/repair-4/prior/tests.txt)

これは公開済み資料への参照であり、新しい添付の公開・PR本文更新・独立評価acceptedを実施した記録ではない。PRはdraftを維持し、commit・push・公開・ready切替は行わない。
