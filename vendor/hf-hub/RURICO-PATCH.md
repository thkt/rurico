# Issue #320: 固定revisionのsnapshot公開

原資料はcrates.ioのhf-hub 1.0.0 archive。SHA-256は
`e7ccb6bcc85dec15413ef5949879f9a5497ca4568ed702547eb83fac23376e5c`で、変更前のCargo.lock checksumと照合した。
`src`、`tests`、README、Cargo.toml.origを同archiveから複製した。上流の宣言はApache-2.0で、ライセンス本文をLICENSE-APACHEへ添付する。

Unixの`src/cache/storage.rs`は既存pointerをunlinkせず公開する。同じcanonical blobへのpointerは再利用し、壊れたpointerや異なるblobへのpointerはエラーで残す。破損cacheは自動unlinkで修復しない。
HTTPの通常取得・304・既存blobとXetのfinalizeは、引き続き同じstorage関数を呼ぶ。Windowsの動作は変更していない。ruricoの終了管理はUnixを対象とする。

Cargo.tomlは合成検証用feature `rurico-test-support`と独立workspace宣言を追加する。src/lib.rsは同featureで本番の同期公開primitiveを公開し、symlink直前のcallbackで停止点を指定できるようにする。通常の公開経路のcallbackは何もしない。
ルートworkspaceはこのpathへ直接依存するため、git経由の利用側にも修正が適用される。vendorをworkspaceのcheck対象から除外し、root packageの暗黙のmembershipを使って元の2 packageを保つ。
上記以外のsource・testsは`cargo fmt --all`による書式変更だけで、機能変更や整理の効果とは数えない。
依存更新時は固定archiveとの差分を再確認する。合成検証は実networkやHF finalize全体、上流の全テスト実行を証明しない。
