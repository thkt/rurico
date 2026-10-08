//! Query normalization for FTS5 indexing and retrieval.
//!
//! Resolves common Japanese/Latin notation drift before sanitization and
//! short-term expansion. Applied to **both** indexed text and query text so
//! the FTS5 token streams agree — applying only to one side leaves the index
//! holding the un-normalized form and silently misses matches.
//!
//! # Pipeline order
//!
//! 1. NFKC compatibility composition (`unicode-normalization`).
//!    Folds full-width Latin/digits to half-width and unifies compatibility
//!    characters. Hiragana ↔ Katakana is intentionally **not** mapped — that
//!    requires a separate Issue once evidence justifies it.
//! 2. ASCII lowercase. Japanese characters are left untouched (case is a
//!    Latin-only concept here); Unicode case folding is deferred to a future
//!    Issue if measurements show a need.
//! 3. Whitespace collapse. Trims leading/trailing whitespace and collapses
//!    runs to a single space so trigram boundaries stay deterministic across
//!    full-width spaces (`U+3000`) NFKC-folded into ASCII spaces.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfkc_quick};

/// Per-step toggles for the [`normalize_for_fts`] pipeline.
///
/// Every step defaults to **on** at runtime — the pipeline is meant to be
/// applied transparently. The `serde` `Deserialize` path uses
/// [`pre_phase_5_disabled`] so historical baseline files (captured before
/// Phase 5 existed) round-trip with normalization disabled, preserving the
/// numbers they were captured under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryNormalizationConfig {
    /// Apply NFKC compatibility composition (full-width → half-width Latin).
    pub nfkc: bool,
    /// Apply ASCII lowercase (`A-Z` → `a-z`). Non-ASCII characters untouched.
    pub ascii_lowercase: bool,
    /// Trim and collapse runs of whitespace to a single space.
    pub collapse_whitespace: bool,
}

impl Default for QueryNormalizationConfig {
    fn default() -> Self {
        Self {
            nfkc: true,
            ascii_lowercase: true,
            collapse_whitespace: true,
        }
    }
}

impl QueryNormalizationConfig {
    /// All steps off — the literal pre-Phase-5 behaviour.
    ///
    /// Used by the serde-default path on `amici::eval::baseline::BaselineSnapshot`
    /// (eval harness migrated to amici per ADR 0006) so a baseline file written
    /// before normalization existed round-trips with the same metric values it
    /// was captured under.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            nfkc: false,
            ascii_lowercase: false,
            collapse_whitespace: false,
        }
    }
}

/// Serde-default factory: pre-Phase-5 baselines lacked this field, so a
/// missing field must resolve to all-OFF, **not** to runtime [`Default`].
#[must_use]
pub fn pre_phase_5_disabled() -> QueryNormalizationConfig {
    QueryNormalizationConfig::disabled()
}

/// Apply the configured normalization pipeline to `text`.
///
/// Idempotent for any config: `normalize_for_fts(normalize_for_fts(x, c), c) ==
/// normalize_for_fts(x, c)`. Callers can layer this over already-normalized
/// input from downstream consumers without breaking the fixed point.
///
/// Returns the input unchanged when every step is disabled (avoids the NFKC
/// allocation on the hot path when callers explicitly opt out).
#[must_use]
pub fn normalize_for_fts(text: &str, config: &QueryNormalizationConfig) -> String {
    normalize_for_fts_cow(text, config).into_owned()
}

/// Borrow unchanged text inside the query pipeline; the public API stays owned.
pub(super) fn normalize_for_fts_cow<'a>(
    text: &'a str,
    config: &QueryNormalizationConfig,
) -> Cow<'a, str> {
    let mut buf = if config.nfkc {
        match is_nfkc_quick(text.chars()) {
            IsNormalized::Yes => Cow::Borrowed(text),
            IsNormalized::No => Cow::Owned(text.nfkc().collect::<String>()),
            IsNormalized::Maybe => normalize_nfkc_maybe(text),
        }
    } else {
        Cow::Borrowed(text)
    };
    if config.ascii_lowercase && buf.bytes().any(|b| b.is_ascii_uppercase()) {
        buf.to_mut().make_ascii_lowercase();
    }
    if config.collapse_whitespace
        && let Cow::Owned(collapsed) = collapse_whitespace(&buf)
    {
        buf = Cow::Owned(collapsed);
    }
    buf
}

/// Reuse the iterator that checks equality: copy the equal prefix only when
/// the first difference requires ownership, then consume the remaining output.
fn normalize_nfkc_maybe(text: &str) -> Cow<'_, str> {
    let mut original = text.char_indices();
    let mut normalized = text.nfkc();
    while let Some(next) = normalized.next() {
        let offset = match original.next() {
            Some((_, ch)) if ch == next => continue,
            Some((offset, _)) => offset,
            None => text.len(),
        };
        let mut output = String::with_capacity(text.len());
        output.push_str(&text[..offset]);
        output.push(next);
        output.extend(normalized);
        return Cow::Owned(output);
    }
    match original.next() {
        Some((offset, _)) => Cow::Owned(text[..offset].to_owned()),
        None => Cow::Borrowed(text),
    }
}

/// One pass over words, allocating only at the first noncanonical separator.
/// Words borrow this same text, so their start addresses establish position
/// without comparing the already-sliced contents again.
fn collapse_whitespace(text: &str) -> Cow<'_, str> {
    let mut output: Option<String> = None;
    let mut offset = 0;
    for word in text.split_whitespace() {
        let separator = if offset == 0 { "" } else { " " };
        if output.is_none()
            && !text[offset..]
                .strip_prefix(separator)
                .is_some_and(|tail| tail.as_ptr() == word.as_ptr())
        {
            let mut owned = String::with_capacity(text.len());
            owned.push_str(&text[..offset]);
            output = Some(owned);
        }
        if let Some(owned) = output.as_mut() {
            owned.push_str(separator);
            owned.push_str(word);
        }
        offset += separator.len() + word.len();
    }
    match output {
        Some(owned) => Cow::Owned(owned),
        None if offset != text.len() => Cow::Owned(text[..offset].to_owned()),
        None => Cow::Borrowed(text),
    }
}

#[cfg(test)]
mod tests;
