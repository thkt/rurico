//! `CandidateArtifacts::<K>::from_paths` is `pub(crate)` in `src/artifacts.rs`,
//! so this external-crate caller MUST NOT compile. Both
//! `embed::CandidateArtifacts` and `reranker::CandidateArtifacts` are type
//! aliases for `artifacts::CandidateArtifacts<K>`, so a single fixture against
//! the canonical definition site covers both downstream-visible names.

use std::path::PathBuf;

fn main() {
    let _ = rurico::artifacts::CandidateArtifacts::<rurico::artifacts::EmbedKind>::from_paths(
        PathBuf::new(),
        PathBuf::new(),
        PathBuf::new(),
    );
}
