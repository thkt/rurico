//! Standalone observation only; no GPU, model or replacement planner.
mod baseline;
mod current;
use current::model_io;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Allocator;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
// Unsafe observation is isolated from product code and its Cargo lint policy.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(p, layout, size) };
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            allocated(size);
        }
        new
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn cpu_us() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    unsafe {
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
        let usage = usage.assume_init();
        (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1_000_000
            + i64::from(usage.ru_utime.tv_usec)
            + i64::from(usage.ru_stime.tv_usec)
    }
}
fn measure(case: &str, variant: &str, repeat: usize, iterations: usize, mut action: impl FnMut()) {
    for _ in 0..5 {
        action();
    }
    let live = LIVE.load(Ordering::Relaxed);
    let calls = CALLS.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    action();
    let allocations = CALLS.load(Ordering::Relaxed) - calls;
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(live);
    let cpu = cpu_us();
    let start = Instant::now();
    for _ in 0..iterations {
        action();
    }
    let elapsed = start.elapsed().as_nanos() as f64 / iterations as f64;
    let cpu_ns = (cpu_us() - cpu) as f64 * 1000.0 / iterations as f64;
    println!(
        "{}",
        serde_json::json!({"case":case,"variant":variant,"repeat":repeat,
        "iterations":iterations,"wall_ns_per_call":elapsed,"cpu_ns_per_call":cpu_ns,
        "rust_allocations_reallocations":allocations,"rust_peak_extra_requested_bytes":peak})
    );
}
fn tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
        processors::template::TemplateProcessing,
    };
    let vocab = [
        ("[UNK]".into(), 0),
        ("[BOS]".into(), 1),
        ("[EOS]".into(), 2),
        ("検索文書:".into(), 3),
        ("語".into(), 4),
    ]
    .into_iter()
    .collect();
    let mut t = tokenizers::Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    t.with_pre_tokenizer(Some(WhitespaceSplit));
    t.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[BOS] $A [EOS]")
            .unwrap()
            .special_tokens(vec![("[BOS]", 1), ("[EOS]", 2)])
            .build()
            .unwrap(),
    ));
    t
}
fn main() {
    let t = std::env::args().nth(1).map_or_else(tokenizer, |path| {
        tokenizers::Tokenizer::from_file(path).unwrap()
    });
    let prefix = t.encode("検索文書: ", false).unwrap().get_ids().to_vec();
    let budget = 8192 - 2 - prefix.len();
    let short = "語 ".repeat(32);
    let mut long = "語 ".repeat(9000);
    while t.encode(long.as_str(), false).unwrap().len() <= 2 * budget {
        long.push_str(&"語 ".repeat(9000));
    }
    assert_eq!(
        baseline::embed::plan(&t, &short, budget, &prefix),
        current::embed::plan(&t, &short, budget, &prefix)
    );
    let expected = baseline::embed::plan(&t, &long, budget, &prefix);
    assert!(
        expected.1[0] >= 3,
        "must exercise real long-document splitting"
    );
    assert_eq!(expected, current::embed::plan(&t, &long, budget, &prefix));
    let offsets = t.encode(long.as_str(), false).unwrap();
    let shrink_end = (budget + 8).min(offsets.len());
    let expected_shrink = baseline::embed::shrink(&t, &long, offsets.get_offsets(), shrink_end);
    assert!(
        expected_shrink.1 < shrink_end,
        "must exercise rejection/shrink"
    );
    assert_eq!(
        expected_shrink,
        current::embed::shrink(&t, &long, offsets.get_offsets(), shrink_end)
    );
    let tokens: Vec<_> = (0..128).map(|n| vec![n; 130 + n as usize % 300]).collect();
    let old_pad = baseline::embed::pad_input(tokens.clone());
    let new_pad = current::embed::pad_input(tokens);
    assert_eq!(
        baseline::embed::pad(&old_pad),
        current::embed::pad(&new_pad)
    );
    let route_tokens: Vec<_> = (0..1024)
        .map(|i| vec![i; [30, 300, 1000, 3000][i as usize % 4]])
        .collect();
    assert_eq!(
        current::embed::route(route_tokens.clone(), true),
        current::embed::route(route_tokens.clone(), false)
    );
    for repeat in 0..7 {
        for version in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
            let variant = if version == 0 { "baseline" } else { "current" };
            measure("padding_clone", variant, repeat, 300, || {
                black_box(if version == 0 {
                    baseline::embed::pad(&old_pad)
                } else {
                    current::embed::pad(&new_pad)
                });
            });
            measure(
                "redundant_sort",
                if version == 0 {
                    "with_sort"
                } else {
                    "without_sort"
                },
                repeat,
                20,
                || {
                    black_box(current::embed::route(route_tokens.clone(), version == 0));
                },
            );
            for (case, text, iterations) in [
                ("short_plan", short.as_str(), 100),
                ("long_plan", long.as_str(), 5),
            ] {
                measure(case, variant, repeat, iterations, || {
                    black_box(if version == 0 {
                        baseline::embed::plan(&t, text, budget, &prefix)
                    } else {
                        current::embed::plan(&t, text, budget, &prefix)
                    });
                });
            }
            measure("shrink", variant, repeat, 5, || {
                black_box(if version == 0 {
                    baseline::embed::shrink(&t, &long, offsets.get_offsets(), shrink_end)
                } else {
                    current::embed::shrink(&t, &long, offsets.get_offsets(), shrink_end)
                });
            });
        }
    }
}
