use super::*;

#[test]
fn w1_has_two_long_texts() {
    let w = workload_w1();
    assert_eq!(w.len(), 2);
    assert!(w[0].contains("apple pie"));
    assert!(w[1].contains("Spain"));
    assert!(
        w[0].len() > 40000,
        "w1[0] should be long, got {}",
        w[0].len()
    );
    assert!(
        w[1].len() > 20000,
        "w1[1] should be long, got {}",
        w[1].len()
    );
}

#[test]
fn w2_has_hundred_short_texts() {
    let w = workload_w2();
    assert_eq!(w.len(), 100);
    assert!(
        w.iter().all(|t| t.len() < 80),
        "all W2 texts should be short"
    );
    assert!(w[0].contains("number 0"));
    assert!(w[99].contains("number 99"));
}

#[test]
fn w3_alternates_long_and_short() {
    let w = workload_w3();
    assert_eq!(w.len(), 10);
    for (i, text) in w.iter().enumerate() {
        if i.is_multiple_of(2) {
            assert!(
                text.len() > 3000,
                "w3[{i}] should be long, got {}",
                text.len()
            );
        } else {
            assert!(
                text.len() < 50,
                "w3[{i}] should be short, got {}",
                text.len()
            );
        }
    }
}
