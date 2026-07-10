// Holdout integration tests for the calc fixture.
//
// These tests are NOT committed alongside the fixture source — they are copied
// into <workspace>/tests/holdout.rs by the eval harness after the agent-under-
// eval has applied its fix. They use only the public `eval_str` entry point.
//
// All five cases exercise the ^ operator semantics that the committed fixture
// does NOT implement, acting as the sealed acceptance gate.

use calc::eval_str;

#[test]
fn right_associativity() {
    // 2^(3^2) = 2^9 = 512, not (2^3)^2 = 8^2 = 64
    assert_eq!(eval_str("2^3^2"), Ok(512.0));
}

#[test]
fn explicit_left_grouping() {
    // Parentheses override right-associativity: (2^3)^2 = 8^2 = 64
    assert_eq!(eval_str("(2^3)^2"), Ok(64.0));
}

#[test]
fn unary_minus_applies_after_power() {
    // -(2^2) = -4, not (-2)^2 = 4
    assert_eq!(eval_str("-2^2"), Ok(-4.0));
}

#[test]
fn negative_exponent() {
    // 2^(-1) = 0.5
    assert_eq!(eval_str("2^-1"), Ok(0.5));
}

#[test]
fn power_binds_tighter_than_multiply() {
    // 2 * (3^2) = 2 * 9 = 18
    assert_eq!(eval_str("2*3^2"), Ok(18.0));
}
