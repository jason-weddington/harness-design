//! A tiny Pratt-parser calculator — a coding-task eval fixture.
//!
//! This crate is committed in a FAILING state on purpose: the `power_basic`
//! test encodes the intended contract and keeps the suite red until the
//! `^` operator is added across all three source files. The workspace
//! `exclude`s `fixtures/*`, so the project's own gates never build or lint
//! this crate.

mod eval;
mod lexer;
mod parser;

/// Evaluate a calculator expression string to an `f64`.
///
/// Supports integer literals (as `f64`), `+`, `-`, `*`, `/`, parentheses,
/// and unary minus. Returns `Err` on unknown tokens, parse errors, or
/// division by zero.
pub fn eval_str(input: &str) -> Result<f64, String> {
    let tokens = lexer::tokenize(input)?;
    let ast = parser::parse(&tokens)?;
    eval::eval(&ast)
}

#[cfg(test)]
mod tests {
    use super::eval_str;

    #[test]
    fn addition_with_multiplication_precedence() {
        assert_eq!(eval_str("1+2*3"), Ok(7.0));
    }

    #[test]
    fn parentheses_override_precedence() {
        assert_eq!(eval_str("(1+2)*3"), Ok(9.0));
    }

    #[test]
    fn unary_minus_negates_grouped_expr() {
        assert_eq!(eval_str("-(3+4)"), Ok(-7.0));
    }

    #[test]
    fn division_produces_float() {
        assert_eq!(eval_str("6/2"), Ok(3.0));
    }

    /// This test is the committed failing case.
    ///
    /// The lexer does not yet recognise `^`, so `eval_str("2^3")` returns
    /// `Err(...)` instead of `Ok(8.0)`. The fix requires adding a `Caret`
    /// token in `src/lexer.rs`, a right-associative binding-power entry in
    /// `src/parser.rs`, and a `Pow` evaluation arm in `src/eval.rs`.
    #[test]
    fn power_basic() {
        assert_eq!(eval_str("2^3"), Ok(8.0));
    }
}
