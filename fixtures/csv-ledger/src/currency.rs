//! Currency display helpers — converts integer cents to a human-readable
//! dollar string.

/// Format a signed cent amount as a dollar string.
///
/// Examples: `1234` → `"$12.34"`, `-550` → `"-$5.50"`, `0` → `"$0.00"`.
///
/// Note: the dollar and cent portions are derived by plain integer division
/// (`/ 100`) and remainder (`% 100`) on the absolute value — both operations
/// truncate toward zero.  For well-formed ledger entries (whole-cent amounts)
/// this is exact; there is no rounding, no half-up, and no floating-point
/// involved.  The deliberate choice to operate on the absolute value before
/// splitting means `-199` displays as `"-$1.99"` rather than `"-$1.-99"`,
/// which would be produced by naively dividing the signed value.
pub fn format_cents(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let abs = cents.unsigned_abs();
    let dollars = abs / 100;
    let remainder = abs % 100;
    format!("{sign}${dollars}.{remainder:02}")
}

#[cfg(test)]
mod tests {
    use super::format_cents;

    #[test]
    fn positive_amount() {
        assert_eq!(format_cents(1234), "$12.34");
    }

    #[test]
    fn negative_amount() {
        assert_eq!(format_cents(-550), "-$5.50");
    }

    #[test]
    fn zero() {
        assert_eq!(format_cents(0), "$0.00");
    }
}
