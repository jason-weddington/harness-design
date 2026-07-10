//! Monthly-totals report over a ledger of transactions.
//!
//! Aggregates parsed transactions into per-month net totals for display.

use crate::parser::Transaction;
use std::collections::BTreeMap;

/// Compute the net total (in cents) for each calendar month.
///
/// Returns a [`BTreeMap`] keyed by `YYYY-MM` month strings so the result is
/// ordered deterministically.  A positive value means net credit (more money
/// received than paid out); a negative value means net debit.
pub fn monthly_totals(transactions: &[Transaction]) -> BTreeMap<String, i64> {
    let mut totals: BTreeMap<String, i64> = BTreeMap::new();
    for t in transactions {
        *totals.entry(t.month.clone()).or_insert(0) += t.amount_cents;
    }
    totals
}

#[cfg(test)]
mod tests {
    use crate::{build_ledger, monthly_totals, parse_csv};

    #[test]
    fn monthly_net_totals_are_correct() {
        // January 2024: two credit transactions only.
        // credit $50.00 + credit $30.00 = +$80.00 = +8000 cents net.
        let csv = "\
2024-01-03,credit,50.00
2024-01-17,credit,30.00
";
        let txns = parse_csv(csv).expect("CSV must parse without error");
        let ledger = build_ledger(txns);
        let totals = monthly_totals(&ledger);
        assert_eq!(
            totals.get("2024-01").copied(),
            Some(8000),
            "January net (credits only) should be +8000 cents"
        );
    }
}
