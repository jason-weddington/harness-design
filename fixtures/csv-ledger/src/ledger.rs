//! Ledger: orders and holds a list of [`Transaction`]s.
//!
//! This module is correct — no bug lives here.  It sorts the raw parsed
//! transactions into chronological order so downstream callers receive a
//! deterministic sequence regardless of the order rows appear in the CSV.

use crate::parser::Transaction;

/// Accept a list of parsed transactions and return them sorted by month.
///
/// The sort is stable and lexicographic on the `YYYY-MM` month string, which
/// is equivalent to chronological order.  All arithmetic remains in integer
/// cents throughout.
pub fn build_ledger(mut transactions: Vec<Transaction>) -> Vec<Transaction> {
    transactions.sort_by(|a, b| a.month.cmp(&b.month));
    transactions
}
