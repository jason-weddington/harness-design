//! CSV parser for expense-ledger records.
//!
//! Each line has three comma-separated fields:
//!
//! ```text
//! YYYY-MM-DD,kind,amount
//! ```
//!
//! * `kind` is either `credit` (money received) or `debit` (money paid out).
//! * `amount` is a decimal dollar value with up to two decimal places,
//!   e.g. `12.50`.  It is converted to integer cents (multiplied by 100)
//!   and stored as an `i64`.
//!
//! # Sign convention
//!
//! Credits are **positive** (money coming in); debits are **negative** (money
//! going out).  The `amount_cents` field on [`Transaction`] carries this sign.

/// A single parsed expense-ledger entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    /// Calendar month in `YYYY-MM` format (the day portion is stripped).
    pub month: String,
    /// Signed amount in cents.  Positive → credit; negative → debit.
    pub amount_cents: i64,
}

/// Parse a block of CSV text into a `Vec<Transaction>`.
///
/// Lines that are empty or start with `#` are skipped (comment/blank lines).
/// Returns an error string describing the first unparseable line.
pub fn parse_csv(csv: &str) -> Result<Vec<Transaction>, String> {
    let mut txns = Vec::new();

    for (lineno, raw) in csv.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.splitn(3, ',').collect();
        if fields.len() != 3 {
            return Err(format!(
                "line {}: expected 3 comma-separated fields, got {}",
                lineno + 1,
                fields.len()
            ));
        }

        let date = fields[0].trim();
        let kind = fields[1].trim();
        let amount_str = fields[2].trim();

        let month = date
            .get(..7)
            .ok_or_else(|| format!("line {}: date '{}' too short for YYYY-MM-DD", lineno + 1, date))?
            .to_string();

        let amount_cents = parse_cents(amount_str)
            .ok_or_else(|| format!("line {}: cannot parse amount '{}'", lineno + 1, amount_str))?;

        // Sign convention: credits are positive, debits are negative.
        let signed = match kind {
            "credit" => -amount_cents,
            "debit" => -amount_cents,
            _ => {
                return Err(format!(
                    "line {}: unknown transaction kind '{}' (expected credit or debit)",
                    lineno + 1,
                    kind
                ));
            }
        };

        txns.push(Transaction { month, amount_cents: signed });
    }

    Ok(txns)
}

/// Parse a non-negative decimal dollar string into integer cents.
///
/// Accepts values with zero, one, or two fractional digits.
/// Returns `None` if the string is not a valid non-negative dollar amount.
fn parse_cents(s: &str) -> Option<i64> {
    match s.split_once('.') {
        None => {
            // Whole dollars only.
            let d: i64 = s.parse().ok()?;
            if d < 0 {
                return None;
            }
            Some(d * 100)
        }
        Some((dollars, frac)) => {
            let d: i64 = dollars.parse().ok()?;
            if d < 0 {
                return None;
            }
            // Normalise fractional part to exactly two digits.
            let cents: i64 = match frac.len() {
                1 => {
                    let x: i64 = frac.parse().ok()?;
                    x * 10
                }
                2 => frac.parse().ok()?,
                _ => return None,
            };
            Some(d * 100 + cents)
        }
    }
}
