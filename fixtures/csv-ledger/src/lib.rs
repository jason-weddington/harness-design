//! csv-ledger: a minimal expense tracker used as a coding-eval fixture.
//!
//! Pipeline: [`parse_csv`] → [`build_ledger`] → [`monthly_totals`].
//!
//! All amounts are integer cents (`i64`); no floating-point arithmetic is
//! used anywhere in the parse/ledger/report pipeline.

pub mod currency;
pub mod ledger;
pub mod parser;
pub mod report;

pub use ledger::build_ledger;
pub use parser::{Transaction, parse_csv};
pub use report::monthly_totals;
