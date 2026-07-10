use crate::parser::Expr;

/// Recursively evaluate an [`Expr`] to an `f64`.
pub fn eval(expr: &Expr) -> Result<f64, String> {
    match expr {
        Expr::Num(n) => Ok(*n),
        Expr::Neg(e) => Ok(-eval(e)?),
        Expr::Add(l, r) => Ok(eval(l)? + eval(r)?),
        Expr::Sub(l, r) => Ok(eval(l)? - eval(r)?),
        Expr::Mul(l, r) => Ok(eval(l)? * eval(r)?),
        Expr::Div(l, r) => {
            let num = eval(l)?;
            let den = eval(r)?;
            if den == 0.0 {
                Err("division by zero".to_string())
            } else {
                Ok(num / den)
            }
        }
    }
}
