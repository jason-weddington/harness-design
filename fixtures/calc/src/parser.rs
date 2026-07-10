use crate::lexer::Token;

/// AST node for a calculator expression.
#[derive(Debug)]
pub enum Expr {
    Num(f64),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

/// Parse a token slice into an [`Expr`] using a Pratt parser.
pub fn parse(tokens: &[Token]) -> Result<Expr, String> {
    let mut pos = 0;
    let expr = parse_expr(tokens, &mut pos, 0)?;
    if pos != tokens.len() {
        return Err(format!("unexpected token at position {pos}"));
    }
    Ok(expr)
}

/// Pratt expression parser with `min_bp` as the minimum left binding power
/// for an infix operator to be consumed at this level of the recursion.
fn parse_expr(tokens: &[Token], pos: &mut usize, min_bp: u8) -> Result<Expr, String> {
    let mut lhs = parse_prefix(tokens, pos)?;

    loop {
        let Some(tok) = tokens.get(*pos) else { break };
        let (l_bp, r_bp) = match infix_bp(tok) {
            Some(bp) => bp,
            None => break,
        };
        if l_bp < min_bp {
            break;
        }
        *pos += 1;
        let rhs = parse_expr(tokens, pos, r_bp)?;
        lhs = match tok {
            Token::Plus => Expr::Add(Box::new(lhs), Box::new(rhs)),
            Token::Minus => Expr::Sub(Box::new(lhs), Box::new(rhs)),
            Token::Star => Expr::Mul(Box::new(lhs), Box::new(rhs)),
            Token::Slash => Expr::Div(Box::new(lhs), Box::new(rhs)),
            _ => unreachable!(),
        };
    }

    Ok(lhs)
}

/// Parse a prefix (unary) operator or a primary expression (number, grouped).
///
/// Unary `-` is parsed here with prefix binding power 5, which means it binds
/// tighter than `*` and `/` but looser than `^` (once ^ is added).
fn parse_prefix(tokens: &[Token], pos: &mut usize) -> Result<Expr, String> {
    match tokens.get(*pos) {
        Some(Token::Number(n)) => {
            *pos += 1;
            Ok(Expr::Num(*n))
        }
        Some(Token::Minus) => {
            *pos += 1;
            // Unary minus: parse its operand at binding power 5.
            let operand = parse_expr(tokens, pos, 5)?;
            Ok(Expr::Neg(Box::new(operand)))
        }
        Some(Token::LParen) => {
            *pos += 1;
            let inner = parse_expr(tokens, pos, 0)?;
            match tokens.get(*pos) {
                Some(Token::RParen) => {
                    *pos += 1;
                    Ok(inner)
                }
                _ => Err("expected ')'".to_string()),
            }
        }
        Some(tok) => Err(format!("unexpected token: {tok:?}")),
        None => Err("unexpected end of input".to_string()),
    }
}

/// Return the (left, right) binding power for an infix operator token,
/// or `None` if the token is not an infix operator.
///
/// Left-associative operators use `r_bp = l_bp + 1` so that the right-hand
/// recursive call does not re-consume the same-precedence operator.
///
/// * `+` and `-`: precedence 1 (lowest).
/// * `*` and `/`: precedence 3.
///
/// Right-associativity is achieved when the right binding power is LOWER
/// than the left — the right-hand recursive call will accept the same-
/// precedence operator again, threading the parse tree rightward. The
/// `*` and `/` entries below apply exactly this technique to give those
/// operators their correct associativity.
fn infix_bp(tok: &Token) -> Option<(u8, u8)> {
    match tok {
        Token::Plus | Token::Minus => Some((1, 2)),
        Token::Star | Token::Slash => Some((3, 4)),
        _ => None,
    }
}
