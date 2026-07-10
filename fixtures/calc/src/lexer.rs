/// Tokens produced by the calculator lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

/// Tokenize `input` into a list of [`Token`]s.
///
/// Integer literal sequences are parsed as `f64`. Whitespace is skipped.
/// Any unrecognised character returns an error.
pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' => {
                chars.next();
            }
            '0'..='9' => {
                let mut s = String::new();
                while matches!(chars.peek(), Some(&d) if d.is_ascii_digit()) {
                    s.push(chars.next().unwrap());
                }
                let val: f64 = s.parse().map_err(|e| format!("number parse error: {e}"))?;
                tokens.push(Token::Number(val));
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            _ => return Err(format!("unknown token: {ch:?}")),
        }
    }

    Ok(tokens)
}
