//! The personal-kb **lexical matcher**: a port of the personal-kb hook's
//! `cues_lite.bash_segments` and `prevention._find_match` / `_matches`, as of
//! `personal_kb` `3bd9131`.
//!
//! A bash command is split into segments, each segment is reduced to a
//! command **class** (`git push`, `systemctl restart`, `ls`, …) plus its
//! non-flag **args**, and an index cue matches a segment when its
//! `target_class` equals the class and its `args_prefix` words prefix the
//! args.
//!
//! **Quotes are not interpreted.** Separators inside quotes still split, and
//! quote characters stay inside tokens — so a matching command written inside
//! a quoted string or a heredoc still matches (`git commit -m "a; git push
//! github"` yields a `git push` segment). That is the hook's behaviour, ported
//! on purpose: the two matchers must agree.
//!
//! **Drift guard.** The table tests below (`bash_segments_vectors` and
//! `find_match_vectors`) were produced by running the Python originals. On any
//! change to either side, regenerate them from the Python original rather
//! than editing expectations by hand.

/// Programs whose first non-flag argument is part of the command class
/// (`git push`, not `git`).
const TWO_WORD_PROGRAMS: [&str; 20] = [
    "git",
    "uv",
    "npm",
    "pnpm",
    "yarn",
    "docker",
    "kubectl",
    "aws",
    "gh",
    "cargo",
    "sam",
    "cdk",
    "terraform",
    "agent-gtd",
    "systemctl",
    "pip",
    "python",
    "python3",
    "pytest",
    "make",
];

/// Split `command` on `&&`, `||`, `;`, `|` and newline, left to right (the
/// two-character separators are tried first at each position). A lone `&` is
/// not a separator.
fn split_segments(command: &str) -> Vec<&str> {
    split_on(command, true)
}

/// [`split_segments`], with newline a separator only when `newline` is set.
fn split_on(command: &str, newline: bool) -> Vec<&str> {
    let bytes = command.as_bytes();
    let mut segments = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let rest = &bytes[i..];
        let sep = if rest.starts_with(b"&&") || rest.starts_with(b"||") {
            2
        } else {
            usize::from(matches!(bytes[i], b';' | b'|') || (newline && bytes[i] == b'\n'))
        };
        if sep == 0 {
            i += 1;
        } else {
            // The separators are ASCII, so these byte indices are char
            // boundaries.
            segments.push(&command[start..i]);
            i += sep;
            start = i;
        }
    }
    segments.push(&command[start..]);
    segments
}

/// `NAME=value` with `NAME` matching `[A-Za-z_][A-Za-z0-9_]*`.
fn is_env_assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The last `/`-separated component that is neither empty nor `.`, or the
/// empty string when there is none.
fn basename(token: &str) -> &str {
    token
        .split('/')
        .rev()
        .find(|c| !c.is_empty() && *c != ".")
        .unwrap_or("")
}

/// One segment's (class, args), or `None` when the segment is skipped.
fn segment_cue(segment: &str) -> Option<(String, Vec<String>)> {
    let raw: Vec<&str> = segment.split_whitespace().collect();
    if raw.first().is_none_or(|t| *t == "cd") {
        return None;
    }
    let mut tokens: &[&str] = &raw;
    while tokens.first().is_some_and(|t| is_env_assignment(t)) {
        tokens = &tokens[1..];
    }
    if tokens.first() == Some(&"sudo") {
        tokens = &tokens[1..];
        while let Some(flag) = tokens.first().filter(|t| t.starts_with('-')) {
            tokens = &tokens[1..];
            if (*flag == "-u" || *flag == "-g") && !tokens.is_empty() {
                tokens = &tokens[1..];
            }
        }
    }
    let first = tokens.first()?;
    let word = basename(first);
    let two_word =
        TWO_WORD_PROGRAMS.contains(&word) && tokens.len() > 1 && !tokens[1].starts_with('-');
    let (class, rest) = if two_word {
        (format!("{word} {}", tokens[1]), &tokens[2..])
    } else {
        (word.to_string(), &tokens[1..])
    };
    if class.is_empty() {
        return None;
    }
    let args = rest
        .iter()
        .filter(|t| !t.starts_with('-'))
        .map(|t| (*t).to_string())
        .collect();
    Some((class, args))
}

/// Split a bash command into `(class, args)` segments, in order. A port of
/// the hook's `cues_lite.bash_segments`; see the module doc for the rules'
/// provenance.
///
/// ```
/// use harness::personal_kb::cues::bash_segments;
/// assert_eq!(
///     bash_segments("cd /a && git push github main"),
///     vec![("git push".to_string(), vec!["github".to_string(), "main".to_string()])]
/// );
/// ```
#[must_use]
pub fn bash_segments(command: &str) -> Vec<(String, Vec<String>)> {
    split_segments(command)
        .into_iter()
        .filter_map(segment_cue)
        .collect()
}

/// The canonical Claude Code tool name the Bash matcher branch serves.
const CANONICAL_BASH: &str = "Bash";

/// Canonical tools whose target is a file path and whose class is its
/// extension.
const FILE_TOOLS: [&str; 5] = ["Read", "Edit", "Write", "MultiEdit", "NotebookEdit"];

/// One cue from the KB's prevention index (canonical tool names); every field
/// is trimmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCue {
    /// The KB resolution entry the cue comes from.
    pub resolution_id: String,
    /// The canonical Claude Code tool the cue fires on (`Bash`, `Edit`, …).
    pub tool: String,
    /// The class the cue fires on (`git push`, `ext:toml`).
    pub target_class: String,
    /// Whitespace-separated args that must prefix the candidate's args; blank
    /// matches any args.
    pub args_prefix: String,
    /// The earlier wrong belief the resolution replaced (may be empty).
    pub wrong_belief: String,
    /// The corrected fact.
    pub corrected_fact: String,
    /// The KB's provenance label (may be empty).
    pub provenance_label: String,
}

fn prefix_matches(cue: &IndexCue, args: &[String]) -> bool {
    let wanted: Vec<&str> = cue.args_prefix.split_whitespace().collect();
    args.len() >= wanted.len() && args.iter().zip(&wanted).all(|(a, w)| a == w)
}

/// The class of a non-Bash `target` for canonical `tool`: `ext:<suffix>` for
/// file tools (`ext:none` when the path has no extension), `""` for an empty
/// target or any other tool. A port of the hook's `cues_lite.target_class`.
#[must_use]
pub fn target_class(tool: &str, target: &str) -> String {
    if !FILE_TOOLS.contains(&tool) || target.is_empty() {
        return String::new();
    }
    let name = target.rsplit('/').next().unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => format!("ext:{}", name[i + 1..].to_lowercase()),
        _ => "ext:none".to_string(),
    }
}

/// The args of a non-Bash target after its class tokens. A port of the hook's
/// `_bash_tokens` + `bash_args_after_class`: splits on `&&`, `||`, `;` and `|`
/// only (not newline), takes the first piece whose first token is not `cd`.
#[must_use]
pub fn bash_args_after_class(s: &str) -> Vec<String> {
    let piece = split_on(s, false)
        .into_iter()
        .find(|p| p.split_whitespace().next().is_some_and(|t| t != "cd"));
    let Some(piece) = piece else {
        return Vec::new();
    };
    let mut tokens: &[&str] = &piece.split_whitespace().collect::<Vec<_>>();
    while tokens.first().is_some_and(|t| is_env_assignment(t)) {
        tokens = &tokens[1..];
    }
    if tokens.first() == Some(&"sudo") {
        tokens = &tokens[1..];
        while let Some(flag) = tokens.first().filter(|t| t.starts_with('-')) {
            tokens = &tokens[1..];
            if (*flag == "-u" || *flag == "-g") && !tokens.is_empty() {
                tokens = &tokens[1..];
            }
        }
    }
    let class_len = match tokens {
        [first, second, ..]
            if TWO_WORD_PROGRAMS.contains(&basename(first)) && !second.starts_with('-') =>
        {
            2
        }
        [] => 0,
        _ => 1,
    };
    tokens[class_len..]
        .iter()
        .filter(|t| !t.starts_with('-'))
        .map(|t| (*t).to_string())
        .collect()
}

/// The first cue matching a call to canonical `tool` with raw `target`. For
/// `Bash`, segments are tried in order, and for each the first cue (in slice
/// order) whose tool and class equal and whose args prefix matches wins;
/// segment order beats cue order. Otherwise the single candidate is
/// (`target_class`, `bash_args_after_class`), with none when the class is
/// empty. A port of the hook's `prevention._find_match`.
#[must_use]
pub fn find_match<'a>(tool: &str, target: &str, cues: &'a [IndexCue]) -> Option<&'a IndexCue> {
    let candidates: Vec<(String, Vec<String>)> = if tool == CANONICAL_BASH {
        bash_segments(target)
    } else {
        let class = target_class(tool, target);
        if class.is_empty() {
            Vec::new()
        } else {
            vec![(class, bash_args_after_class(target))]
        }
    };
    candidates.into_iter().find_map(|(class, args)| {
        cues.iter()
            .find(|cue| cue.tool == tool && cue.target_class == class && prefix_matches(cue, &args))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        IndexCue, TWO_WORD_PROGRAMS, bash_args_after_class, bash_segments, find_match, target_class,
    };

    /// `CLASS[args]` rendering of one segment, matching the vector notation.
    fn render(command: &str) -> Vec<String> {
        bash_segments(command)
            .into_iter()
            .map(|(class, args)| format!("{class}[{}]", args.join(" ")))
            .collect()
    }

    #[test]
    fn two_word_programs_are_pinned() {
        assert_eq!(TWO_WORD_PROGRAMS.len(), 20);
        assert!(TWO_WORD_PROGRAMS.contains(&"agent-gtd"));
    }

    /// Every row produced by the Python original at `personal_kb` `3bd9131`.
    const SEGMENT_VECTORS: &[(&str, &[&str])] = &[
        (
            "cd /a && FOO=1 git push -f github main; ls | wc -l",
            &["git push[github main]", "ls[]", "wc[]"],
        ),
        ("cd /a", &[]),
        (
            "git remote add origin git@h:r",
            &["git remote[add origin git@h:r]"],
        ),
        ("git remote -v", &["git remote[]"]),
        (
            "git commit -qam x && git push github main",
            &["git commit[x]", "git push[github main]"],
        ),
        (
            "cd /app\nmake smoke 2>&1 | tail -5",
            &["make smoke[2>&1]", "tail[]"],
        ),
        (
            "sudo systemctl restart caddy; echo ok",
            &["systemctl restart[caddy]", "echo[ok]"],
        ),
        ("sudo -u dispatch git -C /x status", &["git[/x status]"]),
        ("/usr/bin/git status", &["git status[]"]),
        ("cargo test --workspace", &["cargo test[]"]),
        ("cargo --version", &["cargo[]"]),
        (
            "RUST_LOG=debug cargo nextest run -p harness",
            &["cargo nextest[run harness]"],
        ),
        (
            "uv run pytest -q tests/x.py",
            &["uv run[pytest tests/x.py]"],
        ),
        ("python3 -m pytest", &["python3[pytest]"]),
        (
            "agent-gtd run-status abc --wait",
            &["agent-gtd run-status[abc]"],
        ),
        (
            "echo a || git push origin main",
            &["echo[a]", "git push[origin main]"],
        ),
        ("", &[]),
        ("   ", &[]),
        ("ls -la", &["ls[]"]),
        ("docker compose up -d", &["docker compose[up]"]),
        ("terraform -chdir=infra apply", &["terraform[apply]"]),
        ("kubectl get pods -n x", &["kubectl get[pods x]"]),
        ("a=1 b=2", &[]),
        ("sudo -g wheel -E aws s3 ls", &["aws s3[ls]"]),
        ("a|||git push x", &["a[]", "git push[x]"]),
        ("git push x &&& ls", &["git push[x]", "&[ls]"]),
        ("./git push", &["git push[]"]),
        ("sudo FOO=1 git push", &["FOO=1[git push]"]),
        ("FOO=1 cd /x", &["cd[/x]"]),
        ("git\tpush\torigin", &["git push[origin]"]),
        ("cd /a; cd /b", &[]),
        ("sudo", &[]),
        ("sudo -u", &[]),
        ("git", &["git[]"]),
        ("GIT push", &["GIT[push]"]),
        ("/ foo", &[]),
        ("/usr/bin/sudo git push", &["sudo[git push]"]),
        (
            "git push origin main\ncargo test",
            &["git push[origin main]", "cargo test[]"],
        ),
        ("A-B=1 git push", &["A-B=1[git push]"]),
        ("A=B=C git push", &["git push[]"]),
        ("_X=1 npm install -g x", &["npm install[x]"]),
        (
            "git commit -m \"a; git push github\"",
            &["git commit[\"a]", "git push[github\"]"],
        ),
        ("git/ push", &["git push[]"]),
        ("sudo -- git push", &["git push[]"]),
        (".. push", &["..[push]"]),
        (". push", &[]),
        ("git push origin main &", &["git push[origin main &]"]),
    ];

    #[test]
    fn bash_segments_vectors() {
        let cases = SEGMENT_VECTORS;
        assert_eq!(cases.len(), 47);
        for (i, (command, want)) in cases.iter().enumerate() {
            assert_eq!(render(command), *want, "vector {} {command:?}", i + 1);
        }
    }

    #[test]
    fn non_ascii_commands_do_not_panic() {
        assert_eq!(render("écho é && git push ü"), ["écho[é]", "git push[ü]"]);
    }

    fn cue(id: &str, target_class: &str, args_prefix: &str) -> IndexCue {
        IndexCue {
            resolution_id: id.to_string(),
            tool: "Bash".to_string(),
            target_class: target_class.to_string(),
            args_prefix: args_prefix.to_string(),
            wrong_belief: String::new(),
            corrected_fact: "fact".to_string(),
            provenance_label: String::new(),
        }
    }

    /// `(id, target_class, args_prefix)`.
    type CueSpec = (&'static str, &'static str, &'static str);

    /// Every row produced by the Python `_find_match` at `personal_kb` `3bd9131`.
    const MATCH_VECTORS: &[(&[CueSpec], &str, Option<&str>)] = &[
        (
            &[("kb-a", "git remote", "add")],
            "git remote add origin git@h:r",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "git remote", "add")],
            "cd /x && git remote add origin u",
            Some("kb-a"),
        ),
        (&[("kb-a", "git remote", "add")], "git remote -v", None),
        (
            &[("kb-a", "git remote", "add")],
            "git remote show origin",
            None,
        ),
        (
            &[("kb-a", "git remote", "")],
            "git remote add origin git@h:r",
            Some("kb-a"),
        ),
        (&[("kb-a", "git remote", "")], "git remote -v", Some("kb-a")),
        (
            &[("kb-a", "git remote", "")],
            "git remote show origin",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "systemctl restart", "")],
            "caddy validate --config /etc/caddy/Caddyfile && systemctl restart caddy",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "git push", "github")],
            "git commit -qam x && git push github main",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "git push", "github")],
            "git commit -qam x && git push origin main",
            None,
        ),
        (
            &[("kb-a", "make smoke", "")],
            "cd /app\nmake smoke 2>&1 | tail -5",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "systemctl restart", "")],
            "sudo systemctl restart caddy; echo ok",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "git push", "")],
            "git commit -qam x && echo hi",
            None,
        ),
        (
            &[("kb-a", "git commit", ""), ("kb-b", "git push", "")],
            "git commit -qam x && git push github main",
            Some("kb-a"),
        ),
        (
            &[("kb-b", "git commit", ""), ("kb-a", "git push", "")],
            "git push github main && git commit -qam x",
            Some("kb-a"),
        ),
        (
            &[
                ("kb-a", "git push", "origin main"),
                ("kb-b", "git push", ""),
            ],
            "git push origin main",
            Some("kb-a"),
        ),
        (
            &[
                ("kb-a", "git push", "origin main"),
                ("kb-b", "git push", ""),
            ],
            "git push origin",
            Some("kb-b"),
        ),
        (
            &[("kb-a", "git push", "   ")],
            "git push origin main",
            Some("kb-a"),
        ),
        (
            &[("kb-a", "git push", "origin")],
            "git push -f origin main",
            Some("kb-a"),
        ),
        (&[("kb-a", "git push", "")], "ls -la", None),
        (&[("kb-a", "git push", "")], "GIT push", None),
    ];

    #[test]
    fn find_match_vectors() {
        for (specs, command, want) in MATCH_VECTORS {
            let cues: Vec<IndexCue> = specs.iter().map(|(i, t, a)| cue(i, t, a)).collect();
            let got = find_match("Bash", command, &cues).map(|c| c.resolution_id.as_str());
            assert_eq!(got, *want, "{command:?}");
        }
    }

    #[test]
    fn find_match_on_empty_cues_is_none() {
        assert_eq!(find_match("Bash", "git push github main", &[]), None);
    }

    fn vectors() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../testdata/personal_kb_contracts/matcher_vectors.json"
        ))
        .unwrap()
    }

    #[test]
    fn target_class_vectors() {
        let v = vectors();
        let rows = v["target_class"].as_array().unwrap();
        assert!(!rows.is_empty());
        for r in rows {
            let (tool, target) = (r["tool"].as_str().unwrap(), r["target"].as_str().unwrap());
            assert_eq!(
                target_class(tool, target),
                r["expected"].as_str().unwrap(),
                "{tool} {target:?}"
            );
        }
    }

    #[test]
    fn bash_args_after_class_vectors() {
        let v = vectors();
        let rows = v["bash_args_after_class"].as_array().unwrap();
        assert!(!rows.is_empty());
        for r in rows {
            let target = r["target"].as_str().unwrap();
            let want: Vec<&str> = r["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            assert_eq!(bash_args_after_class(target), want, "{target:?}");
        }
    }

    #[test]
    fn bash_args_after_class_extra_shapes() {
        assert!(bash_args_after_class("cd /a").is_empty());
        assert_eq!(
            bash_args_after_class("cd /a && FOO=1 sudo -u x -E git push -f o m"),
            ["o", "m"]
        );
        assert_eq!(bash_args_after_class("ls\nfoo"), ["foo"]);
        assert!(bash_args_after_class("sudo").is_empty());
    }

    #[test]
    fn find_match_non_bash_vectors() {
        let v = vectors();
        let fm = &v["find_match"];
        let cues: Vec<IndexCue> = fm["index"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| IndexCue {
                resolution_id: c["resolution_id"].as_str().unwrap().to_string(),
                tool: c["tool"].as_str().unwrap().to_string(),
                target_class: c["target_class"].as_str().unwrap().to_string(),
                args_prefix: c["args_prefix"].as_str().unwrap().to_string(),
                wrong_belief: String::new(),
                corrected_fact: "f".to_string(),
                provenance_label: String::new(),
            })
            .collect();
        let cases = fm["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for c in cases {
            let (tool, target) = (c["tool"].as_str().unwrap(), c["target"].as_str().unwrap());
            let got = find_match(tool, target, &cues).map(|c| c.resolution_id.as_str());
            assert_eq!(
                got,
                c["expected_resolution_id"].as_str(),
                "{tool} {target:?}"
            );
        }
    }
}
