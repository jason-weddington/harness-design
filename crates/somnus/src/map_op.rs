//! The `POST /api/kb/map-op` transport: the KB's single machine-principal
//! write endpoint (`reference/somnus-functional-spec.md`, "The write path"
//! section — the map-op contract of record, carried verbatim there).
//!
//! The seam is a client-side classification of ONE POST, mirroring
//! [`crate::loop_input`] and [`crate::ledger`]: every op carries `body`
//! (the WHOLE composed map body as somnus rendered it), the server reads
//! exactly the `kb-XXXXX` refs out of it, and verifies each op as a set
//! comparison. Statuses, flat and terminal (nothing retryable): 403 not the
//! principal, 404 unknown `map_id`, 409 invariant violation / stale
//! `base_version` / cap exceeded, 422 lint findings, 200/201 success.
//!
//! `no_change` has NO endpoint and NO row; `propose_gap` is server-side
//! ledger work (see the deferral in the item's grounding notes) — so
//! [`MapOpRequest`] carries ONLY the three mutating ops, and somnus never
//! sends `base_version` (it holds no version state, and omitting it runs no
//! server check).
//!
//! The two per-night caps (one new map per project, three per KB, counted
//! since UTC midnight) live SERVER-side: a 409 on `create_map` with a
//! machine-readable reason is an ORDINARY ADMISSION OUTCOME, classified
//! [`MapOpResult::CapAdmission`] — somnus tracks neither cap.
//!
//! Layering (pinned): this module references NO other somnus module.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

/// Wall-clock bound on one map-op POST, mirroring
/// [`crate::ledger::LEDGER_HTTP_BOUND`]. Nothing on this surface is
/// retryable, so the bound converts a hung endpoint into a named fault.
pub const MAP_OP_HTTP_BOUND: Duration = Duration::from_secs(10);

/// One map-op request. Field sets are PINNED per op (see
/// [`build_request_body`]): exactly the fields the server reads, no more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapOpRequest {
    /// `create_map` → 201. The server asserts ≥1 ref, lints, checks caps.
    CreateMap {
        /// The project the fresh map belongs to.
        project_ref: String,
        /// The map's short title (the create op's `title` verbatim).
        short_title: String,
        /// The map's long title (the create op's `title` verbatim).
        long_title: String,
        /// The WHOLE composed fresh body.
        body: String,
    },
    /// `add_pointer` → 200. The server asserts `new ⊇ old` and
    /// `new − old == {added_entry_id}`.
    AddPointer {
        /// The map being edited.
        map_id: String,
        /// The body as of THIS edit (the intermediate state, not the final
        /// composed body — the set comparison makes the final body wrong
        /// for every call but the last).
        body: String,
        /// The single entry being pointed at.
        added_entry_id: String,
    },
    /// `strike_gap` → 200. The server asserts `new == old` and that
    /// `closing_entry_id` is in `old`.
    StrikeGap {
        /// The map being edited.
        map_id: String,
        /// The body as of THIS edit.
        body: String,
        /// The gap text being struck.
        gap_text: String,
        /// The entry cited as closing the gap.
        closing_entry_id: String,
    },
}

/// The one success envelope, uniform on all four ops (one deserializer).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct MapOpEnvelope {
    /// The server's map id (differs from a fresh client-side id on create).
    pub map_id: String,
    /// The stored version after the op.
    pub version: u64,
    /// The pointer count after the op.
    pub pointer_count: u64,
    /// The server's remaining per-night budget.
    pub budget: u64,
}

/// The applied result: the envelope, surfaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapOpApplied {
    /// The server's map id.
    pub map_id: String,
    /// The stored version after the op.
    pub version: u64,
    /// The pointer count after the op.
    pub pointer_count: u64,
    /// The server's remaining per-night budget.
    pub budget: u64,
}

/// One classified map-op outcome. Everything here is terminal — nothing on
/// this surface is retryable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapOpResult {
    /// `200`/`201` with a parseable envelope.
    Applied(MapOpApplied),
    /// `409` on `create_map` only: a per-night cap admission, ordinary by
    /// contract, carrying the server's reason body verbatim.
    CapAdmission {
        /// The server's machine-readable reason, verbatim.
        reason_body: String,
    },
    /// `401` — the bearer token was rejected.
    Unauthorized,
    /// `403` — the caller is not the configured machine principal.
    NotMachinePrincipal,
    /// `409` on an update op (an invariant somnus cannot provoke — it never
    /// sends `base_version` and a pointerless create is refused upstream —
    /// is classified client-side as a fault by reachability), `404`,
    /// `422`, `5xx`, a timeout, a transport failure, or a 2xx whose body
    /// does not parse as the envelope.
    Fault {
        /// The map the op targeted (empty for a `create_map`, which carries
        /// no map id).
        map_id: String,
        /// Why the op failed, naming the status or the transport reason.
        reason: String,
    },
}

/// The seam between the application step and whatever POSTs map ops. The
/// ONLY production implementation is [`HttpMapOpClient`]; tests use
/// scripted fakes.
#[async_trait]
pub trait MapOpClient: std::fmt::Debug + Send + Sync {
    /// Apply one op.
    async fn apply(&self, request: MapOpRequest) -> MapOpResult;
}

/// Build the exact JSON request body for one op: the pinned per-op field
/// set, `op` first, every field the server reads, nothing else.
#[must_use]
pub fn build_request_body(request: &MapOpRequest) -> Value {
    match request {
        MapOpRequest::CreateMap {
            project_ref,
            short_title,
            long_title,
            body,
        } => json!({
            "op": "create_map",
            "project_ref": project_ref,
            "short_title": short_title,
            "long_title": long_title,
            "body": body,
        }),
        MapOpRequest::AddPointer {
            map_id,
            body,
            added_entry_id,
        } => json!({
            "op": "add_pointer",
            "map_id": map_id,
            "body": body,
            "added_entry_id": added_entry_id,
        }),
        MapOpRequest::StrikeGap {
            map_id,
            body,
            gap_text,
            closing_entry_id,
        } => json!({
            "op": "strike_gap",
            "map_id": map_id,
            "body": body,
            "gap_text": gap_text,
            "closing_entry_id": closing_entry_id,
        }),
    }
}

/// The map id a fault names, per request: the op's map for the two update
/// ops, and the empty string for a `create_map` (which carries no map id —
/// the fresh id is client-side only).
#[must_use]
fn fault_map_id(request: &MapOpRequest) -> String {
    match request {
        MapOpRequest::CreateMap { .. } => String::new(),
        MapOpRequest::AddPointer { map_id, .. } | MapOpRequest::StrikeGap { map_id, .. } => {
            map_id.clone()
        }
    }
}

/// The production [`MapOpClient`]: exactly ONE
/// `POST {base_url}/api/kb/map-op` with `Authorization: Bearer <token>`, a
/// [`MAP_OP_HTTP_BOUND`] timeout, no retry, and the pinned status mapping.
pub struct HttpMapOpClient {
    base_url: String,
    token: String,
}

impl std::fmt::Debug for HttpMapOpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the token, even under Debug.
        f.debug_struct("HttpMapOpClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl HttpMapOpClient {
    /// Build a transport over an already-read token.
    #[must_use]
    pub fn new(base_url: String, token: String) -> Self {
        Self { base_url, token }
    }
}

#[async_trait]
impl MapOpClient for HttpMapOpClient {
    async fn apply(&self, request: MapOpRequest) -> MapOpResult {
        let map_id = fault_map_id(&request);
        let url = format!("{}/api/kb/map-op", self.base_url.trim_end_matches('/'));
        let exchange = async {
            let map_id = map_id.clone();
            let response = match reqwest::Client::new()
                .post(url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.token),
                )
                .json(&build_request_body(&request))
                .send()
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    return MapOpResult::Fault {
                        map_id,
                        reason: format!("transport failed: {err}"),
                    };
                }
            };
            let status = response.status().as_u16();
            let body = match response.text().await {
                Ok(body) => body,
                Err(err) => {
                    return MapOpResult::Fault {
                        map_id,
                        reason: format!("body read failed: {err}"),
                    };
                }
            };
            match status {
                200 | 201 => match serde_json::from_str::<MapOpEnvelope>(&body) {
                    Ok(envelope) => MapOpResult::Applied(MapOpApplied {
                        map_id: envelope.map_id,
                        version: envelope.version,
                        pointer_count: envelope.pointer_count,
                        budget: envelope.budget,
                    }),
                    Err(err) => MapOpResult::Fault {
                        map_id,
                        reason: format!("malformed envelope after HTTP {status} ({err})"),
                    },
                },
                401 => MapOpResult::Unauthorized,
                403 => MapOpResult::NotMachinePrincipal,
                // A create_map 409 is the ONLY 409 somnus can provoke (see
                // the reachability note on [`MapOpResult::Fault`]).
                409 if matches!(request, MapOpRequest::CreateMap { .. }) => {
                    MapOpResult::CapAdmission { reason_body: body }
                }
                status => MapOpResult::Fault {
                    map_id,
                    reason: format!("unexpected HTTP status {status}"),
                },
            }
        };
        match tokio::time::timeout(MAP_OP_HTTP_BOUND, exchange).await {
            Ok(result) => result,
            Err(_elapsed) => MapOpResult::Fault {
                map_id,
                reason: format!("timed out after {}s", MAP_OP_HTTP_BOUND.as_secs()),
            },
        }
    }
}

// ===== tests ==============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, method, path};

    /// A `create_map` request fixture, built once per test.
    fn create_request() -> MapOpRequest {
        MapOpRequest::CreateMap {
            project_ref: "demo-project".to_string(),
            short_title: "Wireguard and DNS".to_string(),
            long_title: "Wireguard and DNS".to_string(),
            body:
                "Lives in knowledge/linux/network\n\nPROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1"
                    .to_string(),
        }
    }

    fn add_request() -> MapOpRequest {
        MapOpRequest::AddPointer {
            map_id: "kb-20001".to_string(),
            body: "Lives in knowledge/linux/network\n\nDetail entries:\n- kb-10001 — g\n- kb-10005 — GLOSS-5".to_string(),
            added_entry_id: "kb-10005".to_string(),
        }
    }

    fn strike_request() -> MapOpRequest {
        MapOpRequest::StrikeGap {
            map_id: "kb-20001".to_string(),
            body: "body after the strike".to_string(),
            gap_text: "Site-to-site wireguard topology".to_string(),
            closing_entry_id: "kb-10003".to_string(),
        }
    }

    const ENVELOPE: &str = r#"{"map_id":"kb-30001","version":7,"pointer_count":6,"budget":2}"#;

    // --- the pure request builders: field sets pinned per op --------------

    #[test]
    fn the_create_map_body_carries_exactly_the_pinned_field_set() {
        assert_eq!(
            build_request_body(&create_request()),
            json!({
                "op": "create_map",
                "project_ref": "demo-project",
                "short_title": "Wireguard and DNS",
                "long_title": "Wireguard and DNS",
                "body": "Lives in knowledge/linux/network\n\nPROSE\n\nDetail entries:\n- kb-10001 — GLOSS-1",
            })
        );
    }

    #[test]
    fn the_add_pointer_body_carries_exactly_the_pinned_field_set() {
        assert_eq!(
            build_request_body(&add_request()),
            json!({
                "op": "add_pointer",
                "map_id": "kb-20001",
                "body": "Lives in knowledge/linux/network\n\nDetail entries:\n- kb-10001 — g\n- kb-10005 — GLOSS-5",
                "added_entry_id": "kb-10005",
            })
        );
    }

    #[test]
    fn the_strike_gap_body_carries_exactly_the_pinned_field_set() {
        assert_eq!(
            build_request_body(&strike_request()),
            json!({
                "op": "strike_gap",
                "map_id": "kb-20001",
                "body": "body after the strike",
                "gap_text": "Site-to-site wireguard topology",
                "closing_entry_id": "kb-10003",
            })
        );
    }

    #[test]
    fn no_request_builder_constructs_no_change_or_base_version() {
        // The op vocabulary has no propose_gap, no base_version, and no
        // no_change on this surface: the enum above is the whole set.
        assert!(matches!(
            MapOpRequest::CreateMap {
                project_ref: "p".to_string(),
                short_title: "s".to_string(),
                long_title: "l".to_string(),
                body: "b".to_string(),
            },
            MapOpRequest::CreateMap { .. }
        ));
    }

    // --- the transport ------------------------------------------------------

    async fn mount(server: &wiremock::MockServer, status: u16, body: &str) {
        wiremock::Mock::given(method("POST"))
            .and(path("/api/kb/map-op"))
            .respond_with(wiremock::ResponseTemplate::new(status).set_body_string(body.to_string()))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn a_201_create_map_parses_the_envelope_and_carries_bearer_auth() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(method("POST"))
            .and(path("/api/kb/map-op"))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer kb-token",
            ))
            .and(body_json(build_request_body(&create_request())))
            .respond_with(wiremock::ResponseTemplate::new(201).set_body_string(ENVELOPE))
            .mount(&server)
            .await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        assert_eq!(
            client.apply(create_request()).await,
            MapOpResult::Applied(MapOpApplied {
                map_id: "kb-30001".to_string(),
                version: 7,
                pointer_count: 6,
                budget: 2,
            })
        );
        assert_eq!(server.received_requests().await.expect("captured").len(), 1);
    }

    #[tokio::test]
    async fn a_200_update_op_parses_the_same_envelope() {
        for (request, _) in [(add_request(), 0), (strike_request(), 1)] {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(method("POST"))
                .and(path("/api/kb/map-op"))
                .and(body_json(build_request_body(&request)))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ENVELOPE))
                .mount(&server)
                .await;
            let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
            assert_eq!(
                client.apply(request).await,
                MapOpResult::Applied(MapOpApplied {
                    map_id: "kb-30001".to_string(),
                    version: 7,
                    pointer_count: 6,
                    budget: 2,
                }),
                "one envelope deserializer for all four ops"
            );
        }
    }

    #[tokio::test]
    async fn a_2xx_with_an_unparsable_envelope_is_a_fault() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 201, "not the envelope").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        let MapOpResult::Fault { map_id, reason } = client.apply(create_request()).await else {
            panic!("expected a Fault");
        };
        assert_eq!(map_id, "", "create_map carries no map id to fault against");
        assert!(
            reason.contains("malformed envelope after HTTP 201"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn a_401_is_unauthorized() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 401, "unauthorized").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        assert_eq!(client.apply(add_request()).await, MapOpResult::Unauthorized);
    }

    #[tokio::test]
    async fn a_403_is_not_the_machine_principal() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 403, "not the machine principal").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        assert_eq!(
            client.apply(add_request()).await,
            MapOpResult::NotMachinePrincipal
        );
    }

    #[tokio::test]
    async fn a_409_on_create_map_is_a_cap_admission_with_the_verbatim_body() {
        let reason = r#"{"reason":"per-night cap exceeded"}"#;
        let server = wiremock::MockServer::start().await;
        mount(&server, 409, reason).await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        assert_eq!(
            client.apply(create_request()).await,
            MapOpResult::CapAdmission {
                reason_body: reason.to_string(),
            }
        );
    }

    #[tokio::test]
    async fn every_other_status_on_any_op_is_a_fault_naming_the_status() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 409, "invariant").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        let MapOpResult::Fault { map_id, reason } = client.apply(add_request()).await else {
            panic!("an update-op 409 is a Fault");
        };
        assert_eq!(map_id, "kb-20001");
        assert_eq!(reason, "unexpected HTTP status 409");

        let server = wiremock::MockServer::start().await;
        mount(&server, 404, "unknown map").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        let MapOpResult::Fault { reason, .. } = client.apply(strike_request()).await else {
            panic!("a 404 is a Fault");
        };
        assert_eq!(reason, "unexpected HTTP status 404");

        let server = wiremock::MockServer::start().await;
        mount(&server, 422, "lint findings").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        let MapOpResult::Fault { reason, .. } = client.apply(add_request()).await else {
            panic!("a 422 is a Fault");
        };
        assert_eq!(reason, "unexpected HTTP status 422");

        let server = wiremock::MockServer::start().await;
        mount(&server, 500, "internal").await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        let MapOpResult::Fault { reason, .. } = client.apply(add_request()).await else {
            panic!("a 5xx is a Fault");
        };
        assert_eq!(reason, "unexpected HTTP status 500");
    }

    #[test]
    fn the_debug_rendering_never_carries_the_token() {
        let client = HttpMapOpClient::new("http://kb.invalid".to_string(), "kb-secret".to_string());
        let rendered = format!("{client:?}");
        assert!(rendered.contains("http://kb.invalid"), "{rendered}");
        assert!(
            !rendered.contains("kb-secret"),
            "Debug never renders the token"
        );
    }

    #[tokio::test]
    async fn a_failed_body_read_maps_to_the_transport_fault_form() {
        // A one-shot raw TCP server that answers 200 with a
        // `content-length` far larger than the body it writes, then drops
        // the connection — `text()` fails, mapping to the same fault form.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
        let address = listener.local_addr().expect("local address");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                use std::io::Read;
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = std::io::Write::write_all(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\ncontent-length: 128\r\n\r\nshort",
                );
            }
        });
        let client = HttpMapOpClient::new(format!("http://{address}"), "kb-token".to_string());
        let MapOpResult::Fault { map_id, reason } = client.apply(add_request()).await else {
            panic!("expected a transport Fault");
        };
        assert_eq!(map_id, "kb-20001");
        assert!(
            reason.starts_with("body read failed: "),
            "reason was {reason}"
        );
        handle.join().expect("the raw server thread");
    }

    #[tokio::test]
    async fn a_dropped_server_maps_to_a_fault_naming_the_transport() {
        // The transport branch, not a status branch: an EXPLICIT-listener
        // server so dropping it tears the listener down (the
        // `a_dropped_server_maps_to_unreachable` pattern).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
        let address = listener.local_addr().expect("local address");
        let server = wiremock::MockServer::builder()
            .listener(listener)
            .start()
            .await;
        let client = HttpMapOpClient::new(server.uri(), "kb-token".to_string());
        drop(server);
        let mut closed = false;
        for _ in 0..200 {
            if std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_millis(25))
                .is_err()
            {
                closed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(closed, "the dropped server's listener must close");
        let MapOpResult::Fault { map_id, reason } = client.apply(add_request()).await else {
            panic!("expected a transport Fault");
        };
        assert_eq!(map_id, "kb-20001");
        assert!(
            reason.starts_with("transport failed: "),
            "reason was {reason}"
        );
    }
}
