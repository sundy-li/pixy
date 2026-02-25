use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tempfile::tempdir;

const TEXT_RESPONSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-e2e-1\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello from pixy e2e\"},",
    "\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-e2e-1\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
    "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\n",
    "data: [DONE]\n\n"
);

const TOOLCALL_RESPONSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-e2e-tool-1\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,",
    "\"id\":\"call_write_1\",\"type\":\"function\",\"function\":{",
    "\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\\\"note.txt\\\",",
    "\\\"content\\\":\\\"hello from e2e tool\\\"}\"}}]},",
    "\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-e2e-tool-1\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n"
);

const TOOLCALL_FINAL_RESPONSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-e2e-tool-2\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tool finished after write\"},",
    "\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-e2e-tool-2\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
    "\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":7,\"total_tokens\":37}}\n\n",
    "data: [DONE]\n\n"
);

#[test]
fn e2e_cli_prompt_streams_text_from_mock_openai() {
    let conf_dir = tempdir().expect("create temp conf dir");
    let workspace = tempdir().expect("create temp workspace");
    let server = ScriptedOpenAiServer::start(vec![TEXT_RESPONSE_BODY.to_string()]);
    write_pixy_toml(conf_dir.path(), &server.base_url());

    let output = run_pixy_prompt(conf_dir.path(), workspace.path(), "say hello from e2e");
    assert_command_succeeded(&output);

    let stdout = text(&output.stdout);
    assert!(
        stdout.contains("session: "),
        "expected session metadata in stdout, got:\n{stdout}"
    );
    assert!(
        stdout.contains("model: openai-completions/openai/gpt-4o-mini"),
        "expected model metadata in stdout, got:\n{stdout}"
    );
    assert!(
        stdout.contains("hello from pixy e2e"),
        "expected assistant text in stdout, got:\n{stdout}"
    );

    let requests = server.wait_for_requests(1);
    assert_eq!(requests.len(), 1, "expected exactly one LLM request");
    assert!(
        requests[0].contains("\"stream\":true"),
        "request should enable streaming, got body:\n{}",
        requests[0]
    );
    assert!(
        requests[0].contains("say hello from e2e"),
        "request should carry prompt text, got body:\n{}",
        requests[0]
    );
}

#[test]
fn e2e_cli_prompt_executes_write_tool_and_sends_tool_result_back() {
    let conf_dir = tempdir().expect("create temp conf dir");
    let workspace = tempdir().expect("create temp workspace");
    let server = ScriptedOpenAiServer::start(vec![
        TOOLCALL_RESPONSE_BODY.to_string(),
        TOOLCALL_FINAL_RESPONSE_BODY.to_string(),
    ]);
    write_pixy_toml(conf_dir.path(), &server.base_url());

    let output = run_pixy_prompt(
        conf_dir.path(),
        workspace.path(),
        "create note.txt with hello from e2e tool",
    );
    assert_command_succeeded(&output);

    let stdout = text(&output.stdout);
    assert!(
        stdout.contains("tool finished after write"),
        "expected post-tool assistant text in stdout, got:\n{stdout}"
    );

    let note_path = workspace.path().join("note.txt");
    let content = fs::read_to_string(&note_path)
        .unwrap_or_else(|error| panic!("expected {} to exist: {error}", note_path.display()));
    assert_eq!(content, "hello from e2e tool");

    let requests = server.wait_for_requests(2);
    assert_eq!(
        requests.len(),
        2,
        "expected two LLM requests (before/after tool call)"
    );
    assert!(
        requests[0].contains("\"tools\":"),
        "first request should include tool definitions, got:\n{}",
        requests[0]
    );
    assert!(
        requests[0].contains("\"name\":\"write\""),
        "first request should expose write tool schema, got:\n{}",
        requests[0]
    );
    assert!(
        requests[1].contains("\"role\":\"tool\""),
        "second request should include tool result message, got:\n{}",
        requests[1]
    );
    assert!(
        requests[1].contains("\"tool_call_id\":\"call_write_1\""),
        "second request should bind tool result to tool call id, got:\n{}",
        requests[1]
    );
}

fn run_pixy_prompt(conf_dir: &Path, cwd: &Path, prompt: &str) -> Output {
    Command::new(pixy_binary_path())
        .arg("--conf-dir")
        .arg(conf_dir)
        .arg("cli")
        .arg("--no-tui")
        .arg("--no-skills")
        .arg("--cwd")
        .arg(cwd)
        .arg("--prompt")
        .arg(prompt)
        .output()
        .expect("execute pixy binary")
}

fn write_pixy_toml(conf_dir: &Path, base_url: &str) {
    fs::create_dir_all(conf_dir).expect("create conf dir");
    let log_dir = conf_dir.join("logs");
    let content = format!(
        r#"[log]
path = "{}"
level = "error"
stdout = false

[llm]
default_provider = "mock"

[[llm.providers]]
name = "mock"
kind = "chat"
provider = "openai"
api = "openai-completions"
base_url = "{}"
api_key = "test-key"
model = "gpt-4o-mini"
weight = 1
"#,
        log_dir.display(),
        base_url
    );
    fs::write(conf_dir.join("pixy.toml"), content).expect("write pixy.toml");
}

fn pixy_binary_path() -> PathBuf {
    for key in ["CARGO_BIN_EXE_pixy", "NEXTEST_BIN_EXE_pixy"] {
        if let Some(path) = std::env::var_os(key) {
            return PathBuf::from(path);
        }
    }

    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/pixy");
    if fallback.is_file() {
        return fallback;
    }

    panic!(
        "unable to resolve pixy binary path from env (CARGO_BIN_EXE_pixy / NEXTEST_BIN_EXE_pixy) \
or fallback {}",
        fallback.display()
    );
}

fn assert_command_succeeded(output: &Output) {
    if output.status.success() {
        return;
    }
    panic!(
        "pixy exited with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        text(&output.stdout),
        text(&output.stderr)
    );
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

struct ScriptedOpenAiServer {
    base_url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop_tx: Sender<()>,
    join_handle: Option<JoinHandle<()>>,
}

impl ScriptedOpenAiServer {
    fn start(scripted_responses: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind scripted server");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking listener");
        let address = listener.local_addr().expect("resolve local address");

        let responses = Arc::new(Mutex::new(VecDeque::from(scripted_responses)));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (stop_tx, stop_rx) = mpsc::channel();
        let join_handle = Some(spawn_server_loop(
            listener,
            stop_rx,
            responses,
            requests.clone(),
        ));

        Self {
            base_url: format!("http://{address}/v1"),
            requests,
            stop_tx,
            join_handle,
        }
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn wait_for_requests(&self, expected_count: usize) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let snapshot = self.requests.lock().expect("lock requests").clone();
            if snapshot.len() >= expected_count {
                return snapshot;
            }
            if Instant::now() >= deadline {
                return snapshot;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for ScriptedOpenAiServer {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

fn spawn_server_loop(
    listener: TcpListener,
    stop_rx: Receiver<()>,
    scripted_responses: Arc<Mutex<VecDeque<String>>>,
    requests: Arc<Mutex<Vec<String>>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        loop {
            if stop_rx.try_recv().is_ok() {
                break;
            }

            match listener.accept() {
                Ok((mut stream, _addr)) => {
                    if let Ok(body) = read_http_request_body(&mut stream) {
                        if let Ok(mut captured) = requests.lock() {
                            captured.push(body);
                        }
                    }

                    let response_body = scripted_responses
                        .lock()
                        .ok()
                        .and_then(|mut responses| responses.pop_front())
                        .unwrap_or_else(|| TEXT_RESPONSE_BODY.to_string());
                    let _ = write_http_response(&mut stream, &response_body);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    })
}

fn read_http_request_body(stream: &mut TcpStream) -> Result<String, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("set read timeout failed: {error}"))?;

    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| format!("read request header failed: {error}"))?;
        if read == 0 {
            return Err("connection closed before headers were complete".to_string());
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = find_subsequence(&buffer, b"\r\n\r\n") {
            break index + 4;
        }
    };

    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = parse_content_length(headers.as_ref()).unwrap_or(0);
    while buffer.len() < header_end + content_length {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| format!("read request body failed: {error}"))?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    if buffer.len() < header_end + content_length {
        return Err(format!(
            "incomplete HTTP body: expected {content_length} bytes, got {}",
            buffer.len().saturating_sub(header_end)
        ));
    }

    Ok(String::from_utf8_lossy(&buffer[header_end..header_end + content_length]).to_string())
}

fn parse_content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim().eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    })
}

fn write_http_response(stream: &mut TcpStream, body: &str) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|_| stream.write_all(body.as_bytes()))
        .and_then(|_| stream.flush())
        .map_err(|error| format!("write response failed: {error}"))
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
