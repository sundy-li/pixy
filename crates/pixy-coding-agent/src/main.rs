#[tokio::main]
async fn main() {
    if let Err(error) = pixy_coding_agent::cli::run_cli_process().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
