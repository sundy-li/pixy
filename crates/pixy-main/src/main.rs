use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use pixy_coding_agent::cli::ChatArgs;
use pixy_gateway::{GatewayCommand, GatewayStartOptions, run_gateway_command};

mod doctor;

#[derive(Parser, Debug)]
#[command(name = "pixy", version, about = "pixy command dispatcher")]
struct Cli {
    #[arg(long, global = true)]
    conf_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<RootCommand>,
    #[command(flatten)]
    chat: ChatArgs,
}

#[derive(Subcommand, Debug, Clone)]
enum RootCommand {
    Cli(ChatArgs),
    Gateway(GatewayArgs),
    Doctor,
}

#[derive(Args, Debug, Clone)]
struct GatewayArgs {
    #[command(subcommand)]
    command: GatewaySubcommand,
}

#[derive(Subcommand, Debug, Clone)]
enum GatewaySubcommand {
    Start(GatewayStartArgs),
    Stop,
    Restart,
    #[command(hide = true)]
    Serve,
}

#[derive(Args, Debug, Clone)]
struct GatewayStartArgs {
    #[arg(long, default_value_t = false)]
    daemon: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let conf_dir = cli.conf_dir.clone();
    let result = match cli.command {
        Some(RootCommand::Cli(args)) => {
            pixy_coding_agent::cli::run_chat_with_conf(args, conf_dir).await
        }
        Some(RootCommand::Gateway(args)) => run_gateway(args.command, conf_dir).await,
        Some(RootCommand::Doctor) => doctor::run_doctor(conf_dir),
        None => pixy_coding_agent::cli::run_chat_with_conf(cli.chat, conf_dir).await,
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run_gateway(command: GatewaySubcommand, conf_dir: Option<PathBuf>) -> Result<(), String> {
    pixy_gateway::init_conf_dir(conf_dir);
    pixy_gateway::init_tracing();

    match command {
        GatewaySubcommand::Start(start) => {
            run_gateway_command(GatewayCommand::Start(GatewayStartOptions {
                daemon: start.daemon,
            }))
            .await
        }
        GatewaySubcommand::Stop => run_gateway_command(GatewayCommand::Stop).await,
        GatewaySubcommand::Restart => run_gateway_command(GatewayCommand::Restart).await,
        GatewaySubcommand::Serve => pixy_gateway::run_gateway_serve().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_explicit_cli_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "cli", "--no-tui"]);
        assert!(
            parsed.is_ok(),
            "pixy cli should be accepted as explicit CLI entrypoint"
        );
    }

    #[test]
    fn cli_accepts_gateway_start_daemon_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "gateway", "start", "--daemon"]);
        assert!(
            parsed.is_ok(),
            "pixy gateway start --daemon should be accepted"
        );
    }

    #[test]
    fn cli_accepts_conf_dir_global_flag() {
        let parsed =
            Cli::try_parse_from(["pixy", "--conf-dir", "/tmp/pixy-conf", "gateway", "start"]);
        assert!(
            parsed.is_ok(),
            "pixy should accept --conf-dir as global flag"
        );
    }

    #[test]
    fn cli_accepts_doctor_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "doctor"]);
        assert!(parsed.is_ok(), "pixy doctor should be accepted");
    }
}
