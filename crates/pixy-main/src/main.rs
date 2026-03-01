use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use pixy_coding_agent::cli::ChatArgs;
use pixy_gateway::{run_gateway_command, GatewayCommand, GatewayStartOptions};

mod config_cmd;
mod doctor;
mod pixy_home;
mod update_cmd;

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
    Config(ConfigArgs),
    Doctor,
    Update(UpdateArgs),
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

#[derive(Args, Debug, Clone)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigSubcommand,
}

#[derive(Subcommand, Debug, Clone)]
enum ConfigSubcommand {
    Init,
}

#[derive(Args, Debug, Clone)]
struct UpdateArgs {
    #[arg(long)]
    version: Option<String>,
    #[arg(long)]
    repo: Option<String>,
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
        Some(RootCommand::Config(args)) => run_config(args.command, conf_dir),
        Some(RootCommand::Doctor) => doctor::run_doctor(conf_dir),
        Some(RootCommand::Update(args)) => run_update(args),
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

fn run_config(command: ConfigSubcommand, conf_dir: Option<PathBuf>) -> Result<(), String> {
    match command {
        ConfigSubcommand::Init => config_cmd::run_config_init(conf_dir),
    }
}

fn run_update(args: UpdateArgs) -> Result<(), String> {
    update_cmd::run_update(update_cmd::UpdateCommandArgs {
        version: args.version,
        repo: args.repo,
    })
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

    #[test]
    fn cli_accepts_config_init_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "config", "init"]);
        assert!(parsed.is_ok(), "pixy config init should be accepted");
    }

    #[test]
    fn cli_accepts_update_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "update", "--version", "v0.1.0"]);
        assert!(parsed.is_ok(), "pixy update should be accepted");
    }
}
