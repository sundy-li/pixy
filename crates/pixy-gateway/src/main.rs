use clap::{Args, Parser, Subcommand};
use pixy_gateway::{run_gateway_command, GatewayCommand, GatewayStartOptions};

#[derive(Parser, Debug)]
#[command(name = "pixy-gateway", version, about = "pixy gateway runtime")]
struct Cli {
    #[arg(long, global = true)]
    conf_dir: Option<std::path::PathBuf>,
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
    pixy_gateway::init_conf_dir(cli.conf_dir.clone());
    pixy_gateway::init_tracing();
    let result = match cli.command {
        GatewaySubcommand::Start(start) => {
            run_gateway_command(GatewayCommand::Start(GatewayStartOptions {
                daemon: start.daemon,
            }))
            .await
        }
        GatewaySubcommand::Stop => run_gateway_command(GatewayCommand::Stop).await,
        GatewaySubcommand::Restart => run_gateway_command(GatewayCommand::Restart).await,
        GatewaySubcommand::Serve => pixy_gateway::run_gateway_serve().await,
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_start_daemon_flag() {
        let parsed = Cli::try_parse_from(["pixy-gateway", "start", "--daemon"]);
        assert!(parsed.is_ok(), "start --daemon should parse");
    }

    #[test]
    fn cli_parses_restart_command() {
        let parsed = Cli::try_parse_from(["pixy-gateway", "restart"]);
        assert!(parsed.is_ok(), "restart should parse");
    }

    #[test]
    fn cli_parses_conf_dir_global_flag() {
        let parsed = Cli::try_parse_from(["pixy-gateway", "--conf-dir", "/tmp/pixy-conf", "start"]);
        assert!(parsed.is_ok(), "--conf-dir should parse as global flag");
    }
}
