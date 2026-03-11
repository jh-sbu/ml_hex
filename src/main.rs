use clap::{Parser, Subcommand, ValueEnum};
use hex_agents::{HeuristicAgent, RandomAgent};
use hex_tui::{PlayerConfig, TuiConfig};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Play(PlayArgs),
    Train(TrainArgs),
}

#[derive(clap::Args)]
struct PlayArgs {
    #[arg(long, default_value = "hvh")]
    mode: PlayMode,
}

#[derive(ValueEnum, Clone)]
enum PlayMode {
    Hvh,
    Hva,
    Watch,
}

#[derive(clap::Args)]
struct TrainArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    _args: Vec<String>,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Play(args) => {
            let config = match args.mode {
                PlayMode::Hvh => TuiConfig::new(PlayerConfig::Human, PlayerConfig::Human),
                PlayMode::Hva => TuiConfig::new(
                    PlayerConfig::Human,
                    PlayerConfig::agent(HeuristicAgent),
                ),
                PlayMode::Watch => TuiConfig::new(
                    PlayerConfig::agent(HeuristicAgent),
                    PlayerConfig::agent(RandomAgent),
                ),
            };
            if let Err(e) = hex_tui::run(config) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Train(_) => eprintln!("train: not yet implemented"),
    }
}
