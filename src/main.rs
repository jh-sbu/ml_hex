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
    #[command(subcommand)]
    cmd: TrainCmd,
}

#[derive(Subcommand)]
enum TrainCmd {
    Alphazero(AlphazeroArgs),
    Ppo(PpoArgs),
}

#[derive(clap::Args)]
struct AlphazeroArgs {
    #[arg(long, default_value_t = 10)]
    generations: u32,
    #[arg(long, default_value_t = 100)]
    games: u32,
    #[arg(long, default_value_t = 200)]
    sims: u32,
    #[arg(long, default_value_t = 100)]
    gradient_steps: u32,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
}

#[derive(clap::Args)]
struct PpoArgs {
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
        Commands::Train(train) => match train.cmd {
            TrainCmd::Alphazero(args) => {
                let config = hex_train::AlphaZeroConfig {
                    generations: args.generations,
                    games_per_gen: args.games,
                    simulations: args.sims,
                    gradient_steps: args.gradient_steps,
                    batch_size: args.batch_size,
                    ..Default::default()
                };
                if let Err(e) = hex_train::train_alphazero(config) {
                    eprintln!("train error: {e}");
                    std::process::exit(1);
                }
            }
            TrainCmd::Ppo(_) => eprintln!("ppo: not yet implemented"),
        },
    }
}
