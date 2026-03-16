use hex_agents::{
    AlphaZeroAgent, AlphaZeroConfig, HeuristicAgent, MctsAgent, MctsConfig, PpoAgent, RandomAgent,
};
use hex_tui::{PlayerConfig, TuiConfig};

#[derive(clap::Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    Play(PlayArgs),
    Train(TrainArgs),
}

#[derive(clap::Args)]
struct PlayArgs {
    /// Agent spec: human | random | heuristic | mcts[:<budget>]
    ///             | alphazero:<path> | ppo:<path>[,greedy]
    #[arg(long, default_value = "human")]
    red_agent: String,

    #[arg(long, default_value = "human")]
    blue_agent: String,

    #[arg(long, default_value_t = false)]
    swap: bool,
}

#[derive(clap::Args)]
struct TrainArgs {
    #[command(subcommand)]
    cmd: TrainCmd,
}

#[derive(clap::Subcommand)]
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
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,
}

#[derive(clap::Args)]
struct PpoArgs {
    #[arg(long, default_value_t = 50)]
    generations: u32,
    #[arg(long, default_value_t = 20)]
    episodes: u32,
    #[arg(long, default_value_t = 4)]
    ppo_epochs: u32,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,
}

fn make_agent(spec: &str) -> PlayerConfig {
    if spec == "human" {
        return PlayerConfig::Human;
    }
    let (kind, rest) = match spec.find(':') {
        Some(pos) => (&spec[..pos], &spec[pos + 1..]),
        None => (spec, ""),
    };
    match kind {
        "random" => PlayerConfig::agent(RandomAgent),
        "heuristic" => PlayerConfig::agent(HeuristicAgent),
        "mcts" => {
            let budget: u32 = if rest.is_empty() {
                1000
            } else {
                rest.parse().unwrap_or(1000)
            };
            PlayerConfig::agent(MctsAgent::new(MctsConfig {
                rollout_budget: budget,
                ..MctsConfig::default()
            }))
        }
        "alphazero" => {
            if rest.is_empty() {
                eprintln!("alphazero:<path> required");
                std::process::exit(1);
            }
            PlayerConfig::agent(AlphaZeroAgent::load(AlphaZeroConfig {
                checkpoint_path: rest.to_string(),
                ..AlphaZeroConfig::default()
            }))
        }
        "ppo" => {
            let (path, greedy) = match rest.rfind(",greedy") {
                Some(pos) => (&rest[..pos], true),
                None => (rest, false),
            };
            if path.is_empty() {
                eprintln!("ppo:<path>[,greedy] required");
                std::process::exit(1);
            }
            PlayerConfig::agent(PpoAgent::load(path, greedy))
        }
        other => {
            eprintln!("unknown agent '{other}'");
            std::process::exit(1);
        }
    }
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();
    match cli.command {
        Commands::Play(args) => {
            let config = TuiConfig::new(
                make_agent(&args.red_agent),
                make_agent(&args.blue_agent),
                args.swap,
            );
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
                    checkpoint_dir: args.checkpoint_dir,
                    ..Default::default()
                };
                if let Err(e) = hex_train::train_alphazero(config) {
                    eprintln!("train error: {e}");
                    std::process::exit(1);
                }
            }
            TrainCmd::Ppo(args) => {
                let config = hex_train::PpoConfig {
                    generations: args.generations,
                    episodes_per_gen: args.episodes,
                    ppo_epochs: args.ppo_epochs,
                    batch_size: args.batch_size,
                    checkpoint_dir: args.checkpoint_dir,
                    ..Default::default()
                };
                if let Err(e) = hex_train::train_ppo(config) {
                    eprintln!("train error: {e}");
                    std::process::exit(1);
                }
            }
        },
    }
}
