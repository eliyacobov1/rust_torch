use std::env;

use rustorch::api::RustorchService;
use rustorch::data::SyntheticRegressionConfig;
use rustorch::training::TrainerConfig;
use rustorch::{Result, TorchError};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "train-linear" => run_train_linear(&args[2..]),
        "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        command => Err(rustorch::TorchError::InvalidArgument {
            op: "cli",
            msg: format!("unknown command {command}"),
        }),
    }
}

fn run_train_linear(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let samples = parser.get_usize("samples")?.unwrap_or(128);
    let features = parser.get_usize("features")?.unwrap_or(4);
    let targets = parser.get_usize("targets")?.unwrap_or(1);
    let epochs = parser.get_usize("epochs")?.unwrap_or(10);
    let batch_size = parser.get_usize("batch-size")?.unwrap_or(16);
    let learning_rate = parser.get_f32("lr")?.unwrap_or(1e-2);
    let weight_decay = parser.get_f32("weight-decay")?.unwrap_or(0.0);
    let seed = parser.get_u64("seed")?.unwrap_or(42);
    let run_name = parser
        .get("run-name")?
        .unwrap_or_else(|| "linear_regression".to_string());

    let data_config = SyntheticRegressionConfig {
        samples,
        features,
        targets,
        noise_std: parser.get_f32("noise")?.unwrap_or(0.05),
        seed,
    };
    let trainer_config = TrainerConfig {
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        log_every: parser.get_usize("log-every")?.unwrap_or(10),
        checkpoint_every: parser.get_usize("checkpoint-every")?.unwrap_or(1),
        run_name,
        tags: vec!["synthetic".to_string(), "linear".to_string()],
    };

    let service = RustorchService::new(runs_dir)?;
    let report = service.train_synthetic_regression(data_config, trainer_config, seed)?;
    println!(
        "run {} completed: steps={}, final_loss={}, best_loss={}",
        report.report.run_id,
        report.report.total_steps,
        report.report.final_loss,
        report.report.best_loss
    );
    Ok(())
}

fn print_usage() {
    println!(
        "rustorch_cli\n\nUSAGE:\n  rustorch_cli train-linear [options]\n\nOPTIONS:\n  --runs-dir <path>        Directory for experiment runs (default: runs)\n  --samples <n>            Number of samples (default: 128)\n  --features <n>           Number of input features (default: 4)\n  --targets <n>            Number of output targets (default: 1)\n  --epochs <n>             Training epochs (default: 10)\n  --batch-size <n>         Batch size (default: 16)\n  --lr <f>                 Learning rate (default: 1e-2)\n  --weight-decay <f>       Weight decay (default: 0)\n  --noise <f>              Noise stddev for synthetic data (default: 0.05)\n  --seed <n>               RNG seed (default: 42)\n  --run-name <name>        Run name label\n  --log-every <n>          Log metrics every n steps (default: 10)\n  --checkpoint-every <n>   Save checkpoint every n epochs (default: 1)\n  -h, --help               Print this help text\n"
    );
}

struct ArgParser {
    args: Vec<String>,
}

impl ArgParser {
    fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
        }
    }

    fn get(&self, key: &str) -> Result<Option<String>> {
        let flag = format!("--{key}");
        Ok(self
            .args
            .iter()
            .position(|value| value == &flag)
            .and_then(|idx| self.args.get(idx + 1))
            .cloned())
    }

    fn get_usize(&self, key: &str) -> Result<Option<usize>> {
        match self.get(key)? {
            Some(value) => {
                value
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|_| TorchError::InvalidArgument {
                        op: "cli",
                        msg: format!("--{key} expects usize, got '{value}'"),
                    })
            }
            None => Ok(None),
        }
    }

    fn get_u64(&self, key: &str) -> Result<Option<u64>> {
        match self.get(key)? {
            Some(value) => {
                value
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|_| TorchError::InvalidArgument {
                        op: "cli",
                        msg: format!("--{key} expects u64, got '{value}'"),
                    })
            }
            None => Ok(None),
        }
    }

    fn get_f32(&self, key: &str) -> Result<Option<f32>> {
        match self.get(key)? {
            Some(value) => {
                value
                    .parse::<f32>()
                    .map(Some)
                    .map_err(|_| TorchError::InvalidArgument {
                        op: "cli",
                        msg: format!("--{key} expects f32, got '{value}'"),
                    })
            }
            None => Ok(None),
        }
    }
}
