use std::env;

use rustorch::api::RustorchService;
use rustorch::data::{SyntheticClassificationConfig, SyntheticRegressionConfig};
use rustorch::experiment::{CsvExportReport, ExperimentStore, RunFilter, RunStatus};
use rustorch::training::{ClassificationTrainerConfig, TrainerConfig};
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
        "train-mlp" => run_train_mlp(&args[2..]),
        "runs-list" => run_runs_list(&args[2..]),
        "runs-summary" => run_runs_summary(&args[2..]),
        "runs-export-csv" => run_runs_export_csv(&args[2..]),
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

fn run_train_mlp(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let samples = parser.get_usize("samples")?.unwrap_or(256);
    let features = parser.get_usize("features")?.unwrap_or(4);
    let classes = parser.get_usize("classes")?.unwrap_or(3);
    let hidden = parser.get_usize("hidden")?.unwrap_or(16);
    let epochs = parser.get_usize("epochs")?.unwrap_or(12);
    let batch_size = parser.get_usize("batch-size")?.unwrap_or(32);
    let learning_rate = parser.get_f32("lr")?.unwrap_or(5e-2);
    let weight_decay = parser.get_f32("weight-decay")?.unwrap_or(0.0);
    let seed = parser.get_u64("seed")?.unwrap_or(7);
    let run_name = parser
        .get("run-name")?
        .unwrap_or_else(|| "mlp_classifier".to_string());

    let data_config = SyntheticClassificationConfig {
        samples,
        features,
        classes,
        cluster_std: parser.get_f32("cluster-std")?.unwrap_or(0.35),
        seed,
    };
    let trainer_config = ClassificationTrainerConfig {
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        log_every: parser.get_usize("log-every")?.unwrap_or(10),
        checkpoint_every: parser.get_usize("checkpoint-every")?.unwrap_or(1),
        run_name,
        tags: vec![
            "synthetic".to_string(),
            "mlp".to_string(),
            "classification".to_string(),
        ],
    };

    let service = RustorchService::new(runs_dir)?;
    let report =
        service.train_synthetic_classification(data_config, trainer_config, hidden, seed)?;
    println!(
        "run {} completed: steps={}, final_loss={}, best_loss={}, final_accuracy={}, best_accuracy={}",
        report.report.run_id,
        report.report.total_steps,
        report.report.final_loss,
        report.report.best_loss,
        report.report.final_accuracy,
        report.report.best_accuracy
    );
    Ok(())
}

fn print_usage() {
    println!(
        "rustorch_cli\n\nUSAGE:\n  rustorch_cli train-linear [options]\n  rustorch_cli train-mlp [options]\n  rustorch_cli runs-list [options]\n  rustorch_cli runs-summary [options]\n  rustorch_cli runs-export-csv [options]\n\nOPTIONS (shared):\n  --runs-dir <path>        Directory for experiment runs (default: runs)\n  --samples <n>            Number of samples (default: 128)\n  --features <n>           Number of input features (default: 4)\n  --epochs <n>             Training epochs (default: 10)\n  --batch-size <n>         Batch size (default: 16)\n  --lr <f>                 Learning rate (default: 1e-2)\n  --weight-decay <f>       Weight decay (default: 0)\n  --seed <n>               RNG seed (default: 42)\n  --run-name <name>        Run name label\n  --log-every <n>          Log metrics every n steps (default: 10)\n  --checkpoint-every <n>   Save checkpoint every n epochs (default: 1)\n\nOPTIONS (runs-list):\n  --tags <csv>             Filter by tag(s) (comma-separated)\n  --status <status>        Filter by status (running/completed/failed)\n\nOPTIONS (runs-summary):\n  --run-id <id>            Run identifier to summarize\n\nOPTIONS (runs-export-csv):\n  --output <path>          Output CSV path (default: runs_summary.csv)\n  --tags <csv>             Filter by tag(s) (comma-separated)\n  --status <status>        Filter by status (running/completed/failed)\n\nOPTIONS (train-linear):\n  --targets <n>            Number of output targets (default: 1)\n  --noise <f>              Noise stddev for synthetic data (default: 0.05)\n\nOPTIONS (train-mlp):\n  --classes <n>            Number of classes (default: 3)\n  --hidden <n>             Hidden layer size (default: 16)\n  --cluster-std <f>        Cluster stddev for synthetic data (default: 0.35)\n  -h, --help               Print this help text\n"
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

    fn get_csv(&self, key: &str) -> Result<Option<Vec<String>>> {
        let value = self.get(key)?;
        Ok(value.map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(String::from)
                .collect::<Vec<String>>()
        }))
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

fn run_runs_list(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let tags = parser.get_csv("tags")?.unwrap_or_default();
    let statuses = parser.get("status")?.map(parse_status).transpose()?;
    let filter = RunFilter { tags, statuses };
    let store = ExperimentStore::new(runs_dir)?;
    let overviews = store.list_overviews(&filter)?;
    if overviews.is_empty() {
        println!("No runs found.");
        return Ok(());
    }
    for overview in overviews {
        let summary_suffix = if overview.summary.is_some() {
            "summary"
        } else {
            "no_summary"
        };
        println!(
            "{} | {} | {:?} | tags=[{}] | {}",
            overview.metadata.id,
            overview.metadata.name,
            overview.metadata.status,
            overview.metadata.tags.join(","),
            summary_suffix
        );
    }
    Ok(())
}

fn run_runs_summary(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let run_id = parser.get("run-id")?.ok_or(TorchError::InvalidArgument {
        op: "cli",
        msg: "--run-id is required".to_string(),
    })?;
    let store = ExperimentStore::new(runs_dir)?;
    let summary = store.read_summary(&run_id)?;
    let json = serde_json::to_string_pretty(&summary).map_err(|err| TorchError::Experiment {
        op: "cli.runs_summary",
        msg: format!("failed to serialize summary: {err}"),
    })?;
    println!("{json}");
    Ok(())
}

fn run_runs_export_csv(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let output = parser
        .get("output")?
        .unwrap_or_else(|| "runs_summary.csv".to_string());
    let tags = parser.get_csv("tags")?.unwrap_or_default();
    let statuses = parser.get("status")?.map(parse_status).transpose()?;
    let filter = RunFilter { tags, statuses };
    let store = ExperimentStore::new(runs_dir)?;
    let report = store.export_run_summaries_csv(&output, &filter)?;
    print_export_report(&report);
    Ok(())
}

fn parse_status(value: String) -> Result<Vec<RunStatus>> {
    match value.to_ascii_lowercase().as_str() {
        "running" => Ok(vec![RunStatus::Running]),
        "completed" => Ok(vec![RunStatus::Completed]),
        "failed" => Ok(vec![RunStatus::Failed]),
        "all" => Ok(vec![
            RunStatus::Running,
            RunStatus::Completed,
            RunStatus::Failed,
        ]),
        other => Err(TorchError::InvalidArgument {
            op: "cli",
            msg: format!("unknown status {other}"),
        }),
    }
}

fn print_export_report(report: &CsvExportReport) {
    println!(
        "exported {} runs to {} ({} bytes, {:.2} ms)",
        report.rows,
        report.output_path.display(),
        report.bytes_written,
        report.duration_ms
    );
}
