use std::env;

use rustorch::api::RustorchService;
use rustorch::audit::{verify_audit_log, AuditVerificationConfig};
use rustorch::data::{SyntheticClassificationConfig, SyntheticRegressionConfig};
use rustorch::experiment::{
    CsvExportReport, ExperimentStore, MetricAggregation, RunComparisonConfig, RunComparisonReport,
    RunFilter, RunGovernanceConfig, RunGovernanceReport, RunStatus,
};
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
        "runs-compare" => run_runs_compare(&args[2..]),
        "runs-validate" => run_runs_validate(&args[2..]),
        "audit-verify" => run_audit_verify(&args[2..]),
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
        "rustorch_cli\n\nUSAGE:\n  rustorch_cli train-linear [options]\n  rustorch_cli train-mlp [options]\n  rustorch_cli runs-list [options]\n  rustorch_cli runs-summary [options]\n  rustorch_cli runs-export-csv [options]\n  rustorch_cli runs-compare [options]\n  rustorch_cli runs-validate [options]\n  rustorch_cli audit-verify [options]\n\nOPTIONS (shared):\n  --runs-dir <path>        Directory for experiment runs (default: runs)\n  --samples <n>            Number of samples (default: 128)\n  --features <n>           Number of input features (default: 4)\n  --epochs <n>             Training epochs (default: 10)\n  --batch-size <n>         Batch size (default: 16)\n  --lr <f>                 Learning rate (default: 1e-2)\n  --weight-decay <f>       Weight decay (default: 0)\n  --seed <n>               RNG seed (default: 42)\n  --run-name <name>        Run name label\n  --log-every <n>          Log metrics every n steps (default: 10)\n  --checkpoint-every <n>   Save checkpoint every n epochs (default: 1)\n\nOPTIONS (runs-list):\n  --tags <csv>             Filter by tag(s) (comma-separated)\n  --status <status>        Filter by status (running/completed/failed)\n\nOPTIONS (runs-summary):\n  --run-id <id>            Run identifier to summarize\n\nOPTIONS (runs-export-csv):\n  --output <path>          Output CSV path (default: runs_summary.csv)\n  --tags <csv>             Filter by tag(s) (comma-separated)\n  --status <status>        Filter by status (running/completed/failed)\n\nOPTIONS (runs-compare):\n  --run-ids <csv>          Explicit run IDs to compare\n  --baseline-id <id>       Run ID to use as baseline (default: first in set)\n  --metric-agg <name>      Aggregation: min|max|mean|p50|p95|last (default: last)\n  --top-k <n>              Top deltas to print per run (default: 5)\n  --format <name>          Output format: table|json (default: table)\n  --no-graph               Skip pairwise comparison graph\n  --tags <csv>             Filter by tag(s) (comma-separated)\n  --status <status>        Filter by status (running/completed/failed)\n\nOPTIONS (runs-validate):\n  --workers <n>             Number of validation workers\n  --quarantine              Move invalid runs into quarantine\n  --quarantine-dir <path>   Quarantine directory override\n  --output <path>           Write JSON report to path\n  --lenient                 Downgrade schema errors to warnings where possible\n  --no-orphaned             Skip orphaned file detection\n  --no-metrics              Skip metrics.jsonl validation\n  --no-telemetry            Skip telemetry.jsonl validation\n  --audit                   Enable governance audit logging\n  --audit-log <path>        Audit log path override\n  --audit-verify            Verify audit log integrity after validation\n  --no-audit-verify         Skip audit log verification\n  --audit-expected-root <h> Expected Merkle root for audit verification\n  --audit-proofs            Include Merkle proofs in audit verification output\n  --audit-proof-samples <n> Max number of audit proofs to include\n  --no-remediation          Skip remediation ticket generation\n\nOPTIONS (audit-verify):\n  --audit-log <path>        Audit log path to verify\n  --expected-root <hash>    Expected Merkle root to verify against\n  --proofs                  Include Merkle proofs in the output\n  --max-proofs <n>          Maximum proofs to include (default: 5)\n  --output <path>           Write JSON report to path\n\nOPTIONS (train-linear):\n  --targets <n>            Number of output targets (default: 1)\n  --noise <f>              Noise stddev for synthetic data (default: 0.05)\n\nOPTIONS (train-mlp):\n  --classes <n>            Number of classes (default: 3)\n  --hidden <n>             Hidden layer size (default: 16)\n  --cluster-std <f>        Cluster stddev for synthetic data (default: 0.35)\n  -h, --help               Print this help text\n"
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

    fn has_flag(&self, key: &str) -> bool {
        let flag = format!("--{key}");
        self.args.iter().any(|value| value == &flag)
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

fn run_runs_validate(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let output = parser.get("output")?;
    let mut config = RunGovernanceConfig::default();
    if let Some(workers) = parser.get_usize("workers")? {
        config.max_workers = workers;
    }
    if parser.has_flag("no-orphaned") {
        config.include_orphaned_files = false;
    }
    if parser.has_flag("no-metrics") {
        config.check_metrics = false;
    }
    if parser.has_flag("no-telemetry") {
        config.check_telemetry = false;
    }
    if parser.has_flag("quarantine") {
        config.quarantine = true;
    }
    if parser.has_flag("lenient") {
        config.strict_schema = false;
    }
    if let Some(quarantine_dir) = parser.get("quarantine-dir")? {
        config.quarantine_dir = Some(quarantine_dir.into());
    }
    if parser.has_flag("audit") {
        config.audit_log = true;
    }
    if let Some(audit_log_path) = parser.get("audit-log")? {
        config.audit_log = true;
        config.audit_log_path = Some(audit_log_path.into());
    }
    if parser.has_flag("no-audit-verify") {
        config.audit_verify = false;
    }
    if parser.has_flag("audit-verify") {
        config.audit_verify = true;
    }
    if let Some(expected_root) = parser.get("audit-expected-root")? {
        config.audit_expected_root = Some(expected_root);
    }
    if parser.has_flag("audit-proofs") {
        config.audit_include_proofs = true;
    }
    if let Some(samples) = parser.get_usize("audit-proof-samples")? {
        config.audit_max_proofs = samples;
    }
    if parser.has_flag("no-remediation") {
        config.emit_remediation = false;
    }

    let store = ExperimentStore::new(runs_dir)?;
    let report = store.validate_runs(&config)?;
    print_governance_report(&report);
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&report).map_err(|err| TorchError::Experiment {
            op: "cli.runs_validate",
            msg: format!("failed to serialize report: {err}"),
        })?;
        std::fs::write(&output_path, json).map_err(|err| TorchError::Experiment {
            op: "cli.runs_validate",
            msg: format!("failed to write report {}: {err}", output_path),
        })?;
    }
    Ok(())
}

fn run_audit_verify(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let output = parser.get("output")?;
    let audit_log = parser
        .get("audit-log")?
        .map(|value| value.into())
        .unwrap_or_else(|| {
            let mut path = std::path::PathBuf::from(&runs_dir);
            path.push("audit");
            path.push("run_governance_audit.jsonl");
            path
        });
    let expected_root = parser.get("expected-root")?;
    let mut config = AuditVerificationConfig::default();
    config.expected_root = expected_root;
    if parser.has_flag("proofs") {
        config.include_proofs = true;
    }
    if let Some(max_proofs) = parser.get_usize("max-proofs")? {
        config.max_proofs = max_proofs;
    }
    let report = verify_audit_log(&audit_log, &config)?;
    print_audit_report(&report);
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&report).map_err(|err| TorchError::Experiment {
            op: "cli.audit_verify",
            msg: format!("failed to serialize report: {err}"),
        })?;
        std::fs::write(&output_path, json).map_err(|err| TorchError::Experiment {
            op: "cli.audit_verify",
            msg: format!("failed to write report {}: {err}", output_path),
        })?;
    }
    Ok(())
}

fn run_runs_compare(args: &[String]) -> Result<()> {
    let parser = ArgParser::new(args);
    let runs_dir = parser
        .get("runs-dir")?
        .unwrap_or_else(|| "runs".to_string());
    let run_ids = parser.get_csv("run-ids")?.unwrap_or_default();
    let baseline_id = parser.get("baseline-id")?;
    let tags = parser.get_csv("tags")?.unwrap_or_default();
    let statuses = parser.get("status")?.map(parse_status).transpose()?;
    let filter = RunFilter { tags, statuses };
    let metric_aggregation = parser
        .get("metric-agg")?
        .map(|value| MetricAggregation::from_str(&value))
        .transpose()?
        .unwrap_or(MetricAggregation::Last);
    let top_k = parser.get_usize("top-k")?.unwrap_or(5);
    let format = parser.get("format")?.unwrap_or_else(|| "table".to_string());
    let build_graph = !parser.has_flag("no-graph");

    let config = RunComparisonConfig {
        run_ids,
        filter,
        baseline_id,
        metric_aggregation,
        top_k,
        build_graph,
    };

    let store = ExperimentStore::new(runs_dir)?;
    let report = store.compare_runs(&config)?;

    match format.as_str() {
        "table" => print_compare_report(&report),
        "json" => {
            let json =
                serde_json::to_string_pretty(&report).map_err(|err| TorchError::Experiment {
                    op: "cli.runs_compare",
                    msg: format!("failed to serialize report: {err}"),
                })?;
            println!("{json}");
        }
        other => {
            return Err(TorchError::InvalidArgument {
                op: "cli.runs_compare",
                msg: format!("unknown format {other}"),
            })
        }
    }
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
        "exported {} runs to {} ({} bytes, {:.2} ms, validated {} runs in {:.2} ms)",
        report.rows,
        report.output_path.display(),
        report.bytes_written,
        report.duration_ms,
        report.validation_checks,
        report.validation_ms
    );
}

fn print_governance_report(report: &RunGovernanceReport) {
    println!(
        "governance report: total={}, valid={}, invalid={}, quarantined={}, warnings={}",
        report.summary.total_runs,
        report.summary.valid_runs,
        report.summary.invalid_runs,
        report.summary.quarantined_runs,
        report.summary.warning_count
    );
    if let Some(path) = &report.audit_log_path {
        let root = report.audit_merkle_root.as_deref().unwrap_or("n/a");
        println!("audit log: {} (merkle_root={})", path.display(), root);
    }
    if let Some(verification) = &report.audit_verification {
        println!(
            "audit verification: status={:?}, events={}, issues={}, duration_ms={:.2}",
            verification.status,
            verification.total_events,
            verification.issues.len(),
            verification.duration_ms
        );
        if let Some(expected) = &verification.expected_root {
            let matches = verification
                .matches_expected_root
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            let actual = verification.merkle_root.as_deref().unwrap_or("n/a");
            println!(
                "audit root: expected={}, actual={}, matches={}",
                expected, actual, matches
            );
        }
        if !verification.issues.is_empty() {
            for issue in &verification.issues {
                println!(
                    "  - audit issue {:?} (line={}): {}",
                    issue.kind, issue.line, issue.message
                );
            }
        }
    }
    for result in &report.results {
        println!(
            "- {}: {:?} ({} findings, {:.2} ms)",
            result.run_id,
            result.status,
            result.findings.len(),
            result.duration_ms
        );
        for finding in &result.findings {
            println!(
                "  - {:?} {:?}: {}",
                finding.level, finding.category, finding.message
            );
        }
        if let Some(path) = &result.quarantine_path {
            println!("  - quarantined to {}", path.display());
        }
    }
    if report.remediation.total_tickets > 0 {
        println!(
            "remediation tickets: total={}, high_severity={}, quarantined={}",
            report.remediation.total_tickets,
            report.remediation.high_severity,
            report.remediation.quarantined
        );
        for ticket in &report.remediation.tickets {
            println!(
                "  - {} (run={}, severity={:?})",
                ticket.ticket_id, ticket.run_id, ticket.severity
            );
            for action in &ticket.recommended_actions {
                println!("    - action: {action}");
            }
        }
    }
}

fn print_audit_report(report: &rustorch::audit::AuditVerificationReport) {
    println!(
        "audit verification: status={:?}, events={}, issues={}, duration_ms={:.2}",
        report.status,
        report.total_events,
        report.issues.len(),
        report.duration_ms
    );
    if let Some(root) = &report.merkle_root {
        println!("merkle_root: {root}");
    }
    if let Some(expected) = &report.expected_root {
        let matches = report
            .matches_expected_root
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        println!("expected_root: {expected} (matches={matches})");
    }
    if !report.issues.is_empty() {
        println!("issues:");
        for issue in &report.issues {
            println!(
                "  - {:?} (line={}, index={:?}): {}",
                issue.kind, issue.line, issue.index, issue.message
            );
        }
    }
    if !report.proofs.is_empty() {
        println!("proofs:");
        for proof in &report.proofs {
            println!(
                "  - leaf_index={}, valid={}, path_len={}",
                proof.leaf_index,
                proof.valid,
                proof.path.len()
            );
        }
    }
}

fn print_compare_report(report: &RunComparisonReport) {
    println!(
        "baseline: {} ({})",
        report.baseline_id, report.baseline_name
    );
    println!(
        "metric aggregation: {:?}, generated_at_unix: {}",
        report.metric_aggregation, report.generated_at_unix
    );
    println!();
    println!(
        "{:<24} {:<18} {:<10} {:>14} {:>14} {:>10}",
        "run_id", "name", "status", "mean_abs_delta", "mean_delta", "missing"
    );
    println!("{}", "-".repeat(96));
    for comparison in &report.comparisons {
        println!(
            "{:<24} {:<18} {:<10?} {:>14.6} {:>14.6} {:>10}",
            comparison.run_id,
            comparison.name,
            comparison.status,
            comparison.summary.mean_abs_delta,
            comparison.summary.mean_delta,
            comparison.summary.missing_metrics
        );
        for delta in comparison.top_deltas.iter() {
            let pct = delta
                .delta_pct
                .map(|value| format!("{value:.2}%"))
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  - {:<20} base={:>10.4} cand={:>10.4} delta={:>10.4} ({:>8})",
                delta.metric, delta.baseline, delta.candidate, delta.delta, pct
            );
        }
        if !comparison.missing_metrics.is_empty() {
            println!("  - missing: {}", comparison.missing_metrics.join(", "));
        }
        println!();
    }

    if !report.top_deltas.is_empty() {
        println!("global top deltas (by absolute change):");
        for delta in &report.top_deltas {
            let pct = delta
                .delta_pct
                .map(|value| format!("{value:.2}%"))
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  - {:<20} base={:>10.4} cand={:>10.4} delta={:>10.4} ({:>8})",
                delta.metric, delta.baseline, delta.candidate, delta.delta, pct
            );
        }
    }

    if let Some(graph) = &report.graph {
        println!();
        println!(
            "comparison graph: {} nodes, {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );
    }
}
