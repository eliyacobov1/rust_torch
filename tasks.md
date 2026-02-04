# Production-Grade Roadmap (Qx 20xx)

## Deep Sprint (Three Architectural Pillars)
1. **Experiment Persistence Layer** — Run registry, metrics log, artifact tracking, and checkpoint integration.
2. **Service/API Wrapper** — High-level orchestration for training workflows and reproducible configs.
3. **Algorithmic Training Suite** — Model, optimizer, dataset, and trainer pipeline for regression workloads.

**Sprint Outcomes**
- Unified experiment store with JSONL metrics + artifact ledger.
- Service layer for synthetic regression and production-ready training runs.
- End-to-end linear regression training pipeline with optimizer + persistence.

## Strategic Priorities
1. Tensor layout/stride semantics + error taxonomy
2. Serialization & checkpointing (state_dict + metadata)
3. Release hardening & packaging automation

---

## Project Status (Interview Readiness)
- **Status:** Yes — experiment run reporting is now CLI-addressable (list, summary, CSV export) with benchmark coverage; remaining gaps are incremental hardening.

## Minimal Gaps Checklist (100% Readiness)
- [ ] Add streaming quantiles (GK/TDigest) to avoid O(n) metric rollups for large runs.
- [ ] Provide JSON/CSV schema validation for run summaries and exports to guard against partial writes.
- [ ] Add a run comparison command that highlights metric deltas across selected runs.

## Follow-up Tasks (Post-Implementation)
1. **Run comparison reports**: CLI command to compare multiple runs with diff tables + deltas.
2. **Summary rollup acceleration**: adopt streaming quantiles to reduce memory pressure on large metrics logs.
3. **Export schema validation**: publish schema + validate CSV/JSON outputs for automation pipelines.

## Task 1 — Tensor Layout/Stride Semantics + Error Taxonomy
**Goal:** deterministic layout behavior and structured failure modes.

**Requirements**
- Define contiguous/strided tensor invariants.
- Add validation in ops + FX lowering.
- Expose structured `TorchError` variants in Rust + Python.

**Deliverables**
- Design doc + API changes
- Rust tests + Python parity coverage
- Documentation update

**Status**
- ✅ Added overlap-aware stride validation with telemetry counters for layout checks.
- ✅ Extended FX lowering to validate dense, non-overlapping storage layouts.
- ✅ Added error taxonomy coverage and layout-focused tests + benchmark harness.

---

## Task 2 — Serialization & Checkpointing
**Goal:** stable model persistence for production deployment.

**Requirements**
- `state_dict` save/load in Rust + Python
- Versioned metadata (dtype, layout, shape)
- Compatibility checks + upgrade notes

**Deliverables**
- New persistence module + bindings
- Integration tests (save/load parity)
- Docs for checkpoint compatibility

**Status**
- ✅ Implemented versioned checkpoint format (`RTCH` header, metadata + data payload).
- ✅ Rust + Python `state_dict` save/load with layout/dtype validation.
- ✅ Round-trip unit test for checkpoint persistence.

---

## Task 3 — Release Hardening & Packaging
**Goal:** production-grade CI + distribution artifacts.

**Requirements**
- CI matrix (Linux/macOS/Windows)
- Automated wheel builds + versioning
- Perf regression baseline tests

**Deliverables**
- CI pipeline updates
- Release checklist + support matrix

**Status**
- ✅ Expanded CI to a multi-OS matrix (Linux/macOS/Windows) for Rust + Python bindings.
- ✅ Added automated wheel builds with artifact upload for release validation.
- ✅ Added a benchmark regression job for consistent performance tracking.

---

## Task 4 — Autograd Graph Scheduler + Gradient Accumulation
**Goal:** ensure correct, production-grade backpropagation for shared subgraphs and provide observability.

**Requirements**
- Topologically scheduled backward pass with dependency tracking.
- Correct gradient accumulation for tensors used in multiple paths.
- Fail-fast error handling for missing grads/non-scalar losses.
- Lightweight logging/metrics for traversal stats.

**Deliverables**
- Autograd graph executor + stats surface
- Unit test covering shared-node gradient accumulation
- Documentation/roadmap update

**Status**
- ✅ Implemented dependency-tracked autograd traversal with stats + logging.
- ✅ Added shared-node gradient accumulation test.
- ✅ Added configurable parallel backward execution, observer hooks, and autograd benchmark coverage.

---

## Follow-up Tasks (Post-Task 1)
1. **Strided view support with copy-on-write**
   - Enable view semantics for non-contiguous layouts and enforce alias tracking.
2. **Layout-aware kernel dispatch**
   - Select specialized kernels based on stride patterns (contiguous vs. padded vs. transposed).
3. **Layout observability dashboard**
   - Surface layout telemetry counters in the experiment store + CLI.

## Follow-up Tasks (Post-Task 3)
1. **Release checklist automation**
   - Generate release notes, version bumps, and changelog updates from CI metadata.
2. **Cross-platform dependency caching**
   - Introduce sccache and pip caching for faster CI and reproducible builds.
3. **Benchmark telemetry export**
   - Publish benchmark summaries to the experiment store for historical performance tracking.

## Follow-up Tasks (Post-Task 4)
1. **Autograd execution scheduler**
   - Introduce a work-stealing executor for node batches and per-op concurrency tuning.
2. **Autograd tracing exporter**
   - Emit structured spans (per node/batch) into the experiment telemetry store.
3. **Deterministic parallelism controls**
   - Add reproducible scheduling modes with deterministic reduction ordering.

## Technical Debt Log
- **Layout telemetry aggregation**: counters are process-local and not yet wired into the experiment store; schedule a follow-up to emit metrics per run and to reset counters between runs.
- **Benchmark depth in CI**: current CI benchmarks use short measurement windows for speed; schedule nightly runs with full sample sizes for more stable baselines.
- **Autograd parallelism**: current batch-level parallelism uses scoped threads without a dedicated pool; consider a shared executor for large training runs.
