# Senior Interview Readiness Roadmap (v5)

## Scope
Deliver an end-to-end deterministic ML governance platform with audit-grade replay, regression defense, and performance evidence that is interview-ready at a senior/principal bar. This roadmap is **finite** and **sequence-locked**; execute sprints in order.

---

## Sprint 1: Deterministic Replay & Governance-Execution Integrity
**Target subsystem:** governance + execution + audit replay.

**Execution tasks (in order):**
1. **Execution plan audit anchoring** — record execution plans into the audit log alongside governance plans and expose plan hashes/roots in run validation output.
2. **Execution ledger verification** — add a verifier for execution ledgers (hash-chain checks + optional Merkle root) and a replay cursor that validates stage ordering against the recorded plan.
3. **Replay CLI & manifests** — add CLI commands to emit deterministic replay manifests and validate governance + execution plan parity against ledgers.
4. **Determinism test suite** — integration tests that assert identical schedule/plan outputs across seeds, plus replay verification for a recorded run.

**Success criteria:**
- Audit output includes governance **and** execution plan digests with verifiable hashes.
- A replay command can verify a recorded execution ledger end-to-end and fails on any ordering/hash mismatch.
- Determinism tests pass for repeated runs with fixed seeds and fail on intentional perturbation.

---

## Sprint 2: Experiment Intelligence Graph & Regression Gates
**Target subsystem:** experiment comparison + regression defense.

**Execution tasks (in order):**
1. **Persisted delta index & graph artifacts** — store delta index + comparison graph artifacts with schema versions in each run directory.
2. **Regression gate enforcement** — attach regression gate reports to comparisons, emit audit findings, and integrate a hard/soft fail policy.
3. **Provenance-rich summaries** — extend run summaries with comparison provenance (baseline IDs, aggregation strategy, deterministic seed).
4. **Graph integrity tests** — tests validating deterministic ordering and graph reproducibility with fixed seeds.

**Success criteria:**
- Comparison artifacts are durable, schema-versioned, and reproducible across runs.
- Regression gates are enforced (warn or fail) and auditable.
- Deterministic ordering of graph edges/nodes is validated by tests.

---

## Sprint 3: Telemetry + Performance Defense
**Target subsystem:** telemetry budgets, perf scripts, CI gating.

**Execution tasks (in order):**
1. **Budget enforcement pipeline** — apply telemetry budgets during run validation and propagate failures to CI gates.
2. **Perf script normalization** — update all performance scripts to emit wall time + memory delta with consistent output formats.
3. **Structured telemetry coverage** — ensure execution/governance/training stages emit structured, tag-rich telemetry.
4. **CI budget gate** — wire CI scripts to compare telemetry/perf outputs against thresholds and fail on regression.

**Success criteria:**
- Every perf script reports wall time + memory delta.
- CI fails on telemetry budget or perf regression breaches.
- Telemetry coverage includes execution/governance/training stages with consistent tags.

---

## Sprint 4: Training Orchestration & Artifact Integrity
**Target subsystem:** training governance + checkpoints + replay manifests.

**Execution tasks (in order):**
1. **Signed checkpoints** — add hashing/signing for checkpoints and verification on load.
2. **Replay manifests** — emit deterministic training replay manifests (governance plan, seeds, artifact digests).
3. **Resumable training** — support resume/replay flows that validate manifests and ledger state before continuing.
4. **Artifact integrity tests** — tests that detect tampering and validate replay manifests across stages.

**Success criteria:**
- Checkpoints are tamper-evident with verified signatures/hashes.
- Training can be replayed deterministically from a manifest with ledger validation.
- Integrity tests fail on any mutation of artifacts or manifests.

---

## Sprint 5: Kernel/Autograd Complexity & Performance Optimization
**Target subsystem:** tensor ops + autograd + kernel complexity.

**Execution tasks (in order):**
1. **Broadcast/stride optimizations** — implement stride-based broadcast views and eliminate full materialization where possible.
2. **Autograd copy elimination** — remove unnecessary copies in backward paths and enforce deterministic accumulation ordering.
3. **Parallel kernel strategy** — introduce deterministic parallel kernels where safe, with reproducible reductions.
4. **Complexity benchmarks** — add benchmarks that report wall time + memory delta for critical ops (broadcast, matmul, batch iterators).

**Success criteria:**
- Broadcast and batch operations no longer materialize unnecessary copies.
- Autograd backward paths are free of documented copy hot spots.
- Benchmarks show measurable improvements with deterministic results across runs.

