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
