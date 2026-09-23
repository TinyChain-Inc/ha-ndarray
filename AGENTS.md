# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` with host and GPU backends: `src/host/`, `src/opencl/`, and core types in `src/array.rs`, `src/buffer.rs`, `src/ops/`.
- Library entry: `src/lib.rs` (crate type `rlib`/`cdylib`).
- OpenCL kernels: `src/opencl/programs/`.
- Tests: integration tests in `tests/*.rs` (e.g., `tests/construct.rs`).
- Packaging/build metadata: `Cargo.toml`, `Cargo.lock`; container setup in `Dockerfile`.

## Build, Test, and Development Commands
- Build (CPU/host only): `cargo build`.
- Enable features (e.g., OpenCL): `cargo build --features opencl` or all: `cargo build --features all`.
- Run tests: `cargo test` (host) or explicitly select the installed OpenCL
  device class, for example
  `HA_NDARRAY_OPENCL_DEVICE=GPU cargo test --features opencl`.
- Docs: `cargo doc --no-deps` (add `--open` locally to view).
- Format and lint: `cargo fmt --all` and `cargo clippy --all-targets --all-features -D warnings`.

## Coding Style & Naming Conventions
- Rust 2021 edition; 4-space indentation; keep lines reasonably short.
- Modules/files: `snake_case` (e.g., `array.rs`); types/traits: `PascalCase` (e.g., `ArrayBuf`); functions/fields: `snake_case`.
- Handwritten `unsafe` is prohibited outside the OpenCL enqueue boundary. The
  allowlisted enqueue calls may only submit fully configured kernels or buffer
  commands whose arguments, dimensions, and lifetimes have already been
  validated by the surrounding safe API; no domain logic belongs in those blocks.
- CPU and OpenCL execution must use explicit finite in-flight limits. Acquire
  device capacity before allocating buffers or enqueueing commands, keep
  transfers bounded, and propagate saturation to the caller; do not hide it with
  unbounded host queues or an implicit CPU/GPU fallback.
- Select the platform automatically based on workload size within user-configured
  constraints. Respect the permitted platform type, enabled backends, and configured
  OpenCL device class. An array's incoming platform is not a permanent execution pin.
- At a scheduling boundary, use one selected platform for operation construction,
  prerequisite transforms, and the returned array. Axis reductions select using the
  input element count, which represents the work to consume, not the output size.
- Choose the OpenCL device class before admission within the configured constraints.
  If it is absent, fail unless bootstrap explicitly configured a fallback; never
  broaden the allowed classes opportunistically or reroute because a device is busy.
  Workload-based selection is normal scheduling, not recovery from a capability,
  compilation, execution, or capacity failure.
- Run `cargo fmt` and `cargo clippy` before pushing.

## Testing Guidelines
- Framework: Rust `#[test]` with `cargo test`; integration tests live in `tests/*.rs`.
- Name tests descriptively (e.g., `#[test] fn transpose_concat_validates_dims()`), assert both values and shapes.
- Guard GPU-specific tests with feature flags. Use explicit backend adapters for
  kernel conformance and separate tests for automatic workload-based selection;
  do not change production scheduling to pin conformance tests to a backend.
  A required but unavailable device must fail its test job, not fall back to host.
- Keep tests deterministic and fast; seed randomness when used.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g., "implement Array::concat"), reference issues when relevant (e.g., `(#28)`).
- PRs: include a clear description, motivation, feature flags used (`opencl`, `complex`, etc.), and test coverage notes; add benchmarks only if necessary.
- Required: passing `cargo test`, `cargo fmt`, and `cargo clippy` with no new warnings.

## Security & Configuration Tips
- OpenCL support is optional (`--features opencl`) and requires drivers/ICD on the host; verify with `clinfo`.
- The provided `Dockerfile` can build with GPU support; run with `docker run --gpus=all` when testing OpenCL.
