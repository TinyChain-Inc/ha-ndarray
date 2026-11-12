## How to Contribute

We're excited to accept your contributions to ha-ndarray! If you're not sure where to get started,
consider [starting a discussion](https://github.com/TinyChain-Inc/ha-ndarray/discussions)
or [creating an issue](https://github.com/TinyChain-Inc/ha-ndarray/issues).

If you already have some changes you'd like to submit, just send a pull request.
You can find instructions for creating a pull request in
[GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

## Style and contracts

[NUMERICS.md](NUMERICS.md) is the numerical contract; [README.md](README.md)
explains the supported public interfaces. A semantic change updates the contract,
affected guidance/examples, owner tests, and referenced validation commands.

Group imports as standard library, external crates, then internal modules, with
one blank line between groups. Separate items and distinct logical steps with
blank lines. Keep operation construction behind the checked Array APIs.

## Validation

Run the targets that cover the change. Owner tests establish numerical and
ownership behavior; consumer compilation checks affected public APIs. Documentation
changes need link checks and compilation/execution of changed examples. Do not
replace distinct corruption, cancellation, or numerical assertions with broader
consumer tests. Full validation remains required before a release.

Use one shared target directory and sequential Cargo commands with
`CARGO_BUILD_JOBS=1`. Feature configurations share that directory; Cargo separates
their artifacts. Use `CARGO_INCREMENTAL=0` and disable debug symbols when disk
space is limited, retaining ordinary assertions and overflow checks.

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
cargo test --doc --all-features
```

All-feature checks require the OpenCL development loader. For a Host-only change,
select the relevant native features and targets rather than installing a device
runtime to execute unrelated tests. For example:

```sh
cargo test --manifest-path Cargo.toml --test reference test_as_ref -- --exact
cargo test --manifest-path Cargo.toml --lib numerics:: --features complex
cargo test --manifest-path Cargo.toml --release --lib numerics:: --features complex
```

The `tests/unit/` modules contain private backend and numerical fixtures, selected
through `--lib`; public Array tests are separate integration targets. Preserve
normal worker stacks. Compile affected benchmark targets without running timing
campaigns unless measurement is part of the change.

### OpenCL

CPU conformance requires PoCL, the OpenCL development loader, and `m4` for the
native-only MPFR/MPC test oracle. Select the installed device explicitly:

```sh
HA_NDARRAY_OPENCL_DEVICE=CPU cargo test --manifest-path Cargo.toml --lib numerics:: --features opencl,complex
HA_NDARRAY_OPENCL_DEVICE=CPU cargo test --manifest-path Cargo.toml --release --lib numerics:: --features opencl,complex
```

If a Conda toolchain shadows system libraries, set `CC=/usr/bin/cc`,
`CXX=/usr/bin/c++`, `M4=/usr/bin/m4`, and
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc`. Set `POCL_CACHE_DIR`
to writable storage when required. Missing hardware fails a required device job;
it must not produce a skipped pass or Host fallback.

The manually dispatched `opencl-gpu` CI job runs on the GPU runner when `run_gpu`
is enabled. CPU-OpenCL success does not establish GPU conformance; GPU validation
remains a required release gate. The exact accuracy/oracle requirements are in
[NUMERICS.md](NUMERICS.md#conformance-references-and-validation).

### WASM

CI builds `wasm32-unknown-unknown` with default and `wasm,complex,stream` features,
then runs the `wasm` integration target in Node using a source-matched
wasm-bindgen test runner. The setup scripts live in [.github/scripts](.github/scripts).
The MPFR/MPC oracle is native-only. Browser and other WASM target families require
separate runtime validation; Node success does not establish those results.

## Licensing

By contributing code to this project, you represent that you authored the work
(or otherwise have the rights to contribute it), that it is unencumbered by
third-party intellectual property claims, and that you transfer and assign all
right, title, and interest in the contribution to The TinyChain Contributors for
distribution under the Apache 2.0 license. A copy of the license can be found in
the `LICENSE` file in the root directory of the project.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.
