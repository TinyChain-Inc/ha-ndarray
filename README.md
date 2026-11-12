# ha-ndarray

An n-dimensional array library for Rust with Host and optional OpenCL execution.
Operations build lazy, read-only expressions; concrete buffers provide writable
storage. [NUMERICS.md](NUMERICS.md) defines arithmetic, accuracy, shapes, and
backend requirements. [CONTRIBUTING.md](CONTRIBUTING.md) covers validation.

## Getting started

Import the traits for the operations you use. Numerical operations and transforms
consume their receiver; use `as_ref()` to build an expression while retaining its
source.

```rust
use ha_ndarray::{
    range, shape, ArrayBuf, AxisRange, Error, NDArrayRead, NDArrayTransform, NDArrayWrite,
};

# fn main() -> Result<(), Error> {
let source = ArrayBuf::new(vec![1i32, 2, 3, 4], shape![2, 2])?;
let row = source.as_ref().slice(range![AxisRange::At(1), AxisRange::In(0, 2, 1)])?;
assert_eq!(&*row.buffer()?.to_slice()?, &[3, 4]);

let mut copy = row.copy()?;
copy.write_value_at(&[0], 9)?;
assert_eq!(copy.read_value(&[0])?, 9);
assert_eq!(source.read_value(&[1, 0])?, 3);
# Ok(())
# }
```

Shape `[]` is a scalar containing one element; `[0]` is empty. Indexing every axis
and reducing every axis with `keepdims = false` produce scalars. Broadcasting
and dtype conversion are explicit.

## Ownership and evaluation

| Need | Interface |
| --- | --- |
| Inspect or build a scoped expression | `as_ref()` borrows the accessor and owns shape metadata. |
| Retain shared readers | `ArrayAccess::from(array)` stores one shared, read-only accessor. Cloning retains it. |
| Mutate values | Use `NDArrayWrite` on a mutable concrete buffer array. |
| Obtain independent writable storage | `copy()` evaluates and copies the values. |
| Finish an expression or consume a buffer | `into_read()` returns owned storage on the array's current platform. |

Lazy transforms, including slices, are read-only. Concrete buffer arrays do not
implement `Clone`; immutable accessor references and shared `ArrayAccess` arrays
do. Borrowed data cannot outlive its source, and live read borrows prevent writes.
Extracting an owned accessor consumes its array. Native callers handing off a
buffer must stop mutating any retained raw aliases; handoff cannot revoke them.

Consuming reads move owned storage when backend and representation permit it.
Borrowed/shared buffers copy when independent ownership is needed; backend
transfers and small-buffer conversions may allocate. Host FFT can reuse an
eligible owned input workspace. Borrowed and shared FFT inputs remain unchanged.

Fallible array constructors and transforms validate shape products, axes, ranges,
and source cardinality before backend work. Slice, broadcast, transpose, and flip
share validated metadata with their backends. Raw operation constructors and
custom-platform dispatch are internal. Custom native accessors must report
consistent buffer lengths.

## Backends and features

`Host` restricts execution to host memory/CPU, `OpenCL` restricts it to OpenCL,
and the general `Platform` selects by workload size among enabled backends.
Use `ha_ndarray::host::ArrayBuf` for explicitly Host-backed buffers. Converting to
`ArrayAccess` permits subsequent operations to select through the general
platform. Axis reductions select using the input element count, then use that
selection for prerequisite transforms, execution, and the returned array.

With OpenCL enabled, set `HA_NDARRAY_OPENCL_DEVICE` to `CPU`, `GPU`, or
`ACCELERATOR` before platform initialization to constrain device selection.
Otherwise workload thresholds select the device class. A missing device or a
capability, compilation, execution, or capacity failure returns an error; it does
not trigger another backend. Initialization and native OpenCL errors retain their
causes. See [backend requirements](NUMERICS.md#backend-requirements-and-compatibility).

| Feature | Enables |
| --- | --- |
| `complex` | Complex32/Complex64 operations and Host FFT/IFFT. |
| `opencl` | OpenCL execution; requires an installed loader and device driver/ICD. |
| `stream` | destream serialization. |
| `freqfs` | File-guard buffer integration and `stream`; file owners choose their codec. |
| `wasm` | JavaScript entropy for browser/Node random construction; no JavaScript numerical bindings. |
| `all` | `complex`, `freqfs`, `opencl`, and `stream`; excludes `wasm`. |

Without optional features, Host operations are available. Native OpenCL/filesystem
configurations are separate from WASM builds.

## Random construction

`ArrayBuf::random_normal(size)` and `random_uniform(size)` return owned f32
buffers sampled once at construction. Repeated buffer/point reads and shared
expressions observe the same values.

```rust
use ha_ndarray::{ArrayAccess, ArrayBuf, Buffer, Error, NDArrayMath, NDArrayRead};

# fn main() -> Result<(), Error> {
let sample = ArrayBuf::<f32, Buffer<f32>>::random_uniform(16)?;
let shared = ArrayAccess::from(sample);
let difference = shared.clone().sub(shared)?;
assert!(difference.buffer()?.to_slice()?.iter().all(|&value| value == 0.0));
# Ok(())
# }
```

Sampling obtains an eight-byte seed through `getrandom`; provider failures return
`Error::Random` with their cause. Sequences are unspecified across calls and
backends. Empty Host samples require no entropy; explicit OpenCL empty samples
are unsupported. See [distribution and numerical rules](NUMERICS.md#accuracy-and-aggregates).

## WASM and library artifacts

Ordinary builds emit a Rust library. `wasm32-unknown-unknown` supports Host
numerical operations. Enable `wasm` for browser/Node entropy; without it,
nonempty random construction returns `Error::Unsupported`. JavaScript-enabled
modules use wasm-bindgen's loading glue. Native/WASI builds use the host entropy
provider where supported.

For an explicit standalone dynamic library, use
`cargo rustc --lib --features wasm --crate-type cdylib`, adding `--target` when
cross-compiling. Do not unconditionally emit `cdylib` and `rlib` together: Cargo's
unversioned output names can collide when a consumer builds application and test
artifacts with different panic policies. The library leaves panic policy to its
caller.

See [validation guidance](CONTRIBUTING.md#validation) for Host, CPU-OpenCL, GPU,
and Node checks and the boundaries each establishes.
