# ha-ndarray
An n-dimensional array for Rust, with [OpenCL](https://www.khronos.org/opencl/) hardware acceleration
implemented using the [ocl](https://github.com/cogciprocate/ocl) crate.

Use the `opencl` feature flag to enable OpenCL support.

Platform selection is automatic based on workload size within user-configured
constraints. The platform type restricts eligible backends: `Host` selects host
execution, `OpenCL` selects OpenCL execution, and `Platform` chooses between them
when OpenCL is enabled. Converting an array to the general `ArrayAccess` type
allows subsequent scheduling to reselect its backend. Axis reductions select from
the input element count and use that selection for transforms, execution, and
the returned array.

The OpenCL device-class constraint is read during platform initialization. Set
`HA_NDARRAY_OPENCL_DEVICE` to `CPU`, `GPU`, or `ACCELERATOR` to select one class
explicitly; otherwise workload-size thresholds select the class and fail closed
if that class is unavailable. This setting constrains OpenCL device selection;
it does not force every operation on the general `Platform` to use OpenCL.
Selection does not authorize switching to another backend or device after a
capability, compilation, execution, or capacity failure.
For example, run the NVIDIA GPU suite with:

```sh
HA_NDARRAY_OPENCL_DEVICE=GPU cargo test --features opencl
```

OpenCL is a trademark of Apple Inc. used by permission by the Khronos Group. For more information on OpenCL in general, see:

 - [A Gentle Introduction to OpenCL](https://freecontent.manning.com/wp-content/uploads/a-gentle-introduction-to-opencl.pdf) by Matthew Scarpino

 - [The OpenCL C Programming Language](https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html) published by the Khronos Group

 - This excellent overview of OpenCL kernel programming & optimization: https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf

 - A benchmarking tool for comparing numpy, ndarray, and ha-ndarray is available in the `benchmark` branch and can be built there with `cargo run --bin benchmark --features benchmark`.

The `freqfs` feature enables file-guard buffer integration and destream support;
it does not select a filesystem byte codec. File-entry owners implement
`freqfs::FileLoad` and `FileSave` with their chosen codec.

## Numerical compatibility

See [NUMERICS.md](NUMERICS.md) for dtype rules, exceptional values, accuracy limits,
cast compatibility, and unsupported capabilities. Downstream storage libraries
inherit this contract rather than duplicating arithmetic policy.

CPU-OpenCL conformance requires PoCL, the OpenCL development loader, and m4 for
the development-only MPFR/MPC oracle. Run with HA_NDARRAY_OPENCL_DEVICE=CPU and
features opencl,complex. Set CC=/usr/bin/cc, CXX=/usr/bin/c++, M4=/usr/bin/m4,
and CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc when a Conda compiler
shadows system libraries.

Keep native and OpenCL target directories separate (for example target/host-tests
and target/opencl-tests). Set POCL_CACHE_DIR to a writable directory in restricted
environments. GPU conformance uses the manually dispatched opencl-gpu runner;
missing required hardware is a failure, not a skipped pass.

The conformance suite uses certified MPFR/MPC bounds and exact rational aggregate
references; see [the validation contract](NUMERICS.md#conformance-references-and-validation).
Actual GPU validation remains pending and mandatory for full conformance, even
when native and CPU-OpenCL validation pass.
