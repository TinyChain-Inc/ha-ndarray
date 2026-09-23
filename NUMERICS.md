# Numerical behavior

ha-ndarray owns numerical policy for callers and execution backends. The host
and OpenCL implementations obey the same scalar rules. Storage codecs do not
change this contract.

## Supported operations

| Family | Integers | f32/f64 | Complex32/Complex64 |
| --- | --- | --- | --- |
| Storage, geometry, copying, conditional selection | Yes | Yes | Yes |
| Add/subtract/multiply/divide/power, including scalar operands | Yes | Yes | Yes |
| Remainder, ordering, min/max | Yes | Yes | No |
| Array rounding | No | Yes | No |
| Components, argument, conjugation, conjugate transpose | No | No | Yes |
| Absolute value | Same dtype | Same dtype | Corresponding real dtype |
| exp/ln/log and nine trigonometric functions | No | Yes | Yes |
| Equality, boolean operations | u8 output | u8 output | u8 output |
| is_nan/is_inf | No public numeric predicate trait | u8 output | u8 output |
| Cast | All supported destinations | All supported destinations | All supported destinations |
| Sum/product, matrix products, diagonal | Yes | Yes | Yes |
| FFT/IFFT | No | No | Host only |
| Uniform/normal random constructors | No | f32 output only | No |

Integers are i8/i16/i32/i64 and u8/u16/u32/u64. Complex numbers require the complex
feature. Unsupported trait combinations do not compile; unavailable execution
capabilities return Error::Unsupported. OpenCL FFT does not silently copy to a
host implementation: select host execution explicitly.

Transforms and copies preserve payload values. Conditional selection uses nonzero
u8 conditions; unselected branches are not guaranteed to avoid evaluation or
errors. Shapes/dtypes must match where required by the existing API; broadcasting
and casting remain explicit.

## Scalar rules

Integer addition, subtraction, multiplication, absolute value, and nonnegative
powers wrap modulo 2^width. Signed results reinterpret these bits as two's
complement. The absolute value of signed MIN remains MIN. Division truncates
toward zero. Integer division/remainder by zero return zero. MIN/-1 wraps to
MIN; MIN%-1 is zero. Nonzero remainder has the dividend's sign.

Negative integer powers return 1 for base 1, the parity result for base -1, and
zero otherwise, including base zero. 0^0 is 1. Exponents retain their full width,
without floating conversion. Reductions, matrix accumulation, and range
arithmetic use these same rules.

Real floats use nearest-even basic arithmetic and gradual underflow. Preserve
subnormal inputs/outputs, signed zeros, and infinities. Native execution threads
must retain the standard floating-point environment. Overflow and invalid
arithmetic produce IEEE values rather than domain errors or clamping. Nonzero
division by signed zero yields signed infinity; 0/0 yields NaN. NaN payload/sign
are not portable.

Remainder follows Rust % / OpenCL fmod, not Euclidean or IEEE remainder.
The scalar Real::round integer helper is the identity; array rounding remains
float-only. Finite x % infinity is x; infinity % y and x % 0 are NaN. Round uses nearest
integer with ties away from zero and preserves zero sign; abs clears the sign.

ln(±0) is negative infinity; ln of negative real inputs is NaN. Log(x,b) means
ln(x)/ln(b), including exceptional results. Power follows real powf semantics:
x^±0 is 1 even for NaN x, and 1^y is 1 even for NaN y. Negative finite bases with
nonintegral finite exponents produce NaN. Signed-zero/infinity results depend
on exponent sign and odd-integer parity. Other NaN inputs propagate NaN.

Comparisons use IEEE unordered NaN semantics. Min/max propagate any NaN;
among equal signed zeros min selects -0 and max selects +0. Reduction identities
must not replace valid infinities with finite bounds.

Complex operations follow num-complex definitions, exceptional cases, and
principal branches. Branch-cut sides depend on imaginary signed zero. Complex
power with zero exponent is 1+0i. Complex logarithm uses log of the magnitude and
atan2 of the components; complex-base log divides these logarithms.

A number is false exactly when it equals zero. Complex zero requires both
components to be zero. NaN is truthy. Boolean operations and predicates return
exactly u8 0 or 1. Complex is_nan/is_inf inspect either component independently.

## Cast compatibility

Casts preserve number-general's CastFrom<Number> pipeline, not direct C casts:

- f32/Complex32 first convert the real component to i32/u32 for integer
  destinations; f64/Complex64 use i64/u64. This conversion truncates, saturates
  at the intermediate width, and maps NaN to zero, before narrowing.
- Signed-to-unsigned conversion first reinterprets at the source width.
- Unsigned-to-signed conversion first uses the corresponding signed width,
  except u8 first becomes i16.
- Integers through 32 bits first become f32 for float/complex destinations;
  64-bit integers first become f64.
- Complex-to-real takes the real component; real-to-complex supplies zero
  imaginary component. Complex-to-complex converts both components.

Thus -1i8 cast to u64 is 255, u32::MAX cast to i64 is -1, and 16_777_217i32 cast
to f64 is 16_777_216. Regression fixtures pin these compatibility rules.

## Accuracy and aggregates

Basic real arithmetic and individual conversions specified above are correctly
rounded. Finite real transcendental results have an 8-ULP bound. Finite complex
results use componentwise absolute error <=32u*max(1,abs(reference)), where u is
half machine epsilon. Check classification, signed zero, and branches separately.

Elementwise nodes retain separate rounding. Aggregates may reassociate or use
contraction, without bitwise reproducibility. For finite intermediates without
overflow/underflow, use gamma(k)=ku/(1-ku), k=8N for real reductions/dot products
and k=32N for complex equivalents/FFTs. Scale by sums of absolute terms for
sums/dots, the absolute exact product for products, and the input absolute sum
for FFTs. These bounds do not apply for ku>=1 or intermediate range violations.
Extreme-input aggregate classifications can differ with evaluation order.

FFT/IFFT operate independently on each last-axis batch and are unnormalized:
ifft(fft(x)) approximates N*x. Fourier point reads remain unsupported. This
upgrade does not introduce empty-shape support to operations lacking it.

Uniform random constructors return f32 values in [0,1); normal constructors
sample the standard normal distribution with finite Box-Muller inputs. Sequences
need not match across backends or independent evaluations. Range constructors
retain their existing step calculation and apply the scalar rules above.

## Backend requirements and compatibility

Execution platforms are selected automatically from workload size within the
platform type, enabled backends, and user-configured device constraints. Select
once at a scheduling boundary and use that platform consistently for prerequisite
transforms, the operation, and its returned array. Axis reductions use the input
element count. Automatic scheduling does not permit fallback after a numerical
capability or execution failure. Conformance adapters explicitly select the backend
under test independently of the public array scheduler.

OpenCL checks numerical capabilities on the selected device: f64 support,
subnormals, infinities/NaNs, nearest rounding, and correctly rounded f32 division.
Unsupported paths report operation, dtype, device, and required capability.
Compilation is device-specific. Relaxed-math/flush-to-zero options are disabled,
and elementwise contraction is disabled.

Compatibility changes include wide integer powers and zero remainders,
scalar division by zero following the same dtype rules as array division,
NaN-propagating min/max, complex predicates, OpenCL casts/unary operations,
and inverse FFT dispatch. The previously unreadable complex re()/im() outputs
now correctly declare the corresponding real dtype. Other ordinary signatures
and persistent formats are unchanged.

## Conformance references and validation

The shared suite uses explicitly selected host/OpenCL buffers. Development-only
MPFR/MPC directed rounding encloses individual results; composed logarithm and
DFT references propagate bounds through every intermediate operation. Reference
precision starts at 256 bits and doubles through 4096. A correctly rounded
reference is accepted only when both endpoints round directly to the same f32
or f64 value. Unresolved references fail with operation/input diagnostics.
NaNs, infinities, signed zeros, and branch conventions are checked separately.
Integer expectations and finite algebraic sums, products, and dot products use
exact Integer/Rational arithmetic. Number-general cast fixtures remain separate.

Aggregate checks cover f32/f64 and both complex widths, reduction lengths
1, 7, 8, 9, 63, 64, 65, and 129, transformed inputs, multiple axes, and both
keepdims settings. Batched matrix cases include tile boundaries and padding.
Forward and inverse FFTs are checked independently for lengths 1, 3, 8, and 17,
with three distinct complex batches. Each batch has its own error bound;
round-trip bounds include propagated forward error plus inverse error. Reference
rounding never enlarges the specified aggregate tolerances.

Synthetic capability tests cover missing flags, failed queries, complex component
precision, and cast input/output/intermediate precision. These supplement actual
device execution; production still validates the selected queue device before
compilation. Mandatory CPU-OpenCL CI and the manually dispatched `opencl-gpu`
runner execute the same cases. Missing required hardware fails the selected job.

Implementation and native/CPU-OpenCL validation are separate from GPU approval.
Actual GPU conformance remains a mandatory pending gate: CPU OpenCL results do
not establish GPU conformance. A future CubeCL implementation must satisfy the
same contract and shared cases before being advertised as supported.

Validation before the scheduling correction below used PoCL 6.0+debian (OpenCL 3.0, LLVM 18.1.8)
on `cpu-skylake-avx512-AMD Ryzen 7 7840HS w/ Radeon 780M Graphics`, explicitly
selected as a CPU device. The device name does not indicate GPU execution.

| Validation | Debug | Release |
|---|---:|---:|
| Native host, complex enabled, all targets | 57 passed | 57 passed |
| CPU-OpenCL, complex enabled, all targets | 91 passed | 91 passed |

All-feature compilation, Clippy with warnings denied, doctests (no examples),
formatting, and diff checks passed. Actual GPU execution has not been validated.

The axis-reduction scheduler subsequently restored automatic workload-based
selection. Backend conformance also checks explicit reduction adapters so small
OpenCL cases remain device-executed even when the general scheduler chooses host.
