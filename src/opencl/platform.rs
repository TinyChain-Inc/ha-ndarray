use std::sync::Arc;

use number_general as ng;
use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Buffer, Context, Device, DeviceType, Event, Kernel, Platform, Queue};
use rayon::prelude::*;
use smallvec::SmallVec;

use super::ops::*;
use super::{programs, CLElementTrig, CL_PLATFORM, WG_SIZE};
use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::opencl::programs::ElementDual;
use crate::ops::{
    Concat, ConstructConcat, ConstructRange, ElementwiseAbs, ElementwiseBoolean,
    ElementwiseBooleanScalar, ElementwiseCast, ElementwiseCompare, ElementwiseCompareScalar,
    ElementwiseDual, ElementwiseNumeric, ElementwiseScalar, ElementwiseTrig, ElementwiseUnary,
    ElementwiseUnaryBoolean, GatherCond, LinAlgDual, LinAlgUnary, Random, ReduceAll, ReduceAxes,
    Transform,
};
use crate::platform::{Convert, PlatformInstance};
use crate::{Axes, Constant, Error, Float, Number, Range, Real, Shape};

#[cfg(debug_assertions)]
pub const GPU_MIN_SIZE: usize = 128;

#[cfg(not(debug_assertions))]
pub const GPU_MIN_SIZE: usize = 1024; // 1 KiB

#[cfg(debug_assertions)]
pub const ACC_MIN_SIZE: usize = 1024; // 1 KiB

#[cfg(not(debug_assertions))]
pub const ACC_MIN_SIZE: usize = 2_147_483_648; // 1 GiB

#[derive(Clone)]
struct DeviceList {
    devices: Vec<Device>,
    next: Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for DeviceList {
    fn default() -> Self {
        Self {
            devices: Vec::default(),
            next: Arc::new(Default::default()),
        }
    }
}

impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    fn next(&self) -> Option<Device> {
        if self.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

impl From<Vec<Device>> for DeviceList {
    fn from(devices: Vec<Device>) -> Self {
        Self {
            devices,
            next: Arc::new(Default::default()),
        }
    }
}

impl FromIterator<Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<Device>>())
    }
}

pub struct CLPlatform {
    cl_context: Context,
    cl_cpus: DeviceList,
    cl_gpus: DeviceList,
    cl_accs: DeviceList,
    configured_device_type: Option<DeviceType>,
}

impl CLPlatform {
    pub(super) fn default() -> Result<Self, Error> {
        let cl_platform = Platform::first()?;
        Self::try_from(cl_platform).map_err(Error::from)
    }

    fn select_device_type(&self, size_hint: usize) -> DeviceType {
        if let Some(device_type) = self.configured_device_type {
            return device_type;
        }

        if size_hint < GPU_MIN_SIZE {
            DeviceType::CPU
        } else if size_hint < ACC_MIN_SIZE {
            DeviceType::GPU
        } else {
            DeviceType::ACCELERATOR
        }
    }

    fn select_device(&self, device_type: DeviceType) -> Option<Device> {
        match device_type {
            DeviceType::CPU => self.cl_cpus.next(),
            DeviceType::GPU => self.cl_gpus.next(),
            DeviceType::ACCELERATOR => self.cl_accs.next(),
            other => panic!("unsupported OpenCL device type: {other:?}"),
        }
    }
}

impl TryFrom<Platform> for CLPlatform {
    type Error = ocl::Error;

    fn try_from(cl_platform: Platform) -> Result<Self, Self::Error> {
        let devices = Device::list(cl_platform, None)?;
        let cl_context = Context::builder()
            .platform(cl_platform)
            .devices(&devices)
            .build()?;

        let cl_cpus = Device::list(cl_platform, Some(DeviceType::CPU))?;
        let cl_gpus = Device::list(cl_platform, Some(DeviceType::GPU))?;
        let cl_accs = Device::list(cl_platform, Some(DeviceType::ACCELERATOR))?;
        let configured_device_type = match std::env::var("HA_NDARRAY_OPENCL_DEVICE") {
            Ok(value) => match value.to_ascii_lowercase().as_str() {
                "cpu" => Some(DeviceType::CPU),
                "gpu" => Some(DeviceType::GPU),
                "accelerator" => Some(DeviceType::ACCELERATOR),
                _ => {
                    return Err(ocl::Error::from(format!(
                        "invalid HA_NDARRAY_OPENCL_DEVICE {value:?}; expected CPU, GPU, or ACCELERATOR"
                    )));
                }
            },
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => return Err(ocl::Error::from(err.to_string())),
        };

        Ok(Self {
            cl_cpus: cl_cpus.into(),
            cl_gpus: cl_gpus.into(),
            cl_accs: cl_accs.into(),
            cl_context,
            configured_device_type,
        })
    }
}

/// The OpenCL platform
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OpenCL;

impl OpenCL {
    /// Borrow the OpenCL [`Context`] of this platform.
    pub fn context<'a>() -> &'a Context {
        &CL_PLATFORM.cl_context
    }

    /// Copy the given `data` into a new [`Buffer`].
    pub fn copy_into_buffer<T: Number>(data: &[T]) -> Result<Buffer<T>, ocl::Error> {
        let queue = Self::queue(data.len(), &[])?;

        ocl::builders::BufferBuilder::new()
            .len(data.len())
            .queue(queue)
            .copy_host_slice(data)
            .build()
    }

    pub(crate) fn queue(size_hint: usize, options: &[Option<&Queue>]) -> Result<Queue, ocl::Error> {
        let device_type = CL_PLATFORM.select_device_type(size_hint);

        let mut queue = Option::<Queue>::None;
        let mut deps = SmallVec::<[Queue; 3]>::with_capacity(3);

        #[inline]
        fn clone_if_match(
            queue: &Queue,
            device_type: DeviceType,
        ) -> Result<Option<Queue>, ocl::Error> {
            // TODO: is this slow?
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(Some(queue.clone()));
                }
            };

            Ok(None)
        }

        for option in options.iter().filter_map(|q| q.as_ref()) {
            if let Some(q) = clone_if_match(option, device_type)? {
                // Matching device classes do not order independent queues.
                if let Some(previous) = queue.replace(q) {
                    deps.push(previous);
                }
            } else {
                deps.push((*option).clone());
            }
        }

        let queue = if let Some(queue) = queue {
            queue
        } else {
            let device = CL_PLATFORM
                .select_device(device_type)
                .ok_or_else(|| ocl::Error::from(format!("no {device_type:?} device configured")))?;

            Queue::new(&CL_PLATFORM.cl_context, device, None)?
        };

        if !deps.is_empty() {
            let events = deps
                .into_iter()
                .map(|dep| {
                    let event = dep.enqueue_marker::<Event>(None)?;
                    // Submit the producer commands before another queue waits on them.
                    dep.flush()?;

                    Ok(ocl::core::Event::from(event))
                })
                .collect::<Result<SmallVec<[ocl::core::Event; 3]>, ocl::Error>>()?;

            let _ = queue.enqueue_marker(Some(events.as_slice()))?;
        }

        Ok(queue)
    }
}

impl PlatformInstance for OpenCL {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: Number> Constant<T> for OpenCL {
    type Buffer = Buffer<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        let queue = Self::queue(size, &[])?;

        ocl::builders::BufferBuilder::new()
            .len(size)
            .fill_val(value)
            .queue(queue)
            .build()
            .map_err(Error::from)
    }
}

impl<T: Number> Convert<T> for OpenCL {
    type Buffer = Buffer<T>;

    fn convert(&self, buffer: BufferConverter<T>) -> Result<Self::Buffer, Error> {
        buffer
            .to_cl()
            .map_err(Error::from)
            .and_then(|buf| buf.into_buffer())
    }
}

impl<A, T> ConstructConcat<A, T> for OpenCL
where
    A: Access<T>,
    T: Number,
{
    type Op = Concat<A, T>;

    fn concat(self, data: Vec<A>) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Concat::new(data).into())
    }
}

impl<T: Number + PartialOrd> ConstructRange<T> for OpenCL {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        if start <= stop {
            let size_t = T::cast_from(ng::Number::from(size as u64));
            let step = T::div(T::sub(stop, start), size_t);
            Linear::new(start, step, size).map(AccessOp::from)
        } else {
            Err(Error::bounds(format!("invalid range: [{start}, {stop})")))
        }
    }
}

impl<A, L, R, T> GatherCond<A, L, R, T> for OpenCL
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Cond<A, L, R, T>;

    fn cond(self, cond: A, then: L, or_else: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Cond::new(cond, then, or_else).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ElementwiseAbs<A, T> for OpenCL {
    type Op = Unary<A, T, T::Abs>;

    fn abs(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::abs(access).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseBoolean<L, R, T> for OpenCL
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::and(left, right).map(AccessOp::from)
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::or(left, right).map(AccessOp::from)
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::xor(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ElementwiseBooleanScalar<A, T> for OpenCL {
    type Op = Scalar<A, T, u8>;

    fn and_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::and(left, right).map(AccessOp::from)
    }

    fn or_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::and(left, right).map(AccessOp::from)
    }

    fn xor_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::and(left, right).map(AccessOp::from)
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> ElementwiseCast<A, IT, OT> for OpenCL {
    type Op = Cast<A, IT, OT>;

    fn cast(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Cast::new(access).map(AccessOp::from)
    }
}

impl<L, R, T> ElementwiseCompare<L, R, T> for OpenCL
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::eq(left, right).map(AccessOp::from)
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Dual::ge(left, right).map(AccessOp::from)
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Dual::gt(left, right).map(AccessOp::from)
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Dual::le(left, right).map(AccessOp::from)
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Dual::lt(left, right).map(AccessOp::from)
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::ne(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ElementwiseCompareScalar<A, T> for OpenCL {
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::eq(left, right).map(AccessOp::from)
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Scalar::ge(left, right).map(AccessOp::from)
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Scalar::gt(left, right).map(AccessOp::from)
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Scalar::le(left, right).map(AccessOp::from)
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Scalar::lt(left, right).map(AccessOp::from)
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::ne(left, right).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseDual<L, R, T> for OpenCL
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::add(left, right).map(AccessOp::from)
    }

    fn div(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::div(left, right).map(AccessOp::from)
    }

    fn log(self, arg: L, base: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        Dual::log(arg, base).map(AccessOp::from)
    }

    fn mul(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::mul(left, right).map(AccessOp::from)
    }

    fn pow(self, arg: L, exp: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::pow(arg, exp).map(AccessOp::from)
    }

    fn rem(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Dual::rem(left, right).map(AccessOp::from)
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::sub(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ElementwiseScalar<A, T> for OpenCL {
    type Op = Scalar<A, T, T>;

    fn add_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::add(left, right).map(AccessOp::from)
    }

    fn div_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::div(left, right).map(AccessOp::from)
    }

    fn log_scalar(self, arg: A, base: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        Scalar::log(arg, base).map(AccessOp::from)
    }

    fn mul_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::mul(left, right).map(AccessOp::from)
    }

    fn pow_scalar(self, arg: A, exp: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::pow(arg, exp).map(AccessOp::from)
    }

    fn rem_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Scalar::rem(left, right).map(AccessOp::from)
    }

    fn sub_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::sub(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Float> ElementwiseNumeric<A, T> for OpenCL {
    type Op = Unary<A, T, u8>;

    fn is_inf(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::inf(access).map(AccessOp::from)
    }

    fn is_nan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::nan(access).map(AccessOp::from)
    }
}

// TODO: implement this trait separately per-type and remote the CLElementTrig
// boundary
impl<A: Access<T>, T: Float + CLElementTrig> ElementwiseTrig<A, T> for OpenCL {
    type Op = Unary<A, T, T>;

    fn sin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::sin(access).map(AccessOp::from)
    }

    fn asin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::asin(access).map(AccessOp::from)
    }

    fn sinh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::sinh(access).map(AccessOp::from)
    }

    fn cos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::cos(access).map(AccessOp::from)
    }

    fn acos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::acos(access).map(AccessOp::from)
    }

    fn cosh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::cosh(access).map(AccessOp::from)
    }

    fn tan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::tan(access).map(AccessOp::from)
    }

    fn atan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::atan(access).map(AccessOp::from)
    }

    fn tanh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::tanh(access).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Float> ElementwiseUnary<A, T> for OpenCL {
    type Op = Unary<A, T, T>;

    fn exp(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::exp(access).map(AccessOp::from)
    }

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::ln(access).map(AccessOp::from)
    }

    fn round(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Unary::round(access).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ElementwiseUnaryBoolean<A, T> for OpenCL {
    type Op = Unary<A, T, u8>;

    fn not(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::not(access).map(AccessOp::from)
    }
}

#[cfg(feature = "complex")]
impl<A: Access<T>, T: crate::Complex> crate::ops::complex::ElementwiseUnaryComplex<A, T>
    for OpenCL
{
    type Real = Unary<A, T, T::Real>;
    type Complex = Unary<A, T, T>;

    fn angle(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Unary::angle(access).map(AccessOp::from)
    }

    fn conj(self, access: A) -> Result<AccessOp<Self::Complex, Self>, Error> {
        Unary::conj(access).map(AccessOp::from)
    }

    fn re(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Unary::real(access).map(AccessOp::from)
    }

    fn im(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Unary::imag(access).map(AccessOp::from)
    }
}

impl<L, R, T> LinAlgDual<L, R, T> for OpenCL
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = MatMul<L, R, T>;

    fn matmul(
        self,
        left: L,
        right: R,
        dims: [usize; 4],
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        MatMul::new(left, right, dims).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> LinAlgUnary<A, T> for OpenCL {
    type Op = MatDiag<A, T>;

    fn diag(
        self,
        access: A,
        batch_size: usize,
        dim: usize,
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        MatDiag::new(access, batch_size, dim).map(AccessOp::from)
    }
}

impl Random for OpenCL {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        RandomNormal::new(size).map(AccessOp::from)
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        RandomUniform::new(size).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> ReduceAll<A, T> for OpenCL {
    fn all(self, access: A) -> Result<bool, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_and().into_reduction(), T::ONE)?;

        Ok(result.into_par_iter().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_or().into_reduction(), T::ZERO)?;

        Ok(result.into_par_iter().any(|n| n != T::ZERO))
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_max(), crate::numeric::minimum::<T>())?;

        Ok(result
            .into_par_iter()
            .reduce(|| crate::numeric::minimum::<T>(), T::max))
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_min(), crate::numeric::maximum::<T>())?;

        Ok(result
            .into_par_iter()
            .reduce(|| crate::numeric::maximum::<T>(), T::min))
    }

    fn product(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_mul(), T::ONE)?;
        Ok(result.into_par_iter().reduce(|| T::ONE, T::mul))
    }

    fn sum(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all::<T>(&*input, T::cl_add(), T::ZERO)?;
        Ok(result.into_par_iter().reduce(|| T::ZERO, T::add))
    }
}

impl<A: Access<T>, T: Number> ReduceAxes<A, T> for OpenCL {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Reduce::max(access, stride).map(AccessOp::from)
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Reduce::min(access, stride).map(AccessOp::from)
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::product(access, stride).map(AccessOp::from)
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::sum(access, stride).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: Number> Transform<A, T> for OpenCL {
    type Broadcast = View<A, T>;
    type Flip = Flip<A, T>;
    type Slice = Slice<A, T>;
    type Transpose = View<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        View::broadcast(access, shape, broadcast).map(AccessOp::from)
    }

    fn flip(
        self,
        access: A,
        shape: Shape,
        axis: usize,
    ) -> Result<AccessOp<Self::Flip, Self>, Error> {
        Flip::new(access, shape, axis).map(AccessOp::from)
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Slice::new(access, shape, range).map(AccessOp::from)
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        View::transpose(access, shape, permutation).map(AccessOp::from)
    }
}

fn reduce_all<T: Number>(input: &Buffer<T>, reduce: ElementDual, id: T) -> Result<Vec<T>, Error> {
    const MIN_SIZE: usize = 8192;

    let min_size = MIN_SIZE * num_cpus::get();

    if input.len() < min_size {
        let mut result = vec![id; input.len()];
        input.read(result.as_mut_slice()).enq()?;
        return Ok(result);
    }

    let queue = OpenCL::queue(input.len(), &[input.default_queue()])?;

    let program = programs::reduce::reduce(reduce)?;

    let mut buffer = {
        let output = Buffer::<T>::builder()
            .queue(queue.clone())
            .len(input.len().div_ceil(WG_SIZE))
            .fill_val(id)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program.for_queue(&queue)?)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(id)
            .arg(input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        // SAFETY: kernel arguments and dimensions are validated, and all referenced
        // buffers outlive this enqueue.
        unsafe { kernel.enq()? };

        output
    };

    while buffer.len() >= min_size {
        let input = buffer;

        let output = Buffer::<T>::builder()
            .queue(queue.clone())
            .len(input.len().div_ceil(WG_SIZE))
            .fill_val(id)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program.for_queue(&queue)?)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(id)
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        // SAFETY: kernel arguments and dimensions are validated, and all referenced
        // buffers outlive this enqueue.
        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![id; buffer.len()];
    buffer.read(&mut result).enq()?;

    Ok(result)
}

#[cfg(test)]
mod queue_tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn selected_queue_waits_for_other_same_class_producers() -> Result<(), ocl::Error> {
        let producer = OpenCL::queue(2, &[])?;
        let other = Queue::new(OpenCL::context(), producer.device(), None)?;
        let gate = Event::user(OpenCL::context())?;
        let buffer = Buffer::<u32>::builder()
            .queue(producer.clone())
            .len(2)
            .fill_val(0)
            .build()?;
        buffer.cmd().fill(7, None).ewait(&gate).enq()?;
        producer.flush()?;

        let selected = OpenCL::queue(2, &[Some(&producer), Some(&other)])?;
        let (send, recv) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = selected.finish();
            send.send(result).unwrap();
        });
        let early = recv.recv_timeout(Duration::from_secs(1));
        // Always release the producer and join before asserting, even on regression.
        gate.set_complete()?;
        worker.join().unwrap();

        assert!(
            matches!(early, Err(mpsc::RecvTimeoutError::Timeout)),
            "consumer completed before the other input queue's producer: {early:?}"
        );
        recv.recv().unwrap()?;
        let mut values = vec![0; 2];
        buffer.read(&mut values).enq()?;

        assert_eq!(values, vec![7; 2]);
        Ok(())
    }
}
