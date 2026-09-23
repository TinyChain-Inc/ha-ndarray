#![allow(clippy::type_complexity)]
// The public API is intentionally generic over access/platform types; explicit type aliases here
// tend to obscure the actual bounds without meaningfully improving readability.

use std::fmt;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::BufferInstance;
use crate::ops::*;
use crate::platform::PlatformInstance;
#[cfg(feature = "complex")]
use crate::Complex;
use crate::{
    axes, range_shape, shape, strides_for, ArrayAccess, Axes, AxisRange, BufferConverter, Constant,
    Convert, Error, Float, Number, Platform, Range, Real, Shape,
};

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A: Clone, P: Clone> Clone for Array<T, A, P> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            access: self.access.clone(),
            platform: self.platform.clone(),
            dtype: self.dtype,
        }
    }
}

impl<T, A, P> Array<T, A, P> {
    fn apply<O, OT, Op>(self, op: Op) -> Result<Array<OT, AccessOp<O, P>, P>, Error>
    where
        P: Copy,
        Op: Fn(P, A) -> Result<AccessOp<O, P>, Error>,
    {
        let access = (op)(self.platform, self.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    fn reduce_axes<'a, Op>(
        self,
        mut axes: Axes,
        keepdims: bool,
        op: Op,
    ) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error>
    where
        T: Number,
        A: Access<T>,
        P: Transform<A, T> + ReduceAxes<Accessor<'a, T>, T>,
        Op: Fn(P, Accessor<'a, T>, usize) -> Result<AccessOp<P::Op, P>, Error>,
        Accessor<'a, T>: From<A> + From<AccessOp<P::Transpose, P>>,
    {
        axes.sort();
        axes.dedup();

        let platform = P::select(self.size());
        let shape = reduce_axes(&self.shape, &axes, keepdims)?;
        let stride = axes.iter().copied().map(|x| self.shape[x]).product();

        let access = permute_for_reduce(platform, self.access, self.shape, axes)?;
        let access = (op)(platform, access, stride)?;

        Ok(Array {
            access,
            shape,
            platform,
            dtype: PhantomData,
        })
    }

    pub fn access(&self) -> &A {
        &self.access
    }

    pub fn into_access(self) -> A {
        self.access
    }
}

impl<T, L, P> Array<T, L, P> {
    fn apply_dual<O, OT, R, Op>(
        self,
        other: Array<T, R, P>,
        op: Op,
    ) -> Result<Array<OT, AccessOp<O, P>, P>, Error>
    where
        P: Copy,
        Op: Fn(P, L, R) -> Result<AccessOp<O, P>, Error>,
    {
        let access = (op)(self.platform, self.access, other.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

// constructors
impl<'a, T: Number> Array<T, Accessor<'a, T>, Platform> {
    pub fn from<A, P>(array: Array<T, A, P>) -> Self
    where
        A: Into<Accessor<'a, T>>,
        Platform: From<P>,
    {
        Self {
            shape: array.shape,
            access: array.access.into(),
            platform: array.platform.into(),
            dtype: array.dtype,
        }
    }
}

impl<T, B, P> Array<T, AccessBuf<B>, P>
where
    T: Number,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    fn new_inner(platform: P, buffer: B, shape: Shape) -> Result<Self, Error> {
        if !shape.is_empty() && shape.iter().product::<usize>() == buffer.len() {
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.len(),
            )))
        }
    }

    pub fn convert<'a, FB>(buffer: FB, shape: Shape) -> Result<Self, Error>
    where
        FB: Into<BufferConverter<'a, T>>,
        P: Convert<T, Buffer = B>,
    {
        let buffer = buffer.into();
        let platform = P::select(buffer.len());
        let buffer = platform.convert(buffer)?;
        Self::new_inner(platform, buffer, shape)
    }

    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        let platform = P::select(buffer.len());
        Self::new_inner(platform, buffer, shape)
    }
}

impl<T, P> Array<T, AccessBuf<P::Buffer>, P>
where
    T: Number,
    P: Constant<T>,
{
    pub fn constant(value: T, shape: Shape) -> Result<Self, Error> {
        if !shape.is_empty() {
            let size = shape.iter().product();
            let platform = P::select(size);
            let buffer = platform.constant(value, size)?;
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::bounds(
                "cannot construct an array with an empty shape".to_string(),
            ))
        }
    }
}

// copy constructors
impl<T, A, P> Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: Convert<T>,
{
    pub fn copy(&self) -> Result<Array<T, AccessBuf<P::Buffer>, P>, Error> {
        let buffer = self.buffer().and_then(|buf| self.platform.convert(buf))?;

        Ok(Array {
            shape: self.shape.clone(),
            access: buffer.into(),
            platform: self.platform,
            dtype: self.dtype,
        })
    }
}

// op constructors
impl<T, A, P> Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: Transform<A, T>,
    P: ConstructConcat<AccessOp<<P as Transform<A, T>>::Transpose, P>, T>,
    P: Transform<
        AccessOp<<P as ConstructConcat<AccessOp<<P as Transform<A, T>>::Transpose, P>, T>>::Op, P>,
        T,
    >,
{
    pub fn stack<AS>(arrays: AS, axis: usize) -> Result<Array<T, impl Access<T>, P>, Error>
    where
        AS: IntoIterator<Item = Self>,
    {
        let arrays = arrays
            .into_iter()
            .map(|arr| arr.unsqueeze(axes![axis]))
            .collect::<Result<Vec<_>, Error>>()?;

        Array::transpose_concat(arrays, axis)
    }

    pub fn transpose_concat(
        arrays: Vec<Self>,
        axis: usize,
    ) -> Result<Array<T, impl Access<T>, P>, Error> {
        let shape = if let Some(first) = arrays.first() {
            let shape = first.shape();
            if axis < shape.len() {
                Ok(shape)
            } else {
                Err(Error::bounds(format!("{first:?} has no axis {axis}")))
            }
        } else {
            Err(Error::bounds(
                "cannot concatenate an empty list of arrays".to_string(),
            ))
        }?;

        for array in arrays.iter().skip(1) {
            if array.ndim() == shape.len() {
                for (x, (dim, a_dim)) in shape.iter().zip(array.shape()).enumerate() {
                    if x != axis && dim != a_dim {
                        return Err(Error::bounds(format!(
                            "cannot concatenate {:?} with {:?} at axis {axis}",
                            shape,
                            array.shape()
                        )));
                    }
                }
            } else {
                return Err(Error::bounds(format!(
                    "cannot concatenate {:?} with {:?}",
                    shape,
                    array.shape()
                )));
            }
        }

        let mut permutation: Axes = (0..shape.len()).collect();
        permutation.swap(0, axis);

        let arrays = arrays
            .into_iter()
            .map(|array| array.transpose(permutation.clone()))
            .collect::<Result<Vec<Array<T, _, P>>, Error>>()?;

        Array::concat(arrays)?.transpose(permutation)
    }
}

impl<T, A, P> Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ConstructConcat<A, T>,
{
    pub fn concat(arrays: Vec<Self>) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error> {
        let mut array_iter = arrays.iter();
        let first = array_iter.next();

        if let Some(first) = first {
            let mut shape = Shape::from_slice(first.shape());
            for next in array_iter {
                if next.ndim() != shape.len()
                    || (shape.len() > 1 && shape[1..] != next.shape()[1..])
                {
                    return Err(Error::bounds(format!(
                        "cannot concatenate shapes {:?} and {:?}",
                        shape,
                        next.shape()
                    )));
                } else {
                    shape[0] += next.shape()[0];
                }
            }

            Self::concat_inner(arrays, shape)
        } else {
            Err(Error::bounds(
                "cannot concatenate an empty list of arrays".into(),
            ))
        }
    }

    fn concat_inner(
        arrays: Vec<Array<T, A, P>>,
        shape: Shape,
    ) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error> {
        let platform = P::select(shape.iter().product());

        let data = arrays
            .into_iter()
            .map(|array| array.into_access())
            .collect();

        platform.concat(data).map(|access| Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<T: Number, P: PlatformInstance> Array<T, AccessOp<P::Range, P>, P>
where
    P: ConstructRange<T>,
{
    pub fn range(start: T, stop: T, shape: Shape) -> Result<Self, Error> {
        let size = shape.iter().product();
        let platform = P::select(size);

        platform.range(start, stop, size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Normal, P>, P>
where
    P: Random,
{
    /// Sample finite standard-normal f32 values. Sequences are backend-dependent.
    pub fn random_normal(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_normal(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Uniform, P>, P>
where
    P: Random,
{
    /// Sample uniform f32 values in [0, 1). Sequences are backend-dependent.
    pub fn random_uniform(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_uniform(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

// references
impl<T, A, P> Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: PlatformInstance,
{
    pub fn as_mut<'a, B>(&'a mut self) -> Array<T, B, P>
    where
        A: AccessBorrowMut<'a, T, B>,
        B: AccessMut<T> + 'a,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: AccessBorrowMut::borrow_mut(&mut self.access),
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref<'a, B>(&'a self) -> Array<T, B, P>
    where
        A: AccessBorrow<'a, T, B>,
        B: Access<T> + 'a,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: AccessBorrow::borrow(&self.access),
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// helper methods

impl<'a, T: Number> ArrayAccess<'a, T> {
    pub fn unstack(
        self,
        axis: usize,
    ) -> Result<Vec<Array<T, impl Access<T> + 'a, Platform>>, Error> {
        let dim = self
            .shape()
            .get(axis)
            .copied()
            .ok_or_else(|| Error::bounds(format!("{self:?} has no axis {axis}")))?;

        let prefix = if axis == 0 {
            Range::with_capacity(1)
        } else {
            self.shape
                .iter()
                .take(axis)
                .copied()
                .map(|dim| AxisRange::In(0, dim, 1))
                .collect()
        };

        (0..dim)
            .map(|r| {
                let mut range = prefix.clone();
                range.push(AxisRange::At(r));
                range
            })
            .map(|r| self.clone().slice(r))
            .collect()
    }
}

// traits

/// An n-dimensional array
pub trait NDArray: Send + Sync {
    /// The data type of the elements in this array
    type DType: Number;

    /// The platform used to construct operations on this array.
    type Platform: PlatformInstance;

    /// Return the number of dimensions in this array.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Return the number of elements in this array.
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Borrow the shape of this array.
    fn shape(&self) -> &[usize];
}

impl<T, A, P> NDArray for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: PlatformInstance,
{
    type DType = T;
    type Platform = P;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Array absolute value
pub trait NDArrayAbs: NDArray + Sized {
    /// The return type of the absolute value operation
    type Output: Access<<Self::DType as Number>::Abs>;

    /// Construct an absolute value operation.
    fn abs(
        self,
    ) -> Result<Array<<Self::DType as Number>::Abs, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayAbs for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ElementwiseAbs<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn abs(self) -> Result<Array<T::Abs, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.abs(access))
    }
}

/// Access methods for an [`NDArray`]
pub trait NDArrayRead: NDArray + fmt::Debug + Sized {
    /// Read the value of this [`NDArray`] into a [`BufferConverter`].
    fn buffer(&self) -> Result<BufferConverter<'_, Self::DType>, Error>;

    /// Buffer this [`NDArray`] into a new, owned array, allocating only if needed.
    fn into_read(
        self,
    ) -> Result<
        Array<
            Self::DType,
            AccessBuf<<Self::Platform as Convert<Self::DType>>::Buffer>,
            Self::Platform,
        >,
        Error,
    >
    where
        Self::Platform: Convert<Self::DType>;

    /// Read the value at a specific `coord` in this [`NDArray`].
    fn read_value(&self, coord: &[usize]) -> Result<Self::DType, Error>;
}

impl<T, A, P> NDArrayRead for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: PlatformInstance,
{
    fn buffer(&self) -> Result<BufferConverter<'_, T>, Error> {
        self.access.read()
    }

    fn into_read(self) -> Result<Array<Self::DType, AccessBuf<P::Buffer>, Self::Platform>, Error>
    where
        P: Convert<T>,
    {
        let buffer = self.buffer().and_then(|buf| self.platform.convert(buf))?;
        debug_assert_eq!(buffer.len(), self.size());

        Ok(Array {
            shape: self.shape,
            access: buffer.into(),
            platform: self.platform,
            dtype: self.dtype,
        })
    }

    fn read_value(&self, coord: &[usize]) -> Result<T, Error> {
        valid_coord(coord, self.shape())?;

        let strides = strides_for(self.shape(), self.ndim());

        let offset = coord
            .iter()
            .zip(strides)
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.read_value(offset)
    }
}

/// Access methods for a mutable [`NDArray`]
pub trait NDArrayWrite: NDArray + fmt::Debug + Sized {
    /// Overwrite this [`NDArray`] with the value of the `other` array.
    fn write<O: NDArrayRead<DType = Self::DType>>(&mut self, other: &O) -> Result<(), Error>;

    /// Overwrite this [`NDArray`] with a constant scalar `value`.
    fn write_value(&mut self, value: Self::DType) -> Result<(), Error>;

    /// Write the given `value` at the given `coord` of this [`NDArray`].
    fn write_value_at(&mut self, coord: &[usize], value: Self::DType) -> Result<(), Error>;
}

// write ops
impl<T, A, P> NDArrayWrite for Array<T, A, P>
where
    T: Number,
    A: AccessMut<T>,
    P: PlatformInstance,
{
    fn write<O>(&mut self, other: &O) -> Result<(), Error>
    where
        O: NDArrayRead<DType = Self::DType>,
    {
        same_shape("write", self.shape(), other.shape())?;
        other.buffer().and_then(|buf| self.access.write(buf))
    }

    fn write_value(&mut self, value: Self::DType) -> Result<(), Error> {
        self.access.write_value(value)
    }

    fn write_value_at(&mut self, coord: &[usize], value: Self::DType) -> Result<(), Error> {
        valid_coord(coord, self.shape())?;

        let offset = coord
            .iter()
            .zip(strides_for(self.shape(), self.ndim()))
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.write_value_at(offset, value)
    }
}

// op traits

/// Array cast operations
pub trait NDArrayCast<OT: Number>: NDArray + Sized {
    type Output: Access<OT>;

    /// Construct a new array cast operation.
    fn cast(self) -> Result<Array<OT, Self::Output, Self::Platform>, Error>;
}

impl<IT, OT, A, P> NDArrayCast<OT> for Array<IT, A, P>
where
    IT: Number,
    OT: Number,
    A: Access<IT>,
    P: ElementwiseCast<A, IT, OT>,
{
    type Output = AccessOp<P::Op, P>;

    fn cast(self) -> Result<Array<OT, AccessOp<P::Op, P>, P>, Error> {
        Ok(Array {
            shape: self.shape,
            access: self.platform.cast(self.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

/// Axis-wise array reduce operations
pub trait NDArrayReduce<'a>: NDArray + fmt::Debug {
    type Output: Access<Self::DType> + 'a;

    /// Construct a max-reduce operation over the given `axes`.
    fn max(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct a min-reduce operation over the given `axes`.
    fn min(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct a product-reduce operation over the given `axes`.
    fn product(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a sum-reduce operation over the given `axes`.
    fn sum(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<'a, T, A, P> NDArrayReduce<'a> for Array<T, A, P>
where
    T: Number + 'a,
    A: Access<T> + 'a,
    P: Transform<A, T> + ReduceAxes<Accessor<'a, T>, T> + 'a,
    Accessor<'a, T>: From<A> + From<AccessOp<P::Transpose, P>> + 'a,
{
    type Output = AccessOp<P::Op, P>;

    fn max(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::max(platform, access, stride)
        })
    }

    fn min(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::min(platform, access, stride)
        })
    }

    fn product(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::product(platform, access, stride)
        })
    }

    fn sum(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::sum(platform, access, stride)
        })
    }
}

/// Array transform operations
pub trait NDArrayTransform: NDArray + Sized + fmt::Debug {
    /// The type returned by `broadcast`
    type Broadcast: Access<Self::DType>;

    /// The type returned by `flip`
    type Flip: Access<Self::DType>;

    /// The type returned by `slice`
    type Slice: Access<Self::DType>;

    /// The type returned by `transpose`
    type Transpose: Access<Self::DType>;

    /// Broadcast this array into the given `shape`.
    fn broadcast(
        self,
        shape: Shape,
    ) -> Result<Array<Self::DType, Self::Broadcast, Self::Platform>, Error>;

    fn flip(self, axis: usize) -> Result<Array<Self::DType, Self::Flip, Self::Platform>, Error>;

    /// Reshape this `array`.
    fn reshape(self, shape: Shape) -> Result<Self, Error>;

    /// Construct a slice of this array.
    fn slice(self, range: Range) -> Result<Array<Self::DType, Self::Slice, Self::Platform>, Error>;

    /// Contract the given `axes` of this array.
    /// This will return an error if any of the `axes` have dimension > 1.
    fn squeeze(self, axes: Axes) -> Result<Self, Error>;

    /// Expand the given `axes` of this array.
    fn unsqueeze(self, axes: Axes) -> Result<Self, Error>;

    /// Transpose this array according to the given `permutation`.
    /// If no permutation is given, the array axes will be reversed.
    fn transpose<P: Into<Option<Axes>>>(
        self,
        permutation: P,
    ) -> Result<Array<Self::DType, Self::Transpose, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayTransform for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: Transform<A, T>,
{
    type Broadcast = AccessOp<P::Broadcast, P>;
    type Flip = AccessOp<P::Flip, P>;
    type Slice = AccessOp<P::Slice, P>;
    type Transpose = AccessOp<P::Transpose, P>;

    fn broadcast(self, shape: Shape) -> Result<Array<T, AccessOp<P::Broadcast, P>, P>, Error> {
        if !can_broadcast(self.shape(), &shape) {
            return Err(Error::bounds(format!(
                "cannot broadcast {self:?} into {shape:?}"
            )));
        }

        let platform = P::select(shape.iter().product());
        let broadcast = Shape::from_slice(&shape);
        let access = platform.broadcast(self.access, self.shape, broadcast)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }

    fn flip(self, axis: usize) -> Result<Array<T, AccessOp<P::Flip, P>, P>, Error> {
        let platform = self.platform;
        let access = platform.flip(self.access, self.shape.clone(), axis)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }

    fn reshape(mut self, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == self.size() {
            self.shape = shape;
            Ok(self)
        } else {
            Err(Error::bounds(format!(
                "cannot reshape an array with shape {:?} into {shape:?}",
                self.shape
            )))
        }
    }

    fn slice(self, mut range: Range) -> Result<Array<T, AccessOp<P::Slice, P>, P>, Error> {
        for (dim, range) in self.shape.iter().zip(&range) {
            match range {
                AxisRange::At(i) if i < dim => Ok(()),
                AxisRange::In(start, stop, _step) if start < dim && stop <= dim => Ok(()),
                AxisRange::Of(indices) if indices.iter().all(|i| i < dim) => Ok(()),
                range => Err(Error::bounds(format!(
                    "invalid range {range:?} for dimension {dim}"
                ))),
            }?;
        }

        for dim in self.shape.iter().skip(range.len()).copied() {
            range.push(AxisRange::In(0, dim, 1));
        }

        let shape = range_shape(self.shape(), &range);
        let access = self.platform.slice(self.access, &self.shape, range)?;
        let platform = P::select(shape.iter().product());

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }

    fn squeeze(mut self, mut axes: Axes) -> Result<Self, Error> {
        axes.sort();

        for x in axes.into_iter().rev() {
            if x < self.shape.len() {
                self.shape.remove(x);
            } else {
                return Err(Error::bounds(format!("axis out of bounds: {x}")));
            }
        }

        Ok(self)
    }

    fn unsqueeze(mut self, axes: Axes) -> Result<Self, Error> {
        for x in axes {
            if x <= self.shape.len() {
                self.shape.insert(x, 1);
            } else {
                return Err(Error::bounds(format!("axis out of bounds: {x}")));
            }
        }

        Ok(self)
    }

    fn transpose<PA: Into<Option<Axes>>>(
        self,
        permutation: PA,
    ) -> Result<Array<T, AccessOp<P::Transpose, P>, P>, Error> {
        let permutation = if let Some(axes) = permutation.into() {
            if axes.len() == self.ndim()
                && axes.iter().copied().all(|x| x < self.ndim())
                && !(1..axes.len()).any(|i| axes[i..].contains(&axes[i - 1]))
            {
                Ok(axes)
            } else {
                Err(Error::bounds(format!(
                    "invalid permutation for shape {:?}: {:?}",
                    self.shape, axes
                )))
            }
        } else {
            Ok((0..self.ndim()).rev().collect())
        }?;

        let shape = permutation.iter().copied().map(|x| self.shape[x]).collect();
        let platform = self.platform;
        let access = platform.transpose(self.access, self.shape, permutation)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }
}

/// Unary array operations
pub trait NDArrayUnary: NDArray + Sized {
    /// The return type of a unary operation.
    type Output: Access<Self::DType>;

    /// Construct an exponentiation operation.
    fn exp(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a natural logarithm operation.
    fn ln(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an integer rounding operation.
    fn round(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;
}

impl<T, A, P> NDArrayUnary for Array<T, A, P>
where
    T: Float,
    A: Access<T>,
    P: ElementwiseUnary<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn exp(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.exp(access))
    }

    fn ln(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        P: ElementwiseUnary<A, T>,
    {
        self.apply(|platform, access| platform.ln(access))
    }

    fn round(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.apply(|platform, access| platform.round(access))
    }
}

/// Unary boolean array operations
pub trait NDArrayUnaryBoolean: NDArray + Sized {
    /// The return type of a unary operation.
    type Output: Access<u8>;

    /// Construct a boolean not operation.
    fn not(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayUnaryBoolean for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ElementwiseUnaryBoolean<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn not(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.not(access))
    }
}

/// Boolean array operations
pub trait NDArrayBoolean<O>: NDArray + Sized
where
    O: NDArray<DType = Self::DType>,
{
    type Output: Access<u8>;

    /// Construct a boolean and comparison with the `other` array.
    fn and(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean or comparison with the `other` array.
    fn or(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean xor comparison with the `other` array.
    fn xor(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> NDArrayBoolean<Array<T, R, P>> for Array<T, L, P>
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseBoolean<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn and(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("and", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.and(left, right))
    }

    fn or(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("or", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.or(left, right))
    }

    fn xor(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("xor", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.xor(left, right))
    }
}

/// Boolean array operations with a scalar argument
pub trait NDArrayBooleanScalar: NDArray + Sized {
    type Output: Access<u8>;

    /// Construct a boolean and operation with the `other` value.
    fn and_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean or operation with the `other` value.
    fn or_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean xor operation with the `other` value.
    fn xor_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayBooleanScalar for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ElementwiseBooleanScalar<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn and_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.and_scalar(access, other))
    }

    fn or_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.or_scalar(access, other))
    }

    fn xor_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.xor_scalar(access, other))
    }
}

/// Array comparison operations
pub trait NDArrayCompare<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: Access<u8>;

    /// Elementwise equality comparison
    fn eq(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise greater-than-or-equal comparison
    fn ge(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Elementwise greater-than comparison
    fn gt(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Elementwise less-than-or-equal comparison
    fn le(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Elementwise less-than comparison
    fn lt(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Elementwise not-equal comparison
    fn ne(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> NDArrayCompare<Array<T, R, P>> for Array<T, L, P>
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseCompare<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn eq(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.eq(left, right))
    }

    fn ge(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.ge(left, right))
    }

    fn gt(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.gt(left, right))
    }

    fn le(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.le(left, right))
    }

    fn lt(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.lt(left, right))
    }

    fn ne(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.ne(left, right))
    }
}

/// Array-scalar comparison operations
pub trait NDArrayCompareScalar: NDArray + Sized {
    type Output: Access<u8>;

    /// Construct an equality comparison with the `other` value.
    fn eq_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a greater-than comparison with the `other` value.
    fn gt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct an equal-or-greater-than comparison with the `other` value.
    fn ge_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct a less-than comparison with the `other` value.
    fn lt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct an equal-or-less-than comparison with the `other` value.
    fn le_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct an not-equal comparison with the `other` value.
    fn ne_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayCompareScalar for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ElementwiseCompareScalar<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn eq_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.eq_scalar(access, other))
    }

    fn gt_scalar(self, other: Self::DType) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.apply(|platform, access| platform.gt_scalar(access, other))
    }

    fn ge_scalar(self, other: Self::DType) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.apply(|platform, access| platform.ge_scalar(access, other))
    }

    fn lt_scalar(self, other: Self::DType) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.apply(|platform, access| platform.lt_scalar(access, other))
    }

    fn le_scalar(self, other: Self::DType) -> Result<Array<u8, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        self.apply(|platform, access| platform.le_scalar(access, other))
    }

    fn ne_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.ne_scalar(access, other))
    }
}

#[cfg(feature = "complex")]
/// Complex array properties
pub trait NDArrayComplex: NDArray + Sized
where
    Self::DType: Complex,
{
    type Real: Access<<Self::DType as Complex>::Real>;
    type Complex: Access<Self::DType>;

    /// Calculate the angle in the complex plane elementwise.
    fn angle(
        self,
    ) -> Result<Array<<Self::DType as Complex>::Real, Self::Real, Self::Platform>, Error>;

    /// Return the complex conjugate elementwise.
    fn conj(self) -> Result<Array<Self::DType, Self::Complex, Self::Platform>, Error>;

    /// Return the real part of this array elementwise.
    fn re(self)
        -> Result<Array<<Self::DType as Complex>::Real, Self::Real, Self::Platform>, Error>;

    /// Return the imaginary part of this array elementwise.
    fn im(self)
        -> Result<Array<<Self::DType as Complex>::Real, Self::Real, Self::Platform>, Error>;
}

#[cfg(feature = "complex")]
impl<T, A, P> NDArrayComplex for Array<T, A, P>
where
    T: Complex,
    A: Access<T>,
    P: complex::ElementwiseUnaryComplex<A, T>,
{
    type Real = AccessOp<P::Real, P>;
    type Complex = AccessOp<P::Complex, P>;

    fn angle(self) -> Result<Array<T::Real, Self::Real, Self::Platform>, Error> {
        self.apply(|platform, access| platform.angle(access))
    }

    fn conj(self) -> Result<Array<Self::DType, Self::Complex, Self::Platform>, Error> {
        self.apply(|platform, access| platform.conj(access))
    }

    fn re(
        self,
    ) -> Result<Array<<Self::DType as Complex>::Real, Self::Real, Self::Platform>, Error> {
        self.apply(|platform, access| platform.re(access))
    }

    fn im(
        self,
    ) -> Result<Array<<Self::DType as Complex>::Real, Self::Real, Self::Platform>, Error> {
        self.apply(|platform, access| platform.im(access))
    }
}

#[cfg(feature = "complex")]
/// Fourier transforms
pub trait NDArrayFourier: NDArray + Sized
where
    Self::DType: Complex,
{
    type Output: Access<Self::DType>;

    /// Calculate the Fourier transform of the last dimension of this array.
    fn fft(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Calculate the unnormalized inverse Fourier transform of each last-axis batch.
    fn ifft(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

#[cfg(feature = "complex")]
impl<A, T, P> NDArrayFourier for Array<num_complex::Complex<T>, A, P>
where
    A: Access<num_complex::Complex<T>>,
    num_complex::Complex<T>: Complex,
    P: complex::Fourier<A, num_complex::Complex<T>>,
{
    type Output = AccessOp<P::Op, P>;

    fn fft(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        let dim = self
            .shape
            .last()
            .copied()
            .ok_or_else(|| Error::bounds("a scalar value has no Fourier transform".into()))?;

        self.apply(|platform, access| platform.fft(access, dim))
    }

    fn ifft(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        let dim = self
            .shape
            .last()
            .copied()
            .ok_or_else(|| Error::bounds("a scalar value has no Fourier transform".into()))?;

        self.apply(|platform, access| platform.ifft(access, dim))
    }
}

// TODO: it should be possible to implement this with a different other DType, e.g. C32 * f32
/// Array arithmetic operations
pub trait NDArrayMath<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: Access<Self::DType>;

    /// Construct an addition operation with the given `rhs`.
    fn add(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a division operation with the given `rhs`.
    fn div(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a logarithm operation with the given `base`.
    fn log(self, base: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Float;

    /// Construct a multiplication operation with the given `rhs`.
    fn mul(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an operation to raise these data to the power of the given `exp`onent.
    fn pow(self, exp: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an array subtraction operation with the given `rhs`.
    fn sub(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a modulo operation with the given `rhs`.
    fn rem(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;
}

impl<T, L, R, P> NDArrayMath<Array<T, R, P>> for Array<T, L, P>
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseDual<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn add(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("add", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.add(left, right))
    }

    fn div(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("divide", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.div(left, right))
    }

    fn log(
        self,
        base: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        T: Float,
    {
        same_shape("log", self.shape(), base.shape())?;
        self.apply_dual(base, |platform, left, right| platform.log(left, right))
    }

    fn mul(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("multiply", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.mul(left, right))
    }

    fn pow(
        self,
        exp: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("exponentiate", self.shape(), exp.shape())?;
        self.apply_dual(exp, |platform, left, right| platform.pow(left, right))
    }

    fn sub(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("subtract", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.sub(left, right))
    }

    fn rem(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        T: Real,
    {
        same_shape("remainder", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.rem(left, right))
    }
}

/// Array arithmetic operations with a scalar argument
pub trait NDArrayMathScalar: NDArray + Sized {
    type Output: Access<Self::DType>;

    /// Construct a scalar addition operation.
    fn add_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar division operation.
    fn div_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar logarithm operation.
    fn log_scalar(
        self,
        base: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Float;

    /// Construct a scalar multiplication operation.
    fn mul_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar exponentiation operation.
    fn pow_scalar(
        self,
        exp: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar modulo operation.
    fn rem_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real;

    /// Construct a scalar subtraction operation.
    fn sub_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayMathScalar for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ElementwiseScalar<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn add_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.add_scalar(left, rhs))
    }

    fn div_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.div_scalar(left, rhs))
    }

    fn log_scalar(
        self,
        base: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Float,
    {
        self.apply(|platform, arg| platform.log_scalar(arg, base))
    }

    fn mul_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.mul_scalar(left, rhs))
    }

    fn pow_scalar(
        self,
        exp: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, arg| platform.pow_scalar(arg, exp))
    }

    fn rem_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        Self::DType: Real,
    {
        self.apply(|platform, left| platform.rem_scalar(left, rhs))
    }

    fn sub_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.sub_scalar(left, rhs))
    }
}

/// Float-specific array methods
pub trait NDArrayNumeric: NDArray + Sized
where
    Self::DType: Float,
{
    type Output: Access<u8>;

    /// Test which elements of this array are infinite.
    #[allow(clippy::wrong_self_convention)]
    // This consumes `self` to allow in-place graph construction, consistent with other ndarray ops.
    fn is_inf(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Test which elements of this array are not-a-number.
    #[allow(clippy::wrong_self_convention)]
    // This consumes `self` to allow in-place graph construction, consistent with other ndarray ops.
    fn is_nan(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayNumeric for Array<T, A, P>
where
    T: Float,
    A: Access<T>,
    P: ElementwiseNumeric<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn is_inf(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.is_inf(access))
    }

    fn is_nan(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.is_nan(access))
    }
}

/// Boolean array reduce operations
pub trait NDArrayReduceBoolean: NDArrayRead {
    /// Return `true` if this array contains only non-zero elements.
    fn all(self) -> Result<bool, Error>;

    /// Return `true` if this array contains any non-zero elements.
    fn any(self) -> Result<bool, Error>;
}

impl<T, A, P> NDArrayReduceBoolean for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ReduceAll<A, T>,
{
    fn all(self) -> Result<bool, Error> {
        self.platform.all(self.access)
    }

    fn any(self) -> Result<bool, Error> {
        self.platform.any(self.access)
    }
}

/// Array reduce operations
pub trait NDArrayReduceAll: NDArrayRead {
    /// Return the maximum of all elements in this array.
    fn max_all(self) -> Result<Self::DType, Error>
    where
        Self::DType: Real;

    /// Return the minimum of all elements in this array.
    fn min_all(self) -> Result<Self::DType, Error>
    where
        Self::DType: Real;

    /// Return the product of all elements in this array.
    fn product_all(self) -> Result<Self::DType, Error>;

    /// Return the sum of all elements in this array.
    fn sum_all(self) -> Result<Self::DType, Error>;
}

impl<T, A, P> NDArrayReduceAll for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: ReduceAll<A, T>,
{
    fn max_all(self) -> Result<Self::DType, Error>
    where
        T: Real,
    {
        self.platform.max(self.access)
    }

    fn min_all(self) -> Result<Self::DType, Error>
    where
        T: Real,
    {
        self.platform.min(self.access)
    }

    fn product_all(self) -> Result<Self::DType, Error> {
        self.platform.product(self.access)
    }

    fn sum_all(self) -> Result<T, Error> {
        self.platform.sum(self.access)
    }
}

impl<T, A, P> fmt::Debug for Array<T, A, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a {} array of shape {:?}",
            std::any::type_name::<T>(),
            self.shape
        )
    }
}

/// Array trigonometry methods
pub trait NDArrayTrig: NDArray + Sized {
    type Output: Access<Self::DType>;

    /// Construct a new sine operation.
    fn sin(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new arcsine operation.
    fn asin(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic sine operation.
    fn sinh(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new cos operation.
    fn cos(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new arccosine operation.
    fn acos(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic cosine operation.
    fn cosh(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new tangent operation.
    fn tan(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new arctangent operation.
    fn atan(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic tangent operation.
    fn tanh(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayTrig for Array<T, A, P>
where
    T: Float,
    A: Access<T>,
    P: ElementwiseTrig<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn sin(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.sin(access))
    }

    fn asin(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.asin(access))
    }

    fn sinh(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.sinh(access))
    }

    fn cos(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.cos(access))
    }

    fn acos(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.acos(access))
    }

    fn cosh(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.cosh(access))
    }

    fn tan(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.tan(access))
    }

    fn atan(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.atan(access))
    }

    fn tanh(self) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.tanh(access))
    }
}

/// Conditional selection (boolean logic) methods
pub trait NDArrayWhere<T, L, R>: NDArray<DType = u8> + fmt::Debug
where
    T: Number,
{
    type Output: Access<T>;

    /// Construct a boolean selection operation.
    /// The resulting array will return values from `then` where `self` is `true`
    /// and from `or_else` where `self` is `false`.
    fn cond(self, then: L, or_else: R) -> Result<Array<T, Self::Output, Self::Platform>, Error>;
}

impl<T, A, L, R, P> NDArrayWhere<T, Array<T, L, P>, Array<T, R, P>> for Array<u8, A, P>
where
    T: Number,
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    P: GatherCond<A, L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn cond(
        self,
        then: Array<T, L, P>,
        or_else: Array<T, R, P>,
    ) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        same_shape("cond", self.shape(), then.shape())?;
        same_shape("cond", self.shape(), or_else.shape())?;

        let access = self
            .platform
            .cond(self.access, then.access, or_else.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

/// Matrix dual operations
pub trait MatrixDual<O>: NDArray + fmt::Debug
where
    O: NDArray<DType = Self::DType> + fmt::Debug,
{
    type Output: Access<Self::DType>;

    /// Construct an operation to multiply this matrix or batch of matrices with the `other`.
    fn matmul(self, other: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> MatrixDual<Array<T, R, P>> for Array<T, L, P>
where
    T: Number,
    L: Access<T>,
    R: Access<T>,
    P: LinAlgDual<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn matmul(
        self,
        other: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        let dims = matmul_dims(&self.shape, &other.shape).ok_or_else(|| {
            Error::bounds(format!(
                "invalid dimensions for matrix multiply: {:?} and {:?}",
                self.shape, other.shape
            ))
        })?;

        let mut shape = Shape::with_capacity(self.ndim());
        shape.extend(self.shape.iter().rev().skip(2).rev().copied());
        shape.push(dims[1]);
        shape.push(dims[3]);

        let platform = P::select(dims.iter().product());

        let access = platform.matmul(self.access, other.access, dims)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }
}

/// Matrix unary operations
pub trait MatrixUnary: NDArray + fmt::Debug {
    type Diag: Access<Self::DType>;
    type Transpose: Access<Self::DType>;

    /// Transpose a matrix or batch of matrices (i.e., transpose the last two dimensions).
    fn mt(self) -> Result<Array<Self::DType, Self::Transpose, Self::Platform>, Error>;

    /// Construct an operation to read the diagonal(s) of this matrix or batch of matrices.
    /// This will return an error if the last two dimensions of the batch are unequal.
    fn diag(self) -> Result<Array<Self::DType, Self::Diag, Self::Platform>, Error>;
}

impl<T, A, P> MatrixUnary for Array<T, A, P>
where
    T: Number,
    A: Access<T>,
    P: LinAlgUnary<A, T> + Transform<A, T>,
{
    type Diag = AccessOp<<P as LinAlgUnary<A, T>>::Op, P>;
    type Transpose = AccessOp<<P as Transform<A, T>>::Transpose, P>;

    fn mt(self) -> Result<Array<Self::DType, Self::Transpose, Self::Platform>, Error> {
        let ndim = self.ndim();
        let mut permutation = Axes::with_capacity(ndim);
        permutation.extend(0..self.ndim() - 2);
        permutation.push(ndim - 1);
        permutation.push(ndim - 2);
        self.transpose(permutation)
    }

    fn diag(self) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error> {
        if self.ndim() >= 2 && self.shape.last() == self.shape.iter().nth_back(1) {
            let batch_size = self.shape.iter().rev().skip(2).product();
            let dim = self.shape.last().copied().expect("dim");

            let shape = self.shape.iter().rev().skip(1).rev().copied().collect();
            let platform = P::select(batch_size * dim * dim);
            let access = platform.diag(self.access, batch_size, dim)?;

            Ok(Array {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::bounds(format!(
                "invalid shape for diagonal: {:?}",
                self.shape
            )))
        }
    }
}

#[cfg(feature = "complex")]
/// Complex matrix unary operations
pub trait MatrixUnaryComplex: MatrixUnary
where
    Self::DType: Complex,
{
    type Hermitian: Access<Self::DType>;

    /// Construct the conjugate transpose of a matrix or batch of matrices.
    fn mh(self) -> Result<Array<Self::DType, Self::Hermitian, Self::Platform>, Error>;
}

#[cfg(feature = "complex")]
impl<T, A, P> MatrixUnaryComplex for Array<T, A, P>
where
    T: Complex,
    A: Access<T>,
    P: complex::ElementwiseUnaryComplex<Self::Transpose, T> + LinAlgUnary<A, T> + Transform<A, T>,
{
    type Hermitian = AccessOp<P::Complex, P>;

    fn mh(self) -> Result<Array<Self::DType, Self::Hermitian, Self::Platform>, Error> {
        self.mt().and_then(|array| array.conj())
    }
}

#[inline]
fn can_broadcast(left: &[usize], right: &[usize]) -> bool {
    if left.len() < right.len() {
        return can_broadcast(right, left);
    }

    for (l, r) in left.iter().copied().rev().zip(right.iter().copied().rev()) {
        if l == r || l == 1 || r == 1 {
            // pass
        } else {
            return false;
        }
    }

    true
}

#[inline]
fn matmul_dims(left: &[usize], right: &[usize]) -> Option<[usize; 4]> {
    let mut left = left.iter().copied().rev();
    let mut right = right.iter().copied().rev();

    let b = left.next()?;
    let a = left.next()?;

    let c = right.next()?;
    if right.next()? != b {
        return None;
    }

    let mut batch_size = 1;
    loop {
        match (left.next(), right.next()) {
            (Some(l), Some(r)) if l == r => {
                batch_size *= l;
            }
            (None, None) => break,
            _ => return None,
        }
    }

    Some([batch_size, a, b, c])
}

#[inline]
fn permute_for_reduce<'a, T, A, P>(
    platform: P,
    access: A,
    shape: Shape,
    axes: Axes,
) -> Result<Accessor<'a, T>, Error>
where
    T: Number,
    A: Access<T>,
    P: Transform<A, T>,
    Accessor<'a, T>: From<A> + From<AccessOp<P::Transpose, P>>,
{
    let mut permutation = Axes::with_capacity(shape.len());
    permutation.extend((0..shape.len()).filter(|x| !axes.contains(x)));
    permutation.extend(axes);

    if permutation.iter().copied().enumerate().all(|(i, x)| i == x) {
        Ok(Accessor::from(access))
    } else {
        platform
            .transpose(access, shape, permutation)
            .map(Accessor::from)
    }
}

#[inline]
fn reduce_axes(shape: &[usize], axes: &[usize], keepdims: bool) -> Result<Shape, Error> {
    let mut shape = Shape::from_slice(shape);

    for x in axes.iter().copied().rev() {
        if x >= shape.len() {
            return Err(Error::bounds(format!(
                "axis {x} is out of bounds for {shape:?}"
            )));
        } else if keepdims {
            shape[x] = 1;
        } else {
            shape.remove(x);
        }
    }

    if shape.is_empty() {
        Ok(shape![1])
    } else {
        Ok(shape)
    }
}

#[inline]
pub fn same_shape(op_name: &'static str, left: &[usize], right: &[usize]) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else if can_broadcast(left, right) {
        Err(Error::bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?} (consider broadcasting)"
        )))
    } else {
        Err(Error::bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?}"
        )))
    }
}

#[inline]
fn valid_coord(coord: &[usize], shape: &[usize]) -> Result<(), Error> {
    if coord.len() == shape.len() && coord.iter().zip(shape).all(|(i, dim)| i < dim) {
        return Ok(());
    }

    Err(Error::bounds(format!(
        "invalid coordinate {coord:?} for shape {shape:?}"
    )))
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    #[test]
    fn reduction_reselects_host_for_workload_size() {
        for size in [8, crate::host::VEC_MIN_SIZE] {
            let source =
                crate::host::ArrayBuf::new(vec![1u32; size].into(), shape![2, size / 2]).unwrap();
            let mut source = ArrayAccess::from(source);
            // Simulate an inherited platform chosen for a different workload.
            source.platform = Platform::Host(if size < crate::host::VEC_MIN_SIZE {
                crate::host::Host::Heap(crate::host::Heap)
            } else {
                crate::host::Host::Stack(crate::host::Stack)
            });
            let result = source.sum(axes![0], false).unwrap();
            assert_eq!(result.platform, Platform::select(size));
            assert!(matches!(result.buffer().unwrap(), BufferConverter::Host(_)));
            assert_eq!(
                result.buffer().unwrap().to_slice().unwrap().as_ref(),
                vec![2u32; size / 2]
            );
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn reduction_reselects_backend_and_accessor_together() {
        for size in [8, crate::opencl::GPU_MIN_SIZE] {
            let source =
                crate::host::ArrayBuf::new(vec![1u32; size].into(), shape![2, size / 2]).unwrap();
            let mut source = ArrayAccess::from(source);
            source.platform = if size < crate::opencl::GPU_MIN_SIZE {
                Platform::CL(crate::opencl::OpenCL)
            } else {
                Platform::Host(crate::host::Host::Heap(crate::host::Heap))
            };
            let result = source.sum(axes![0], false).unwrap();
            assert_eq!(result.platform, Platform::select(size));
            let buffer = result.buffer().unwrap();
            assert_eq!(
                matches!(buffer, BufferConverter::CL(_)),
                size >= crate::opencl::GPU_MIN_SIZE
            );
            assert_eq!(buffer.to_slice().unwrap().as_ref(), vec![2u32; size / 2]);
        }
    }
}
