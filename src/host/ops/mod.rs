use std::f32::consts::PI;
use std::iter;
use std::marker::PhantomData;

use frand::Rand;
use number_general as ng;
use rayon::join;
use rayon::prelude::*;

use crate::access::Access;
use crate::ops::{Concat, Enqueue, FlipSpec, Op, ReadValue, SliceSpec, ViewSpec};
#[cfg(feature = "complex")]
use crate::Complex;
use crate::{
    strides_for, AccessMut, Axes, BufferConverter, Error, Float, Number, Platform, Range, Real,
    Shape, Strides,
};

use super::buffer::Buffer;
use super::platform::{Heap, Host, Stack};
use super::{SliceConverter, StackVec, VEC_MIN_SIZE};

#[cfg(feature = "complex")]
pub mod complex;

macro_rules! host_enqueue {
    ($this:expr, $cond:expr, $t:ty) => {
        if $cond {
            Enqueue::<Stack, $t>::enqueue($this).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, $t>::enqueue($this).map(Buffer::Heap)
        }
    };
}

pub struct Cast<A, IT, OT> {
    access: A,
    dtype: PhantomData<(IT, OT)>,
}

impl<A, IT, OT> Cast<A, IT, OT> {
    pub fn new(access: A) -> Self {
        Self {
            access,
            dtype: PhantomData,
        }
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> Op for Cast<A, IT, OT> {
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> Enqueue<Heap, OT> for Cast<A, IT, OT> {
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .into_par_iter()
                    .copied()
                    .map(|n| n.into())
                    .map(OT::cast_from)
                    .collect()
            })
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> Enqueue<Stack, OT> for Cast<A, IT, OT> {
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .iter()
                    .copied()
                    .map(|n| n.into())
                    .map(OT::cast_from)
                    .collect()
            })
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> Enqueue<Host, OT> for Cast<A, IT, OT> {
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, OT)
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> ReadValue<Host, OT> for Cast<A, IT, OT> {
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access
            .read_value(offset)
            .map(|n| n.into())
            .map(OT::cast_from)
    }
}

pub struct Cond<A, L, R, T> {
    cond: A,
    then: L,
    or_else: R,
    dtype: PhantomData<T>,
}

impl<A, L, R, T> Cond<A, L, R, T> {
    pub fn new(cond: A, then: L, or_else: R) -> Self {
        Self {
            cond,
            then,
            or_else,
            dtype: PhantomData,
        }
    }
}

impl<A, L, R, T> Op for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    fn size(&self) -> usize {
        debug_assert_eq!(self.cond.size(), self.then.size());
        debug_assert_eq!(self.cond.size(), self.or_else.size());
        self.cond.size()
    }
}

impl<A, L, R, T> Enqueue<Stack, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let cond = self.cond.read()?.to_slice()?;
        let then = self.then.read()?.to_slice()?;
        let or_else = self.or_else.read()?.to_slice()?;

        let output = cond
            .iter()
            .copied()
            .zip(then.iter().copied().zip(or_else.iter().copied()))
            .map(|(cond, (then, or_else))| if cond != 0 { then } else { or_else })
            .collect();

        Ok(output)
    }
}

impl<A, L, R, T> Enqueue<Heap, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (cond, (then, or_else)) = join(
            || self.cond.read().and_then(|buf| buf.to_slice()),
            || {
                join(
                    || self.then.read().and_then(|buf| buf.to_slice()),
                    || self.or_else.read().and_then(|buf| buf.to_slice()),
                )
            },
        );

        let (cond, (then, or_else)) = (cond?, (then?, or_else?));

        let output = cond
            .into_par_iter()
            .copied()
            .zip(then.into_par_iter().zip(or_else.into_par_iter()))
            .map(
                |(cond, (then, or_else))| {
                    if cond != 0 {
                        then
                    } else {
                        or_else
                    }
                },
            )
            .copied()
            .collect();

        Ok(output)
    }
}

impl<A, L, R, T> Enqueue<Host, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, T)
    }
}

impl<A, L, R, T> ReadValue<Host, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let (cond, (then, or_else)) = join(
            || self.cond.read_value(offset),
            || {
                join(
                    || self.then.read_value(offset),
                    || self.or_else.read_value(offset),
                )
            },
        );

        let (cond, (then, or_else)) = (cond?, (then?, or_else?));

        if cond != 0 {
            Ok(then)
        } else {
            Ok(or_else)
        }
    }
}

impl<A, T> Enqueue<Host, T> for Concat<A, T>
where
    A: Access<T>,
    T: Number,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut buffer = Vec::with_capacity(self.size());

        for access in self.data() {
            let data = access.read()?.to_slice()?;
            buffer.par_extend(data.into_par_iter().copied());
        }

        Ok(buffer.into())
    }
}

impl<A, T> ReadValue<Host, T> for Concat<A, T>
where
    A: Access<T>,
    T: Number,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        ReadValue::<Platform, T>::read_value(self, offset)
    }
}

pub struct Dual<L, R, IT, OT> {
    left: L,
    right: R,
    zip: fn(IT, IT) -> OT,
}

// arithmetic
impl<L, R, T: Number> Dual<L, R, T, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::add,
        }
    }

    pub fn div(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::div,
        }
    }

    pub fn mul(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::mul,
        }
    }

    pub fn pow(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::pow,
        }
    }

    pub fn rem(left: L, right: R) -> Self
    where
        T: Real,
    {
        Self {
            left,
            right,
            zip: T::rem,
        }
    }

    pub fn sub(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::sub,
        }
    }
}

// floating-point arithmetic
impl<L, R, T: Float> Dual<L, R, T, T> {
    pub fn log(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: T::log,
        }
    }
}

// boolean operations
impl<L, R, T: Number> Dual<L, R, T, u8> {
    pub fn and(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l != T::ZERO && r != T::ZERO { 1 } else { 0 },
        }
    }

    pub fn or(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l != T::ZERO || r != T::ZERO { 1 } else { 0 },
        }
    }

    pub fn xor(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| {
                if (l != T::ZERO) ^ (r != T::ZERO) {
                    1
                } else {
                    0
                }
            },
        }
    }
}

// comparison
impl<L, R, T: Number> Dual<L, R, T, u8> {
    pub fn eq(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l == r { 1 } else { 0 },
        }
    }

    pub fn ne(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l != r { 1 } else { 0 },
        }
    }
}

impl<L, R, T: Number + PartialOrd> Dual<L, R, T, u8> {
    pub fn ge(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l >= r { 1 } else { 0 },
        }
    }

    pub fn gt(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l > r { 1 } else { 0 },
        }
    }

    pub fn le(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l <= r { 1 } else { 0 },
        }
    }

    pub fn lt(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l < r { 1 } else { 0 },
        }
    }
}

impl<L, R, IT, OT> Op for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, IT, OT> Enqueue<Stack, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_slice()?;
        let right = self.right.read()?.to_slice()?;
        exec_dual(self.zip, left, right)
    }
}

impl<L, R, IT, OT> Enqueue<Heap, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<L, R, IT, OT> Enqueue<Host, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, OT)
    }
}

impl<L, R, IT, OT> ReadValue<Host, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        try_join_value(&self.left, &self.right, offset).map(|(l, r)| (self.zip)(l, r))
    }
}

pub struct Flip<A, T> {
    access: A,
    spec: FlipSpec,
    dtype: PhantomData<T>,
}

impl<A, T> Flip<A, T> {
    pub fn new(access: A, shape: Shape, axis: usize) -> Result<Self, Error> {
        FlipSpec::new(shape, axis).map(|spec| Self {
            access,
            spec,
            dtype: PhantomData,
        })
    }
}

impl<A, T> Op for Flip<A, T>
where
    A: Access<T>,
    T: Number,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, T> Enqueue<Heap, T> for Flip<A, T>
where
    A: Access<T>,
    T: Number,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        (0..self.size())
            .into_par_iter()
            .map(|offset| self.read_value(offset))
            .collect()
    }
}

impl<A, T> Enqueue<Stack, T> for Flip<A, T>
where
    A: Access<T>,
    T: Number,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        (0..self.size())
            .map(|offset| self.read_value(offset))
            .collect()
    }
}

impl<A, T> Enqueue<Host, T> for Flip<A, T>
where
    A: Access<T>,
    T: Number,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, T)
    }
}

impl<A, T> ReadValue<Host, T> for Flip<A, T>
where
    A: Access<T>,
    T: Number,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        debug_assert!(offset < self.size());
        let offset = self.spec.source_offset(offset);
        self.access.read_value(offset)
    }
}

pub struct Linear<T> {
    start: T,
    step: T,
    size: usize,
}

impl<T> Linear<T> {
    pub fn new(start: T, step: T, size: usize) -> Self {
        Self { start, step, size }
    }

    #[inline]
    fn value_at(&self, offset: usize) -> T
    where
        T: Number,
    {
        let offset = T::cast_from(ng::Number::from(offset as u64));
        T::add(self.start, T::mul(offset, self.step))
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: Number> Enqueue<Stack, T> for Linear<T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let buffer = (0..self.size).map(|offset| self.value_at(offset)).collect();

        Ok(buffer)
    }
}

impl<T: Number> Enqueue<Heap, T> for Linear<T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let buffer = (0..self.size)
            .into_par_iter()
            .map(|offset| self.value_at(offset))
            .collect();

        Ok(buffer)
    }
}

impl<T: Number> Enqueue<Host, T> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size < VEC_MIN_SIZE, T)
    }
}

impl<T: Number> ReadValue<Host, T> for Linear<T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        Ok(self.value_at(offset))
    }
}

pub struct MatDiag<A, T> {
    access: A,
    dim: usize,
    batch_size: usize,
    dtype: PhantomData<T>,
}

impl<A, T> MatDiag<A, T> {
    pub fn new(access: A, batch_size: usize, dim: usize) -> Self {
        Self {
            access,
            dim,
            batch_size,
            dtype: PhantomData,
        }
    }
}

impl<A: Access<T>, T: Number> Op for MatDiag<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size(), self.batch_size * self.dim * self.dim);
        self.batch_size * self.dim
    }
}

impl<A: Access<T>, T: Number> Enqueue<Heap, T> for MatDiag<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_slice()?;

        let diagonals = input
            .par_chunks_exact(self.dim * self.dim)
            .flat_map(|matrix| {
                matrix
                    .par_chunks_exact(self.dim)
                    .enumerate()
                    .map(|(i, row)| row[i])
            })
            .collect();

        Ok(diagonals)
    }
}

impl<A: Access<T>, T: Number> Enqueue<Stack, T> for MatDiag<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_slice()?;

        let diagonals = input
            .chunks_exact(self.dim * self.dim)
            .flat_map(|matrix| {
                matrix
                    .chunks_exact(self.dim)
                    .enumerate()
                    .map(|(i, row)| row[i])
            })
            .collect();

        Ok(diagonals)
    }
}

impl<A: Access<T>, T: Number> Enqueue<Host, T> for MatDiag<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, T)
    }
}

impl<A: Access<T>, T: Number> ReadValue<Host, T> for MatDiag<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let batch = offset / self.dim;
        let i = offset % self.dim;
        let source_offset = (batch * self.dim * self.dim) + (i * self.dim) + i;
        self.access.read_value(source_offset)
    }
}

pub struct MatMul<L, R, T> {
    left: L,
    right: R,
    batch_size: usize,
    dims: [usize; 3],
    dtype: PhantomData<T>,
}

impl<L, R, T> MatMul<L, R, T> {
    pub fn new(left: L, right: R, dims: [usize; 4]) -> Self {
        let [batch_size, a, b, c] = dims;

        Self {
            left,
            right,
            batch_size,
            dims: [a, b, c],
            dtype: PhantomData,
        }
    }
}

impl<L, R, T> Op for MatMul<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: Send + Sync,
{
    fn size(&self) -> usize {
        self.batch_size * self.dims[0] * self.dims[2]
    }
}

impl<L, R, T> Enqueue<Stack, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_slice()?;
        let right = self.right.read()?.to_slice()?;

        let [a, b, c] = self.dims;

        let mut product = StackVec::with_capacity(self.batch_size * a * c);

        for batch in 0..self.batch_size {
            let l_start = batch * a * b;
            let r_start = batch * b * c;

            for x in 0..a {
                for z in 0..c {
                    let mut sum = T::ZERO;

                    for y in 0..b {
                        let l_offset = l_start + (x * b) + y;
                        let r_offset = r_start + (y * c) + z;
                        sum = T::add(sum, T::mul(left[l_offset], right[r_offset]));
                    }

                    product.push(sum)
                }
            }
        }

        debug_assert_eq!(product.len(), self.size());

        Ok(product)
    }
}

impl<L, R, T> Enqueue<Heap, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let [a, b, c] = self.dims;

        let (left, right) = try_join_read(&self.left, &self.right)?;

        // transpose the right matrices
        let right_size = b * c;
        let right_matrices = right.par_chunks_exact(right_size).map(|right| {
            let mut right_t = vec![T::ZERO; right_size];
            transpose::transpose(right, &mut right_t[..], c, b);
            right_t
        });

        let left_size = a * b;
        let left_matrices = left.par_chunks_exact(left_size);

        let output_size = a * c;
        let mut output = Vec::<T>::with_capacity(self.batch_size * output_size);
        let output_matrices = left_matrices
            .zip(right_matrices)
            .map(|(lm, rm)| {
                let mut out = Vec::<T>::with_capacity(output_size);

                let product = lm
                    .par_chunks_exact(b)
                    .map(|row| {
                        rm.par_chunks_exact(b).map(move |col| {
                            // chunk the dot product to encourage the compiler to vectorize
                            let col = col.par_chunks(8).map(|cc| cc.iter().copied());

                            row.par_chunks(8)
                                .zip(col)
                                .map(|(rc, cc)| {
                                    rc.iter()
                                        .copied()
                                        .zip(cc)
                                        .map(|(r, c)| T::mul(r, c))
                                        .reduce(T::add)
                                        .expect("sum")
                                })
                                .reduce(|| T::ZERO, T::add)
                        })
                    })
                    .flatten();

                out.par_extend(product);
                out
            })
            .flatten();

        output.par_extend(output_matrices);

        debug_assert_eq!(output.len(), self.batch_size * output_size);

        Ok(output)
    }
}

impl<L, R, T> Enqueue<Host, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(
            self,
            self.left.size() < VEC_MIN_SIZE && self.right.size() < VEC_MIN_SIZE,
            T
        )
    }
}

impl<L, R, T> ReadValue<Host, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    fn read_value(&self, _offset: usize) -> Result<T, Error> {
        Err(Error::bounds(
            "reading an individual value from a matrix multiplication is not implemented"
                .to_string(),
        ))
    }
}

pub struct Scalar<A, IT, OT> {
    access: A,
    scalar: IT,
    op: fn(IT, IT) -> OT,
}

impl<A, IT, OT> Scalar<A, IT, OT> {
    fn new(access: A, scalar: IT, op: fn(IT, IT) -> OT) -> Self {
        Self { access, scalar, op }
    }
}

impl<A, T: Number> Scalar<A, T, T> {
    pub fn add(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::add)
    }

    pub fn div(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::div)
    }

    pub fn mul(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::mul)
    }

    pub fn pow(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::pow)
    }

    pub fn rem(access: A, scalar: T) -> Self
    where
        T: Real,
    {
        Self::new(access, scalar, T::rem)
    }

    pub fn sub(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::sub)
    }
}

impl<A, T: Float> Scalar<A, T, T> {
    pub fn log(access: A, scalar: T) -> Self {
        Self::new(access, scalar, T::log)
    }
}

impl<A, T> Scalar<A, T, u8>
where
    T: Number,
{
    pub fn and(access: A, scalar: T) -> Self {
        Self::new(access, scalar, |l, r| {
            if (l != T::ZERO) && (r != T::ZERO) {
                1
            } else {
                0
            }
        })
    }

    pub fn or(access: A, scalar: T) -> Self {
        Self::new(access, scalar, |l, r| {
            if (l != T::ZERO) || (r != T::ZERO) {
                1
            } else {
                0
            }
        })
    }

    pub fn xor(access: A, scalar: T) -> Self {
        Self::new(access, scalar, |l, r| {
            if (l != T::ZERO) ^ (r != T::ZERO) {
                1
            } else {
                0
            }
        })
    }

    pub fn eq(access: A, scalar: T) -> Self {
        Self::new(access, scalar, |l, r| if l == r { 1 } else { 0 })
    }

    pub fn ge(access: A, scalar: T) -> Self
    where
        T: Real,
    {
        Self::new(access, scalar, |l, r| if l >= r { 1 } else { 0 })
    }

    pub fn gt(access: A, scalar: T) -> Self
    where
        T: Real,
    {
        Self::new(access, scalar, |l, r| if l > r { 1 } else { 0 })
    }

    pub fn le(access: A, scalar: T) -> Self
    where
        T: Real,
    {
        Self::new(access, scalar, |l, r| if l <= r { 1 } else { 0 })
    }

    pub fn lt(access: A, scalar: T) -> Self
    where
        T: Real,
    {
        Self::new(access, scalar, |l, r| if l < r { 1 } else { 0 })
    }

    pub fn ne(access: A, scalar: T) -> Self {
        Self::new(access, scalar, |l, r| if l != r { 1 } else { 0 })
    }
}

impl<A, IT, OT> Op for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<Heap, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .map(|l| (self.op)(l, self.scalar))
                    .collect()
            })
    }
}

impl<A, IT, OT> Enqueue<Stack, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .as_ref()
                    .iter()
                    .copied()
                    .map(|l| (self.op)(l, self.scalar))
                    .collect()
            })
    }
}

impl<A, IT, OT> Enqueue<Host, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, OT)
    }
}

impl<A, IT, OT> ReadValue<Host, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access
            .read_value(offset)
            .map(|n| (self.op)(n, self.scalar))
    }
}

pub struct RandomNormal {
    size: usize,
}

impl RandomNormal {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    fn box_muller(u: [f32; 2]) -> [f32; 2] {
        let [u1, u2] = u;
        let r = ((1.0 - u1).ln() * -2.).sqrt();
        let theta = 2. * PI * u2;
        [r * theta.cos(), r * theta.sin()]
    }
}

impl Op for RandomNormal {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap, f32> for RandomNormal {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = Rand::new();

        let mut output = (0..self.size.div_ceil(2))
            .flat_map(|_| Self::box_muller([rng.gen(), rng.gen()]))
            .collect::<Vec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Stack, f32> for RandomNormal {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = Rand::new();

        let mut output = iter::repeat_with(|| [rng.gen(), rng.gen()])
            .take(self.size.div_ceil(2))
            .flat_map(Self::box_muller)
            .collect::<StackVec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Host, f32> for RandomNormal {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size < VEC_MIN_SIZE, f32)
    }
}

impl ReadValue<Host, f32> for RandomNormal {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Err(Error::bounds(
            "cannot calculate an individual value of a random normal distribution".to_string(),
        ))
    }
}

pub struct RandomUniform {
    size: usize,
}

impl RandomUniform {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Op for RandomUniform {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap, f32> for RandomUniform {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = Rand::new();
        Ok((0..self.size).map(|_| rng.gen()).collect())
    }
}

impl Enqueue<Stack, f32> for RandomUniform {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = Rand::new();
        Ok((0..self.size).map(|_| rng.gen()).collect())
    }
}

impl Enqueue<Host, f32> for RandomUniform {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size < VEC_MIN_SIZE, f32)
    }
}

impl ReadValue<Host, f32> for RandomUniform {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Ok(Rand::new().gen())
    }
}

pub struct Reduce<A, T> {
    access: A,
    stride: usize,
    reduce: fn(T, T) -> T,
    id: T,
}

impl<A, T> Reduce<A, T>
where
    T: Number,
{
    pub fn product(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: T::mul,
            id: T::ONE,
        }
    }

    pub fn sum(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: T::add,
            id: T::ZERO,
        }
    }
}

impl<A, T> Reduce<A, T>
where
    T: Real,
{
    pub fn max(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: Real::max,
            id: crate::numeric::minimum::<T>(),
        }
    }

    pub fn min(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: Real::min,
            id: crate::numeric::maximum::<T>(),
        }
    }
}

impl<A: Access<T>, T: Number> Op for Reduce<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size() % self.stride, 0);
        self.access.size() / self.stride
    }
}

impl<A: Access<T>, T: Number> Enqueue<Heap, T> for Reduce<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .chunks_exact(self.stride)
                    .map(|chunk| {
                        chunk
                            // encourage the compiler to vectorize
                            .par_chunks(8)
                            .map(|chunk| {
                                chunk.iter().copied().reduce(self.reduce).expect("reduced")
                            })
                            .reduce(|| self.id, self.reduce)
                    })
                    .collect()
            })
    }
}

impl<A: Access<T>, T: Number> Enqueue<Stack, T> for Reduce<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .chunks_exact(self.stride)
                    .map(|chunk| chunk.iter().copied().reduce(self.reduce).expect("reduced"))
                    .collect()
            })
    }
}

impl<A: Access<T>, T: Number> Enqueue<Host, T> for Reduce<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(
            self,
            self.stride < VEC_MIN_SIZE && self.size() < VEC_MIN_SIZE,
            T
        )
    }
}

impl<A: Access<T>, T: Number> ReadValue<Host, T> for Reduce<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let offset = offset * self.stride;

        if offset < self.access.size() {
            (offset..(offset + self.stride))
                .into_par_iter()
                .map(|offset| self.access.read_value(offset))
                .try_reduce(|| self.id, |r, v| Ok((self.reduce)(r, v)))
        } else {
            Err(Error::bounds(format!(
                "invalid offset {offset} for a reduce op with size {}",
                self.size()
            )))
        }
    }
}

pub struct Slice<A, T> {
    access: A,
    spec: SliceSpec,
    dtype: PhantomData<T>,
}

impl<A, T> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Self {
        let spec = SliceSpec::new(shape, range);

        Self {
            access,
            spec,
            dtype: PhantomData,
        }
    }
}

impl<A: Send + Sync, T: Copy + Send + Sync> Slice<A, T> {
    fn read(&self, source: &[T]) -> Result<StackVec<T>, Error> {
        let output = (0..self.size())
            .map(|offset_out| self.spec.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }

    fn read_parallel(&self, source: &[T]) -> Result<Vec<T>, Error> {
        let output = (0..self.size())
            .into_par_iter()
            .map(|offset_out| self.spec.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }
}

impl<A, T> Slice<A, T>
where
    T: Number,
    A: AccessMut<T>,
{
    fn overwrite<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        if data.len() == self.size() {
            let data = data.to_slice()?;

            for (offset, value) in data.iter().copied().enumerate() {
                let source_offset = self.spec.source_offset(offset);
                self.access.write_value_at(source_offset, value)?;
            }

            Ok(())
        } else {
            Err(Error::bounds(format!(
                "cannot overwrite a slice of size {} with a buffer of size {}",
                self.size(),
                data.len(),
            )))
        }
    }

    fn overwrite_value(&mut self, value: T) -> Result<(), Error> {
        for offset in 0..self.access.size() {
            let source_offset = self.spec.source_offset(offset);
            self.access.write_value_at(source_offset, value)?;
        }

        Ok(())
    }

    fn overwrite_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        let source_offset = self.spec.source_offset(offset);
        self.access.write_value_at(source_offset, value)
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: Number> Enqueue<Heap, T> for Slice<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read_parallel(&buf))
    }
}

impl<A: Access<T>, T: Number> Enqueue<Stack, T> for Slice<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read(&buf))
    }
}

impl<A: Access<T>, T: Number> Enqueue<Host, T> for Slice<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, T)
    }
}

impl<A: Access<T>, T: Number> ReadValue<Host, T> for Slice<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let offset = self.spec.source_offset(offset);
        self.access.read_value(offset)
    }
}

impl<A, T> crate::ops::Write<Heap, T> for Slice<A, T>
where
    T: Number,
    A: AccessMut<T>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.overwrite(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

impl<A, T> crate::ops::Write<Stack, T> for Slice<A, T>
where
    T: Number,
    A: AccessMut<T>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.overwrite(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

impl<A, T> crate::ops::Write<Host, T> for Slice<A, T>
where
    T: Number,
    A: AccessMut<T>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.overwrite(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

pub struct Unary<A, IT, OT> {
    access: A,
    op: fn(IT) -> OT,
}

impl<A: Access<T>, T: Float> Unary<A, T, T> {
    pub fn exp(access: A) -> Self {
        Self { access, op: T::exp }
    }

    pub fn ln(access: A) -> Self {
        Self { access, op: T::ln }
    }
}

impl<A: Access<T>, T: Number> Unary<A, T, T::Abs> {
    pub fn abs(access: A) -> Self {
        Self {
            access,
            op: Number::abs,
        }
    }
}

impl<A: Access<T>, T: Real> Unary<A, T, T> {
    pub fn round(access: A) -> Self {
        Self {
            access,
            op: Real::round,
        }
    }
}

impl<A: Access<T>, T: Float> Unary<A, T, T> {
    pub fn sin(access: A) -> Self {
        Self {
            access,
            op: |n| n.sin(),
        }
    }

    pub fn asin(access: A) -> Self {
        Self {
            access,
            op: |n| n.asin(),
        }
    }

    pub fn sinh(access: A) -> Self {
        Self {
            access,
            op: |n| n.sinh(),
        }
    }

    pub fn cos(access: A) -> Self {
        Self {
            access,
            op: |n| n.cos(),
        }
    }

    pub fn acos(access: A) -> Self {
        Self {
            access,
            op: |n| n.acos(),
        }
    }

    pub fn cosh(access: A) -> Self {
        Self {
            access,
            op: |n| n.cosh(),
        }
    }

    pub fn tan(access: A) -> Self {
        Self {
            access,
            op: |n| n.tan(),
        }
    }

    pub fn atan(access: A) -> Self {
        Self {
            access,
            op: |n| n.atan(),
        }
    }

    pub fn tanh(access: A) -> Self {
        Self {
            access,
            op: |n| n.tanh(),
        }
    }
}

impl<A: Access<T>, T: Number> Unary<A, T, u8> {
    pub fn not(access: A) -> Self {
        Self {
            access,
            op: |n| if n == T::ZERO { 1 } else { 0 },
        }
    }
}

impl<A: Access<T>, T: Float> Unary<A, T, u8> {
    pub fn inf(access: A) -> Self {
        Self {
            access,
            op: |n| if n.is_inf() { 1 } else { 0 },
        }
    }

    pub fn nan(access: A) -> Self {
        Self {
            access,
            op: |n| if n.is_nan() { 1 } else { 0 },
        }
    }
}

#[cfg(feature = "complex")]
impl<A, T> Unary<A, T, T>
where
    A: Access<T>,
    T: Complex,
{
    pub fn conj(access: A) -> Self {
        Self {
            access,
            op: |n| n.conj(),
        }
    }
}

#[cfg(feature = "complex")]
impl<A, T> Unary<A, T, T::Real>
where
    A: Access<T>,
    T: Complex,
{
    pub fn angle(access: A) -> Self {
        Self {
            access,
            op: |n| n.angle(),
        }
    }

    pub fn re(access: A) -> Self {
        Self {
            access,
            op: |n| n.re(),
        }
    }

    pub fn im(access: A) -> Self {
        Self {
            access,
            op: |n| n.im(),
        }
    }
}

impl<A, IT, OT> Op for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<Heap, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.into_par_iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Stack, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Host, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, OT)
    }
}

impl<A, IT, OT> ReadValue<Host, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: Number,
    OT: Number,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access.read_value(offset).map(|n| (self.op)(n))
    }
}

pub struct View<A, T> {
    access: A,
    spec: ViewSpec,
    dtype: PhantomData<T>,
}

impl<A: Access<T>, T: Number> View<A, T> {
    pub fn broadcast(access: A, shape: Shape, broadcast: Shape) -> Self {
        let source_strides = strides_for(&shape, shape.len()).collect();

        Self {
            access,
            spec: ViewSpec::new(broadcast, source_strides),
            dtype: PhantomData,
        }
    }

    pub fn transpose(access: A, shape: Shape, axes: Axes) -> Self {
        let strides = strides_for(&shape, shape.len()).collect::<Strides>();
        let shape = axes.iter().copied().map(|x| shape[x]).collect::<Strides>();
        let source_strides = axes.into_iter().map(|x| strides[x]).collect::<Strides>();

        Self {
            access,
            spec: ViewSpec::new(shape, source_strides),
            dtype: PhantomData,
        }
    }
}

impl<A: Access<T>, T: Number> Op for View<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: Number> Enqueue<Stack, T> for View<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read().and_then(|source| source.to_slice())?;

        let buffer = (0..self.spec.size())
            .map(|offset| self.spec.source_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

impl<A: Access<T>, T: Number> Enqueue<Heap, T> for View<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read().and_then(|source| source.to_slice())?;

        let buffer = (0..self.spec.size())
            .into_par_iter()
            .map(|offset| self.spec.source_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

impl<A: Access<T>, T: Number> Enqueue<Host, T> for View<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        host_enqueue!(self, self.size() < VEC_MIN_SIZE, T)
    }
}

impl<A: Access<T>, T: Number> ReadValue<Host, T> for View<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}

fn exec_dual<IT: Number, OT: Number>(
    zip: fn(IT, IT) -> OT,
    left: SliceConverter<IT>,
    right: SliceConverter<IT>,
) -> Result<StackVec<OT>, Error> {
    let output = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

fn exec_dual_parallel<IT: Number, OT: Number>(
    zip: fn(IT, IT) -> OT,
    left: SliceConverter<IT>,
    right: SliceConverter<IT>,
) -> Result<Vec<OT>, Error> {
    let output = left
        .into_par_iter()
        .copied()
        .zip(right.into_par_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

#[inline]
fn try_join_read<'a, L, R, T>(
    left: &'a L,
    right: &'a R,
) -> Result<(SliceConverter<'a, T>, SliceConverter<'a, T>), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    let (l, r) = join(
        || left.read().and_then(|buf| buf.to_slice()),
        || right.read().and_then(|buf| buf.to_slice()),
    );

    Ok((l?, r?))
}

#[inline]
fn try_join_value<'a, L, R, T>(left: &'a L, right: &'a R, offset: usize) -> Result<(T, T), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    let (l, r) = join(|| left.read_value(offset), || right.read_value(offset));

    Ok((l?, r?))
}
