use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::ops::ReadOp;
use crate::platform::PlatformInstance;
use crate::{Buffer, Error, Number};

/// A type which allows accessing array data.
///
/// Native implementations must report the length of the buffer returned by `read`.
/// Low-level operations rely on consistent accessor lengths and validated shapes;
/// application input should enter through the fallible array constructors.
pub trait Access<T: Number>: Send + Sync {
    /// Read the data of this accessor as a [`BufferConverter`].
    fn read(&self) -> Result<BufferConverter<'_, T>, Error>;

    /// Consume this accessor, moving owned data when its implementation supports it.
    /// Borrowed or shared accessors produce an independent buffer.
    fn into_buffer(self) -> Result<Buffer<T>, Error>
    where
        Self: Sized,
    {
        self.read()?.into_buffer()
    }

    /// Access a single value.
    fn read_value(&self, offset: usize) -> Result<T, Error>;

    /// Return the data size.
    fn size(&self) -> usize;
}

/// A type which allows accessing array data mutably
pub trait AccessMut<T: Number>: Access<T> + fmt::Debug {
    /// Overwrite these data with the given `data`.
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error>;

    /// Overwrite these data with a single value.
    fn write_value(&mut self, value: T) -> Result<(), Error>;

    /// Overwrite a single value.
    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error>;
}

impl<T, A> Access<T> for &A
where
    T: Number,
    A: Access<T> + ?Sized,
{
    fn read(&self) -> Result<BufferConverter<'_, T>, Error> {
        (**self).read()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        (**self).read_value(offset)
    }

    fn size(&self) -> usize {
        (**self).size()
    }
}

impl<T, A> Access<T> for &mut A
where
    T: Number,
    A: Access<T> + ?Sized,
{
    fn read(&self) -> Result<BufferConverter<'_, T>, Error> {
        (**self).read()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        (**self).read_value(offset)
    }

    fn size(&self) -> usize {
        (**self).size()
    }
}

impl<T, A> AccessMut<T> for &mut A
where
    T: Number,
    A: AccessMut<T> + ?Sized,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        (**self).write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        (**self).write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        (**self).write_value_at(offset, value)
    }
}

/// A struct which provides n-dimensional access to an underlying [`BufferInstance`]
pub struct AccessBuf<B> {
    buffer: B,
}

impl<B: Clone> Clone for AccessBuf<B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

impl<B> AccessBuf<B> {
    /// Borrow the underlying [`BufferInstance`] of this [`AccessBuf`].
    pub fn inner(&self) -> &B {
        &self.buffer
    }

    /// Borrow the underlying [`BufferInstance`] of this [`AccessBuf`] mutably.
    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    /// Destructure this [`AccessBuf`] into its underlying [`BufferInstance`].
    pub fn into_inner(self) -> B {
        self.buffer
    }
}

impl<B> From<B> for AccessBuf<B> {
    fn from(buffer: B) -> Self {
        Self { buffer }
    }
}

impl<T, B> Access<T> for AccessBuf<B>
where
    T: Number,
    B: BufferInstance<T>,
{
    fn read(&self) -> Result<BufferConverter<'_, T>, Error> {
        Ok(self.buffer.read())
    }

    fn into_buffer(self) -> Result<Buffer<T>, Error> {
        self.buffer.into_buffer()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.buffer.read_value(offset)
    }

    fn size(&self) -> usize {
        self.buffer.len()
    }
}

impl<T, B> AccessMut<T> for AccessBuf<B>
where
    T: Number,
    B: BufferMut<T>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.buffer.write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.buffer.write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.buffer.write_value_at(offset, value)
    }
}

impl<B: fmt::Debug> fmt::Debug for AccessBuf<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "access {:?}", self.buffer)
    }
}

/// A struct which provides n-dimensional access to the result of an array operation.
/// A read-only lazy operation. Mutate a concrete buffer or an explicit copy instead.
///
/// ```compile_fail
/// use ha_ndarray::{ArrayBuf, NDArrayTransform, NDArrayWrite, shape, range};
/// let mut source = ArrayBuf::new(vec![0i32; 4], shape![4]).unwrap();
/// let mut view = source.as_ref().slice(range![(0..2).into()]).unwrap();
/// view.write_value(1).unwrap();
/// ```
pub struct AccessOp<O, P> {
    op: O,
    platform: PhantomData<P>,
}

impl<O, P> AccessOp<O, P> {
    /// Convert the given [`AccessOp`] to a more general type of [`PlatformInstance`].
    pub fn wrap<FO, FP>(access: AccessOp<FO, FP>) -> Self
    where
        FO: Into<O>,
        FP: Into<P>,
    {
        Self {
            op: access.op.into(),
            platform: PhantomData,
        }
    }
}

impl<O, P> From<O> for AccessOp<O, P> {
    fn from(op: O) -> Self {
        Self {
            op,
            platform: PhantomData,
        }
    }
}

impl<O, P, T> Access<T> for AccessOp<O, P>
where
    T: Number,
    O: ReadOp<P, T>,
    P: PlatformInstance,
{
    fn read(&self) -> Result<BufferConverter<'static, T>, Error> {
        self.op.enqueue().map(|buffer| buffer.into())
    }

    fn into_buffer(self) -> Result<Buffer<T>, Error> {
        let buffer: BufferConverter<'static, T> = self.op.into_enqueue()?.into();
        buffer.into_buffer()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.op.read_value(offset)
    }

    fn size(&self) -> usize {
        self.op.size()
    }
}

impl<O, P: fmt::Debug> fmt::Debug for AccessOp<O, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "access op {:?} on {:?}",
            std::any::type_name::<O>(),
            self.platform
        )
    }
}

/// A shared, type-erased accessor for independently retained consumers.
/// Clone retains the same accessor; consuming a shared buffer copies its data.
#[derive(Clone)]
pub struct Accessor<'a, T: Number>(Arc<dyn Access<T> + 'a>);

impl<'a, T: Number> Accessor<'a, T> {
    fn new<A: Access<T> + 'a>(access: A) -> Self {
        Self(Arc::new(access))
    }
}

impl<'a, T: Number> Access<T> for Accessor<'a, T> {
    fn read(&self) -> Result<BufferConverter<'_, T>, Error> {
        self.0.read()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.0.read_value(offset)
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<'a, T, B> From<AccessBuf<B>> for Accessor<'a, T>
where
    T: Number,
    B: BufferInstance<T> + 'a,
{
    fn from(access: AccessBuf<B>) -> Self {
        Self::new(access)
    }
}

impl<'a, T, O, P> From<AccessOp<O, P>> for Accessor<'a, T>
where
    T: Number,
    O: ReadOp<P, T> + 'a,
    P: PlatformInstance + 'a,
{
    fn from(access: AccessOp<O, P>) -> Self {
        Self::new(access)
    }
}

impl<'a, T, A> From<&'a A> for Accessor<'a, T>
where
    T: Number,
    A: Access<T> + ?Sized + 'a,
{
    fn from(access: &'a A) -> Self {
        Self::new(access)
    }
}

impl<'a, T, A> From<&'a mut A> for Accessor<'a, T>
where
    T: Number,
    A: Access<T> + ?Sized + 'a,
{
    fn from(access: &'a mut A) -> Self {
        Self::new(access)
    }
}

#[cfg(test)]
#[path = "../tests/unit/access.rs"]
mod tests;
