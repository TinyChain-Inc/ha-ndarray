use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

use smallvec::SmallVec;

use super::VEC_MIN_SIZE;
use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::{Error, Number};

/// A stack-allocated buffer.
pub type StackVec<T> = SmallVec<[T; VEC_MIN_SIZE]>;

impl<T: Number> BufferInstance<T> for StackVec<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        self.as_slice().into()
    }

    fn into_buffer(self) -> Result<crate::Buffer<T>, Error> {
        Ok(self.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&self.as_slice(), offset)
    }

    fn len(&self) -> usize {
        StackVec::len(self)
    }
}

impl<T: Number> BufferMut<T> for StackVec<T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.as_mut_slice().write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.as_mut_slice().write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.as_mut_slice().write_value_at(offset, value)
    }
}

impl<T: Number> BufferInstance<T> for Vec<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        self.as_slice().into()
    }

    fn into_buffer(self) -> Result<crate::Buffer<T>, Error> {
        Ok(self.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&self.as_slice(), offset)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T: Number> BufferMut<T> for Vec<T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.as_mut_slice().write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.as_mut_slice().write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.as_mut_slice().write_value_at(offset, value)
    }
}

impl<T: Number> BufferInstance<T> for &[T] {
    fn read(&self) -> BufferConverter<'_, T> {
        (*self).into()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.get(offset).copied().ok_or_else(|| {
            Error::bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                self.len()
            ))
        })
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T: Number> BufferInstance<T> for &mut [T] {
    fn read(&self) -> BufferConverter<'_, T> {
        (&**self).into()
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&&**self, offset)
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T: Number> BufferMut<T> for &mut [T] {
    fn write<'b>(&mut self, data: BufferConverter<'b, T>) -> Result<(), Error> {
        if data.len() == self.len() {
            let data = data.to_slice()?;
            self.copy_from_slice(&data);
            Ok(())
        } else {
            Err(Error::bounds(format!(
                "cannot overwrite a buffer of size {} with one of size {}",
                self.len(),
                data.len()
            )))
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.fill(value);
        Ok(())
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        if offset < self.len() {
            self[offset] = value;
            Ok(())
        } else {
            Err(Error::bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                self.len()
            )))
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Buffer<T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
}

impl<T> get_size::GetSize for Buffer<T> {
    fn get_heap_size(&self) -> usize {
        match self {
            Self::Heap(values) => values.capacity() * std::mem::size_of::<T>(),
            Self::Stack(values) if values.spilled() => values.capacity() * std::mem::size_of::<T>(),
            Self::Stack(_) => 0,
        }
    }
}

impl<T: Clone> Buffer<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Heap(data) => data,
            Self::Stack(data) => data.into_vec(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        match self {
            Self::Heap(data) => data.to_vec(),
            Self::Stack(data) => data.to_vec(),
        }
    }
}

impl<T> Borrow<[T]> for Buffer<T> {
    fn borrow(&self) -> &[T] {
        match self {
            Self::Heap(buf) => buf.borrow(),
            Self::Stack(buf) => buf.borrow(),
        }
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        match self {
            Self::Heap(buf) => buf.as_mut_slice(),
            Self::Stack(buf) => buf.as_mut_slice(),
        }
    }
}

impl<T: Number> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        BufferConverter::Host(self.into())
    }

    fn into_buffer(self) -> Result<crate::Buffer<T>, Error> {
        Ok(self.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        match self {
            Self::Heap(buf) => buf.read_value(offset),
            Self::Stack(buf) => buf.read_value(offset),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Heap(buf) => buf.len(),
            Self::Stack(buf) => buf.len(),
        }
    }
}

impl<T: Number> BufferMut<T> for Buffer<T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        match self {
            Self::Heap(buf) => buf.write(data),
            Self::Stack(buf) => buf.write(data),
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        match self {
            Self::Heap(buf) => buf.write_value(value),
            Self::Stack(buf) => buf.write_value(value),
        }
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        match self {
            Self::Heap(buf) => buf.write_value_at(offset, value),
            Self::Stack(buf) => buf.write_value_at(offset, value),
        }
    }
}

impl<T> From<StackVec<T>> for Buffer<T> {
    fn from(buf: StackVec<T>) -> Self {
        Self::Stack(buf)
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Heap(buf)
    }
}

#[derive(Clone)]
/// A buffer in host memory, either borrowed or owned
pub enum SliceConverter<'a, T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
    Slice(&'a [T]),
}

impl<'a, T> SliceConverter<'a, T> {
    /// Return the number of elements in this buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Heap(vec) => vec.len(),
            Self::Stack(vec) => vec.len(),
            Self::Slice(slice) => slice.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T: Copy> SliceConverter<'a, T> {
    /// Return this buffer as an owned [`Vec`].
    /// This will allocate a new [`Vec`] if this buffer is a [`StackVec`] or borrowed slice.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Heap(vec) => vec,
            Self::Stack(vec) => vec.into_vec(),
            Self::Slice(slice) => slice.to_vec(),
        }
    }

    /// Return this buffer as an owned [`StackVec`].
    pub fn into_stackvec(self) -> StackVec<T> {
        match self {
            Self::Heap(vec) => vec.into(),
            Self::Stack(vec) => vec,
            Self::Slice(slice) => StackVec::from_slice(slice),
        }
    }

    /// Return this buffer as an owned host [`Buffer`].
    pub fn into_buffer(self) -> Buffer<T> {
        match self {
            Self::Heap(vec) => Buffer::Heap(vec),
            Self::Stack(vec) => Buffer::Stack(vec),
            Self::Slice(slice) => {
                if slice.len() < VEC_MIN_SIZE {
                    Buffer::Stack(StackVec::from_slice(slice))
                } else {
                    Buffer::Heap(slice.to_vec())
                }
            }
        }
    }
}

impl<T> From<Buffer<T>> for SliceConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        match buf {
            Buffer::Heap(buf) => SliceConverter::Heap(buf),
            Buffer::Stack(buf) => SliceConverter::Stack(buf),
        }
    }
}

impl<'a, T> From<&'a Buffer<T>> for SliceConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        match buf {
            Buffer::Heap(slice) => slice.as_slice().into(),
            Buffer::Stack(slice) => slice.as_slice().into(),
        }
    }
}

impl<T> From<StackVec<T>> for SliceConverter<'static, T> {
    fn from(vec: StackVec<T>) -> Self {
        Self::Stack(vec)
    }
}

impl<T> From<Vec<T>> for SliceConverter<'static, T> {
    fn from(vec: Vec<T>) -> Self {
        Self::Heap(vec)
    }
}

impl<'a, T> From<&'a [T]> for SliceConverter<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> Deref for SliceConverter<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Heap(data) => data.as_slice(),
            Self::Stack(data) => data.as_slice(),
            Self::Slice(slice) => slice,
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for SliceConverter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}
