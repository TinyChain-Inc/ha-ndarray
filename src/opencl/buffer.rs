use std::ops::Deref;

use ocl::Buffer;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::opencl::OpenCL;
use crate::{Error, Number};

impl<T: Number> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        BufferConverter::CL(self.into())
    }

    fn into_buffer(self) -> Result<crate::Buffer<T>, Error> {
        Ok(self.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        if offset < self.len() {
            let slice = self.map().offset(offset).len(1).read();
            // SAFETY: the safe wrapper validates the offset and keeps the mapped buffer
            // alive.
            let value = unsafe { slice.enq()? };
            let value = value.first().copied().expect("value");
            Ok(value)
        } else {
            Err(Error::bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                self.len()
            )))
        }
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Number> BufferMut<T> for Buffer<T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        if data.len() == Buffer::len(self) {
            let data = data.to_cl()?;
            let queue = OpenCL::queue(self.len(), &[data.default_queue(), self.default_queue()])?;
            data.cmd().queue(&queue).copy(self, None, None).enq()?;
            self.set_default_queue(queue);
            Ok(())
        } else {
            Err(Error::bounds(format!(
                "cannot overwrite a buffer of size {} with one of size {}",
                Buffer::len(self),
                data.len()
            )))
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        let buf = Buffer::builder()
            .queue(OpenCL::queue(Buffer::len(self), &[self.default_queue()])?)
            .len(Buffer::len(self))
            .fill_val(value)
            .build()?;

        *self = buf;
        Ok(())
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        if offset < Buffer::len(self) {
            let queue = OpenCL::queue(self.len(), &[self.default_queue()])?;
            let value = [value];
            // The blocking write completes before the temporary host value is released.
            self.cmd()
                .queue(&queue)
                .write(&value[..])
                .offset(offset)
                .enq()?;
            self.set_default_queue(queue);
            Ok(())
        } else {
            Err(Error::bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                Buffer::len(self)
            )))
        }
    }
}

impl<T: Number> BufferInstance<T> for &Buffer<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        BufferConverter::CL((*self).into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(*self, offset)
    }

    fn len(&self) -> usize {
        Buffer::len(self)
    }
}

impl<T: Number> BufferInstance<T> for &mut Buffer<T> {
    fn read(&self) -> BufferConverter<'_, T> {
        BufferConverter::CL((&**self).into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(*self, offset)
    }

    fn len(&self) -> usize {
        Buffer::<T>::len(self)
    }
}

impl<T: Number> BufferMut<T> for &mut Buffer<T> {
    fn write<'b>(&mut self, data: BufferConverter<'b, T>) -> Result<(), Error> {
        BufferMut::write(&mut **self, data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        BufferMut::write_value(&mut **self, value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        BufferMut::write_value_at(&mut **self, offset, value)
    }
}

/// A buffer in OpenCL memory
#[derive(Clone)]
pub enum CLConverter<'a, T: Number> {
    Owned(Buffer<T>),
    Borrowed(&'a Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: Number> CLConverter<'a, T> {
    /// Return this buffer as an owned [`Buffer`].
    /// This will allocate a new [`Buffer`] only if this buffer is borrowed.
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Borrowed(buffer) => {
                let cl_queue = buffer.default_queue().ok_or_else(|| {
                    Error::unsupported("OpenCL buffer has no default queue".to_string())
                })?;
                let copy = Buffer::builder()
                    .queue(cl_queue.clone())
                    .len(buffer.len())
                    .build()?;

                buffer.copy(&copy, None, None).enq()?;

                Ok(copy)
            }
        }
    }

    /// Return the number of elements in this buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(buffer) => Buffer::len(buffer),
            Self::Borrowed(buffer) => Buffer::len(buffer),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: Number> Deref for CLConverter<'a, T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Buffer<T> {
        match self {
            Self::Owned(buffer) => buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

impl<T: Number> From<Buffer<T>> for CLConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        Self::Owned(buf)
    }
}

impl<'a, T: Number> From<&'a Buffer<T>> for CLConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        Self::Borrowed(buf)
    }
}
