//! The host platform (i.e. the CPU and main memory)

pub use buffer::*;
pub use platform::*;

use crate::access::AccessBuf;

mod buffer;
pub(crate) mod ops;
mod platform;

pub type ArrayBuf<T> = crate::array::Array<T, AccessBuf<Buffer<T>>, Host>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        shape, slice, AxisRange, Error, MatrixDual, NDArray, NDArrayCompare, NDArrayRead,
        NDArrayReduceBoolean, NDArrayTransform, NDArrayWrite, Shape,
    };

    #[test]
    fn test_matmul_large() -> Result<(), Error> {
        let shapes: Vec<(Shape, Shape, Shape)> = vec![
            (shape![2, 3], shape![3, 4], shape![2, 4]),
            (shape![2, 2, 3], shape![2, 3, 4], shape![2, 2, 4]),
            (shape![9, 7], shape![7, 12], shape![9, 12]),
            (shape![16, 8], shape![8, 24], shape![16, 24]),
            (shape![3, 2, 9], shape![3, 9, 1], shape![3, 2, 1]),
            (shape![2, 15, 26], shape![2, 26, 37], shape![2, 15, 37]),
            (shape![3, 15, 26], shape![3, 26, 37], shape![3, 15, 37]),
            (shape![8, 44, 1], shape![8, 1, 98], shape![8, 44, 98]),
        ];

        for (left_shape, right_shape, output_shape) in shapes {
            let left = vec![1.; left_shape.iter().product()];
            let left = ArrayBuf::new(left.into(), left_shape)?;
            let right = vec![1.; right_shape.iter().product()];
            let right = ArrayBuf::new(right.into(), right_shape)?;

            let expected = *left.shape().last().unwrap();

            let actual = left.matmul(right)?;
            assert_eq!(actual.shape(), output_shape.as_slice());

            let actual = actual.buffer()?.to_slice()?;
            assert!(
                actual.iter().copied().all(|n| n == expected as f32),
                "expected {expected} but found {actual:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_slice() -> Result<(), Error> {
        let array = ArrayBuf::new(vec![0; 6].into(), shape![2, 3])?;
        let slice = array
            .as_ref()
            .slice(slice![AxisRange::In(0, 2, 1), AxisRange::At(1)])?;

        let zeros = ArrayBuf::new(vec![0, 0].into(), shape![2])?;
        let ones = ArrayBuf::new(vec![1, 1].into(), shape![2])?;

        assert!(slice.as_ref().eq(zeros)?.all()?);

        let mut copy = slice.copy()?;
        copy.write(&ones)?;
        assert!(copy.eq(ones)?.all()?);
        assert_eq!(&*array.buffer()?.to_slice()?, &[0; 6]);

        Ok(())
    }
}
