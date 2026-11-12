#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use ha_ndarray::{
    axes, shape, Error, NDArray, NDArrayMathScalar, NDArrayRead, NDArrayReduce, NDArrayReduceAll,
};
#[cfg(feature = "wasm")]
use ha_ndarray::{ArrayAccess, NDArrayMath, NDArrayWrite};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn scalar_and_heap_operations() -> Result<(), Error> {
    let scalar = ha_ndarray::host::ArrayBuf::new(vec![7i32].into(), shape![])?;
    assert_eq!(scalar.read_value(&[])?, 7);
    let reduced = scalar.sum(axes![], false)?;
    assert!(reduced.shape().is_empty());
    assert_eq!(reduced.read_value(&[])?, 7);

    let array =
        ha_ndarray::host::ArrayBuf::new((0..128i32).collect::<Vec<_>>().into(), shape![128])?;
    assert_eq!(array.add_scalar(1)?.sum_all()?, 8256);

    let singleton =
        ha_ndarray::host::ArrayBuf::new(vec![-0.0f32].into(), shape![1])?.sum(axes![0], false)?;
    assert_eq!(singleton.read_value(&[])?.to_bits(), (-0.0f32).to_bits());
    assert_eq!(
        singleton.buffer()?.to_slice()?[0].to_bits(),
        (-0.0f32).to_bits()
    );

    let heap = ha_ndarray::host::ArrayBuf::new(vec![-0.0f64; 65].into(), shape![65, 1])?
        .sum(axes![1], false)?;
    for i in 0..65 {
        assert_eq!(heap.read_value(&[i])?.to_bits(), (-0.0f64).to_bits());
    }
    assert!(heap
        .buffer()?
        .to_slice()?
        .iter()
        .all(|value| value.to_bits() == (-0.0f64).to_bits()));
    Ok(())
}

#[wasm_bindgen_test]
fn random_construction_respects_the_entropy_feature() -> Result<(), Error> {
    use ha_ndarray::host::ArrayBuf;

    assert_eq!(ArrayBuf::<f32>::random_normal(0)?.size(), 0);
    assert_eq!(ArrayBuf::<f32>::random_uniform(0)?.size(), 0);
    #[cfg(not(feature = "wasm"))]
    {
        for size in [1, 3, 64, 65] {
            assert!(matches!(
                ArrayBuf::<f32>::random_normal(size),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                ArrayBuf::<f32>::random_uniform(size),
                Err(Error::Unsupported(_))
            ));
        }
    }
    #[cfg(feature = "wasm")]
    {
        for size in [1, 3, 64, 65] {
            for array in [
                ArrayBuf::<f32>::random_normal(size)?,
                ArrayBuf::<f32>::random_uniform(size)?,
            ] {
                let expected = array.buffer()?.to_slice()?.into_vec();
                assert!(expected.iter().all(|value| value.is_finite()));
                assert_eq!(&*array.buffer()?.to_slice()?, &expected);
                for (i, value) in expected.iter().enumerate() {
                    assert_eq!(array.read_value(&[i])?.to_bits(), value.to_bits());
                }
                let mut copy = array.copy()?;
                copy.write_value(42.)?;
                assert_eq!(&*array.buffer()?.to_slice()?, &expected);
                let shared = ArrayAccess::from(array);
                assert!(shared
                    .clone()
                    .sub(shared.clone())?
                    .buffer()?
                    .to_slice()?
                    .iter()
                    .all(|&value| value == 0.));
                assert_eq!(
                    &*shared.as_ref().into_read()?.buffer()?.to_slice()?,
                    &expected
                );
                assert_eq!(&*shared.into_read()?.buffer()?.to_slice()?, &expected);
            }
        }
    }
    Ok(())
}
