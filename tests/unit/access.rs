use std::sync::Arc;

use super::Accessor;
use crate::{shape, ArrayAccess, ArrayBuf, Error};

#[test]
fn accessor_identity_conversion_retains_the_shared_owner() -> Result<(), Error> {
    let array = ArrayAccess::from(ArrayBuf::new(vec![1, 2, 3], shape![3])?);
    let retained = array.clone();
    assert_eq!(Arc::strong_count(&retained.access().0), 2);

    let array = ArrayAccess::from(array);
    assert!(Arc::ptr_eq(&array.access().0, &retained.access().0));
    assert_eq!(Arc::strong_count(&retained.access().0), 2);

    #[expect(
        clippy::useless_conversion,
        reason = "exercise identity erasure without another shared owner"
    )]
    let access = Accessor::from(array.into_access());
    assert!(Arc::ptr_eq(&access.0, &retained.access().0));
    assert_eq!(Arc::strong_count(&retained.access().0), 2);
    drop(access);
    assert_eq!(Arc::strong_count(&retained.access().0), 1);
    Ok(())
}
