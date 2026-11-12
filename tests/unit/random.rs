use std::error::Error as _;

use crate::Error;

#[test]
fn entropy_errors_preserve_the_native_cause() {
    let native = getrandom::Error::UNSUPPORTED;
    let error = Error::Random(native);
    for error in [error.clone(), error] {
        assert!(matches!(error, Error::Random(cause) if cause == native));
        assert_eq!(error.source().unwrap().to_string(), native.to_string());
        assert!(error
            .source()
            .unwrap()
            .downcast_ref::<getrandom::Error>()
            .is_some());
        assert!(error.to_string().contains(&native.to_string()));
        assert!(format!("{error:?}").contains(&native.to_string()));
    }
}
