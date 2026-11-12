#![cfg(feature = "opencl")]

use ha_ndarray::opencl::OpenCL;

#[test]
fn invalid_opencl_configuration_returns_errors() {
    const CHILD: &str = "HA_NDARRAY_INVALID_CONFIG_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let first = OpenCL::context().expect_err("invalid configuration");
        let second = OpenCL::context().expect_err("retained error");
        assert_eq!(first.to_string(), second.to_string());
        assert!(std::error::Error::source(&first).is_some());
        assert!(OpenCL::copy_into_buffer(&[1i32]).is_err());
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "invalid_opencl_configuration_returns_errors",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("HA_NDARRAY_OPENCL_DEVICE", "invalid-device")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
