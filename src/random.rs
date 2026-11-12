use crate::Error;

/// Acquire fresh seed material without assuming the host has a system clock.
pub(crate) fn rng() -> Result<frand::Rand, Error> {
    #[cfg(any(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        feature = "wasm"
    ))]
    {
        let mut seed = [0; 8];
        getrandom::fill(&mut seed).map_err(Error::Random)?;
        Ok(frand::Rand::with_seed(u64::from_le_bytes(seed)))
    }
    #[cfg(not(any(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        feature = "wasm"
    )))]
    {
        Err(Error::unsupported(
            "random construction requires an entropy provider; enable the wasm feature for browser/Node hosts".into(),
        ))
    }
}

#[cfg(test)]
#[path = "../tests/unit/random.rs"]
mod tests;
