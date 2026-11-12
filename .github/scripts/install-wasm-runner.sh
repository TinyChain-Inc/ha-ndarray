#!/usr/bin/env bash
set -euo pipefail

# Match the runner's schema to the version resolved for this checkout.
version="$(python3 - <<'PYTHON'
import tomllib
with open("Cargo.lock", "rb") as source:
    lock = tomllib.load(source)
print(next(package["version"] for package in lock["package"] if package["name"] == "wasm-bindgen"))
PYTHON
)"
archive="wasm-bindgen-${version}-x86_64-unknown-linux-musl"
curl --fail --silent --show-error --location \
    "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/${version}/${archive}.tar.gz" \
    -o "$RUNNER_TEMP/wasm-bindgen.tar.gz"
tar -xzf "$RUNNER_TEMP/wasm-bindgen.tar.gz" -C "$RUNNER_TEMP"
rm "$RUNNER_TEMP/wasm-bindgen.tar.gz"
"$RUNNER_TEMP/$archive/wasm-bindgen" --version
printf '%s\n' "$RUNNER_TEMP/$archive" >> "$GITHUB_PATH"
