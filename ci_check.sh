cargo fmt -- --check && cargo clippy --examples --features=serde -- -D warnings && cargo build --verbose && cargo test --verbose && cargo test --verbose --examples && cargo +nightly-2024-01-16 bench --verbose --features=bench --no-run && cargo build --examples --verbose --features=serde && cargo build --no-default-features && cargo build --features=default,library
