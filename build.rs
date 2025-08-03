use std::env;

fn main() {
    for (name, value) in env::vars() {
        if name.starts_with("DEP_FFMPEG_") {
            if value == "true" {
                println!(
                    r#"cargo:rustc-cfg=feature="{}""#,
                    name["DEP_FFMPEG_".len()..name.len()].to_lowercase()
                );
            }
            println!(
                r#"cargo:rustc-check-cfg=cfg(feature, values("{}"))"#,
                name["DEP_FFMPEG_".len()..name.len()].to_lowercase()
            );
        }
    }

    if env::var("CARGO_CFG_TARGET_ARCH").unwrap().contains("wasm") {
        println!("cargo:rustc-link-arg=-sALLOW_MEMORY_GROWTH=1");
        println!("cargo:rustc-link-arg=--no-entry");
    }
}
