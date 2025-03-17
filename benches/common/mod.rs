//! Common utilities for benchmarks.

use std::path::{Path, PathBuf};

#[cfg(feature = "ffmpeg")]
use bliss_audio::decoder::ffmpeg::FFmpegDecoder;
#[cfg(feature = "symphonia")]
use bliss_audio::decoder::symphonia::SymphoniaDecoder;
use bliss_audio::{
    decoder::{Decoder, PreAnalyzedSong},
    BlissResult,
};

#[cfg(not(any(feature = "ffmpeg", feature = "symphonia")))]
compile_error!("At least one decoder must be enabled to run the benchmarks.");

/// The Decoder trait currently isn't object-safe,
/// so we can't store a list of `dyn Decoder` objects directly.
///
/// To get around this, we use this struct as a manual vtable.
#[derive(Clone, Copy)]
pub struct DecoderVTable {
    /// The name of this decoder, used for display purposes.
    pub name: &'static str,
    /// The `decode` function for this decoder.
    #[allow(dead_code)]
    pub decode: &'static (dyn (Fn(&Path) -> BlissResult<PreAnalyzedSong>) + Send + Sync),
}

#[cfg(feature = "ffmpeg")]
const FFMPEG_VTABLE: DecoderVTable = DecoderVTable {
    name: "FFmpeg",
    decode: &|path| FFmpegDecoder::decode(path),
};

#[cfg(feature = "symphonia")]
const SYMPHONIA_VTABLE: DecoderVTable = DecoderVTable {
    name: "Symphonia",
    decode: &|path| SymphoniaDecoder::decode(path),
};

pub const DECODERS: &'static [DecoderVTable] = &[FFMPEG_VTABLE, SYMPHONIA_VTABLE];

/// Get the path to the given test file.
pub fn test_file(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
        .canonicalize()
        .unwrap()
}
