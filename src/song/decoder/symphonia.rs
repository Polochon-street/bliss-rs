//! Decoder implementation that uses the `symphonia` crate to decode audio files, and the `rubato` crate to resample the audio files.
//!
//! Upstreamed from the `mecomp-analysis` crate.

use std::{fs::File, time::Duration};

// use rodio::Source;
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use symphonia::{
    core::{
        audio::{AudioBufferRef, SampleBuffer, SignalSpec},
        codecs::{DecoderOptions, CODEC_TYPE_NULL},
        errors::Error,
        formats::{FormatOptions, FormatReader},
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
        units,
    },
    default::get_probe,
};
use thiserror::Error;

use crate::{BlissError, BlissResult, SAMPLE_RATE};

use super::{Decoder, PreAnalyzedSong};

#[derive(Debug, Error, PartialEq, Eq, Clone)]
/// Error raised when trying to decode a song with the SymphoniaDecoder.
pub enum SymphoniaDecoderError {
    #[error("Failed to resample audio: {0}")]
    /// Error raised when trying to resample audio.
    /// (from rubato)
    ResampleError(String),
    #[error("Failed to create resampler: {0}")]
    /// Error raised when trying to create a resampler.
    /// (from rubato)
    ResamplerConstructionError(String),
    #[error("IO Error: {0}")]
    /// General IO error.
    IoError(String),
    #[error("Failed to decode audio: {0}")]
    /// Error raised when trying to decode audio.
    /// (from symphonia)
    DecodeError(String),
    #[error("Unsupported codec")]
    /// Error raised when trying to decode a file with an unsupported codec.
    UnsupportedCodec,
    #[error("No supported audio tracks")]
    /// Error raised when trying to decode a file with no supported audio tracks.
    NoSupportedAudioTracks,
    #[error("No streams")]
    /// Error raised when trying to decode a file with no streams.
    NoStreams,
    #[error("The audio source's duration is either unknown or infinite")]
    /// Error raised when the audio source's duration is either unknown or infinite.
    IndeterminantDuration,
}

impl From<rubato::ResampleError> for SymphoniaDecoderError {
    fn from(err: rubato::ResampleError) -> Self {
        Self::ResampleError(err.to_string())
    }
}
impl From<rubato::ResamplerConstructionError> for SymphoniaDecoderError {
    fn from(err: rubato::ResamplerConstructionError) -> Self {
        Self::ResamplerConstructionError(err.to_string())
    }
}
impl From<std::io::Error> for SymphoniaDecoderError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}
impl From<Error> for SymphoniaDecoderError {
    fn from(err: Error) -> Self {
        Self::DecodeError(err.to_string())
    }
}

impl Into<BlissError> for SymphoniaDecoderError {
    fn into(self) -> BlissError {
        BlissError::DecodingError(self.to_string())
    }
}

const MAX_DECODE_RETRIES: usize = 3;

#[allow(clippy::module_name_repetitions)]
pub(crate) struct SymphoniaDecoder {
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    current_span_offset: usize,
    format: Box<dyn FormatReader>,
    total_duration: Option<Duration>,
    buffer: SampleBuffer<f32>,
    spec: SignalSpec,
}

impl SymphoniaDecoder {
    pub fn new(mss: MediaSourceStream) -> Result<Self, SymphoniaDecoderError> {
        match SymphoniaDecoder::init(mss) {
            Err(e) => match e {
                Error::IoError(e) => Err(SymphoniaDecoderError::IoError(e.to_string())),
                Error::SeekError(_) => {
                    unreachable!("Seek errors should not occur during initialization")
                }
                error => Err(SymphoniaDecoderError::DecodeError(error.to_string())),
            },
            Ok(Some(decoder)) => Ok(decoder),
            Ok(None) => Err(SymphoniaDecoderError::NoStreams),
        }
    }

    /// A "substantial portion" of this implementation comes from the `rodio` crate,
    /// https://github.com/RustAudio/rodio/blob/1c2cd2f6d99c005533b7a2b4c19ef41728f62116/src/decoder/symphonia.rs
    /// and is licensed under the MIT License.
    fn init(mss: MediaSourceStream) -> symphonia::core::errors::Result<Option<SymphoniaDecoder>> {
        let hint = Hint::new();
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let mut probed = get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        let stream = match probed.format.default_track() {
            Some(stream) => stream,
            None => return Ok(None),
        };

        // Select the first supported track
        let track = probed
            .format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(Error::Unsupported("No track with supported codec"))?;

        let track_id = track.id;

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;
        let total_duration = stream
            .codec_params
            .time_base
            .zip(stream.codec_params.n_frames)
            .map(|(base, spans)| base.calc_time(spans).into());

        let mut decode_errors: usize = 0;
        let decoded = loop {
            let current_span = match probed.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(_)) => break decoder.last_decoded(),
                Err(e) => return Err(e),
            };

            // If the packet does not belong to the selected track, skip over it
            if current_span.track_id() != track_id {
                continue;
            }

            match decoder.decode(&current_span) {
                Ok(decoded) => break decoded,
                Err(e) => match e {
                    Error::DecodeError(_) => {
                        decode_errors += 1;
                        if decode_errors > MAX_DECODE_RETRIES {
                            return Err(e);
                        } else {
                            continue;
                        }
                    }
                    _ => return Err(e),
                },
            }
        };
        let spec = decoded.spec().to_owned();
        let buffer = SymphoniaDecoder::get_buffer(decoded, &spec);
        Ok(Some(SymphoniaDecoder {
            decoder,
            current_span_offset: 0,
            format: probed.format,
            total_duration,
            buffer,
            spec,
        }))
    }

    #[inline]
    fn get_buffer(decoded: AudioBufferRef, spec: &SignalSpec) -> SampleBuffer<f32> {
        let duration = units::Duration::from(decoded.capacity() as u64);
        let mut buffer = SampleBuffer::<f32>::new(duration, *spec);
        buffer.copy_interleaved_ref(decoded);
        buffer
    }
}

impl Decoder for SymphoniaDecoder {
    /// A function that should decode and resample a song, optionally
    /// extracting the song's metadata such as the artist, the album, etc.
    ///
    /// The output sample array should be resampled to f32le, one channel, with a sampling rate
    /// of 22050 Hz. Anything other than that will yield wrong results.
    #[allow(clippy::missing_inline_in_public_items)]
    fn decode(path: &std::path::Path) -> BlissResult<PreAnalyzedSong> {
        // open the file
        let file = File::open(path)
            .map_err(SymphoniaDecoderError::from)
            .map_err(Into::into)?;
        // create the media source stream
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let source = Self::new(mss).map_err(Into::into)?;

        // we need to collapse the audio source into one channel
        // channels are interleaved, so if we have 2 channels, `[1, 2, 3, 4]` and `[5, 6, 7, 8]`,
        // they will be stored as `[1, 5, 2, 6, 3, 7, 4, 8]`
        //
        // we can make this mono by averaging the channels
        //
        // TODO: Figure out how ffmpeg does it, and do it the same way
        let num_channels = source.spec.channels.count();
        let sample_rate = source.spec.rate;
        let Some(total_duration) = source.total_duration else {
            return Err(SymphoniaDecoderError::IndeterminantDuration.into());
        };

        let mono_sample_array = {
            let mut mono_sample_array = Vec::with_capacity(
                (SAMPLE_RATE as f32 * total_duration.as_secs_f32() + 1.0) as usize,
            );
            let mut iter = source.into_iter();
            while let Some(left) = iter.next() {
                let mut right = 0.;
                for _ in 1..num_channels {
                    right += iter.next().unwrap_or(0.0);
                }
                mono_sample_array.push((left + right) / (num_channels as f32));
            }
            mono_sample_array.shrink_to_fit();
            mono_sample_array
        };

        // then we need to resample the audio source into 22050 Hz
        let resampled_array = if sample_rate == SAMPLE_RATE {
            mono_sample_array
        } else {
            let window = WindowFunction::Blackman;
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: calculate_cutoff(256, window),
                oversampling_factor: 128,
                interpolation: SincInterpolationType::Cubic,
                window,
            };

            let mut resampler = SincFixedIn::new(
                f64::from(SAMPLE_RATE) / f64::from(sample_rate),
                1.0,
                params,
                mono_sample_array.len(),
                1,
            )
            .map_err(SymphoniaDecoderError::from)
            .map_err(Into::into)?;
            resampler
                .process(&[&mono_sample_array], None)
                .map_err(SymphoniaDecoderError::from)
                .map_err(Into::into)?[0]
                .clone()
        };

        Ok(PreAnalyzedSong {
            path: path.to_owned(),
            sample_array: resampled_array,
            ..Default::default()
        })
    }
}

/// This implementation comes from the `rodio` crate,
/// https://github.com/RustAudio/rodio/blob/1c2cd2f6d99c005533b7a2b4c19ef41728f62116/src/decoder/symphonia.rs
/// and is licensed under the MIT License.
impl Iterator for SymphoniaDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_span_offset >= self.buffer.len() {
            let mut decode_errors = 0;
            let decoded = loop {
                let packet = self.format.next_packet().ok()?;
                let decoded = match self.decoder.decode(&packet) {
                    Ok(decoded) => decoded,
                    Err(_) => {
                        decode_errors += 1;
                        if decode_errors > MAX_DECODE_RETRIES {
                            return None;
                        } else {
                            continue;
                        }
                    }
                };

                // Loop until we get a packet with audio frames. This is necessary because some
                // formats can have packets with only metadata, particularly when rewinding, in
                // which case the iterator would otherwise end with `None`.
                // Note: checking `decoded.frames()` is more reliable than `packet.dur()`, which
                // can resturn non-zero durations for packets without audio frames.
                if decoded.frames() > 0 {
                    break decoded;
                }
            };

            decoded.spec().clone_into(&mut self.spec);
            self.buffer = SymphoniaDecoder::get_buffer(decoded, &self.spec);
            self.current_span_offset = 0;
        }

        let sample = *self.buffer.samples().get(self.current_span_offset)?;
        self.current_span_offset += 1;

        Some(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::{Decoder as DecoderTrait, SymphoniaDecoder as Decoder};
    use adler32::RollingAdler32;
    use pretty_assertions::assert_eq;
    use std::path::Path;

    fn _test_decode(path: &Path, expected_hash: u32) {
        let song = Decoder::decode(path).unwrap();
        let mut hasher = RollingAdler32::new();
        for sample in &song.sample_array {
            hasher.update_buffer(&sample.to_le_bytes());
        }

        assert_eq!(expected_hash, hasher.hash());
    }

    // expected hashs Obtained through
    // ffmpeg -i data/s16_stereo_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le -f hash -hash adler32 -

    #[test]
    fn test_decode_wav() {
        let expected_hash = 0xde831e82;
        _test_decode(Path::new("data/piano.wav"), expected_hash);
    }

    #[test]
    #[ignore = "fails when asked to convert stereo to mono, ig ffmpeg does it differently, but I'm not sure what the difference actually is"]
    fn test_resample_multi() {
        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let expected_hash = 0xbbcba1cf;
        _test_decode(&path, expected_hash);
    }

    #[test]
    #[ignore = "fails when asked to convert stereo to mono, ig ffmpeg does it differently, but I'm not sure what the difference actually is"]
    fn test_resample_stereo() {
        let path = Path::new("data/s16_stereo_22_5kHz.flac");
        let expected_hash = 0x1d7b2d6d;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_decode_mono() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash adler32 -
        let expected_hash = 0x5e01930b;
        _test_decode(&path, expected_hash);
    }

    #[test]
    #[ignore = "fails when asked to convert stereo to mono, ig ffmpeg does it differently, but I'm not sure what the difference actually is"]
    fn test_decode_mp3() {
        let path = Path::new("data/s32_stereo_44_1_kHz.mp3");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.mp3 -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash adler32 -
        //1030601839
        let expected_hash = 0x69ca6906;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_dont_panic_no_channel_layout() {
        let path = Path::new("data/no_channel.wav");
        Decoder::decode(path).unwrap();
    }

    #[test]
    fn test_decode_right_capacity_vec() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        let song = Decoder::decode(path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len(), // + SAMPLE_RATE as usize, // The + SAMPLE_RATE is because bliss-rs would add an extra second as a buffer, we don't need to because we know the exact length of the song
            sample_array.capacity()
        );

        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let song = Decoder::decode(path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len(), // + SAMPLE_RATE as usize,
            sample_array.capacity()
        );

        let path = Path::new("data/capacity_fix.ogg");
        let song = Decoder::decode(path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len(), // + SAMPLE_RATE as usize,
            sample_array.capacity()
        );
    }
}
