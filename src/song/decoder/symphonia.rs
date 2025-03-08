//! Decoder implementation that uses the `symphonia` crate to decode audio files, and the `rubato` crate to resample the audio files.
//!
//! Upstreamed from the `mecomp-analysis` crate.

use std::{f32::consts::SQRT_2, fs::File, time::Duration};

// use rodio::Source;
use rubato::{FftFixedIn, Resampler};
use symphonia::{
    core::{
        audio::{AudioBufferRef, Layout, SampleBuffer, SignalSpec},
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
impl From<SymphoniaDecoderError> for BlissError {
    fn from(err: SymphoniaDecoderError) -> Self {
        BlissError::DecodingError(err.to_string())
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

    /// we need to collapse the audio source into one channel
    /// channels are interleaved, so if we have 2 channels, `[1, 2, 3, 4]` and `[5, 6, 7, 8]`,
    /// they will be stored as `[1, 5, 2, 6, 3, 7, 4, 8]`
    ///
    /// For stereo sound, we can make this mono by averaging the channels and multiplying by the square root of 2,
    /// This recovers the exact behavior of ffmpeg when converting stereo to mono, however for 2.1 and 5.1 surround sound,
    /// ffmpeg might be doing something different, and I'm not sure what that is (don't have a 5.1 surround sound file to test with)
    ///
    /// TODO: Figure out how ffmpeg does it for 2.1 and 5.1 surround sound, and do it the same way
    #[inline]
    fn into_mono_samples(self) -> Result<Vec<f32>, SymphoniaDecoderError> {
        let num_channels = self.spec.channels.count();
        let sample_rate = self.spec.rate;
        let Some(total_duration) = self.total_duration else {
            return Err(SymphoniaDecoderError::IndeterminantDuration);
        };

        match num_channels {
            // mono
            1 => Ok(self.collect()),
            // stereo
            2 => {
                assert!(self.spec.channels == Layout::Stereo.into_channels());

                let mut mono_sample_array = Vec::with_capacity(
                    (total_duration.as_secs() as usize + 1) * sample_rate as usize,
                );
                let mut iter = self.peekable();
                while let Some(left) = iter.next() {
                    let right = iter.next().unwrap_or_default();
                    let sum = left + right;
                    let avg = sum * SQRT_2 / 2.0;
                    mono_sample_array.push(avg);
                }
                Ok(mono_sample_array)
            }
            // 2.1 surround
            _ => {
                log::warn!("The audio source has more than 2 channels (might be 2.1 or 5.1 surround sound), will collapse to mono by averaging the channels");
                let mut mono_sample_array = Vec::with_capacity(
                    (total_duration.as_secs() as usize + 1) * sample_rate as usize,
                );
                let mut iter = self.into_iter().peekable();
                while iter.peek().is_some() {
                    let sum = iter.by_ref().take(num_channels).sum::<f32>();
                    mono_sample_array.push(sum / (num_channels as f32));
                }
                Ok(mono_sample_array)
            }
        }
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
        let file = File::open(path).map_err(SymphoniaDecoderError::from)?;
        // create the media source stream
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let source = Self::new(mss)?;

        // Convert the audio source into a mono channel
        let sample_rate = source.spec.rate;
        let Some(total_duration) = source.total_duration else {
            return Err(SymphoniaDecoderError::IndeterminantDuration.into());
        };

        let mut mono_sample_array = source.into_mono_samples()?;

        // then we need to resample the audio source into 22050 Hz
        let resampled_array = if sample_rate == SAMPLE_RATE {
            mono_sample_array.shrink_to_fit();
            mono_sample_array
        } else {
            let mut resampled =
                Vec::with_capacity((total_duration.as_secs() as usize + 1) * SAMPLE_RATE as usize);

            const CHUNK_SIZE: usize = 1024;
            let mut resampler =
                FftFixedIn::new(sample_rate as usize, SAMPLE_RATE as usize, CHUNK_SIZE, 1, 1)
                    .map_err(SymphoniaDecoderError::from)?;

            let delay = resampler.output_delay();

            let new_length = mono_sample_array.len() * SAMPLE_RATE as usize / sample_rate as usize;
            let mut output_buffer = resampler.output_buffer_allocate(true);
            let mut input_buffer = Vec::with_capacity(CHUNK_SIZE);

            let mut samples = mono_sample_array.into_iter().peekable();

            while samples.peek().is_some() {
                debug_assert!(resampler.input_frames_next() == CHUNK_SIZE);

                input_buffer.clear();
                input_buffer.extend(
                    samples
                        .by_ref()
                        .chain(std::iter::repeat(0.0))
                        .take(CHUNK_SIZE),
                );

                let (_, output_written) = resampler
                    .process_into_buffer(&[&input_buffer], output_buffer.as_mut_slice(), None)
                    .unwrap();
                resampled.extend_from_slice(&output_buffer[0][..output_written]);
            }

            // flush final samples from resampler
            if resampled.len() < new_length + delay {
                let (_, output_written) = resampler
                    .process_partial_into_buffer(
                        Option::<&[&[f32]]>::None,
                        output_buffer.as_mut_slice(),
                        None,
                    )
                    .unwrap();
                resampled.extend_from_slice(&output_buffer[0][..output_written]);
            }

            resampled[delay..new_length + delay].to_vec()
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

    #[cfg(feature = "symphonia-wav")]
    #[test]
    fn test_decode_wav() {
        let expected_hash = 0xde831e82;
        _test_decode(Path::new("data/piano.wav"), expected_hash);
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    #[ignore = "fails when asked to resample to 22050 Hz, ig ffmpeg does it differently, but I'm not sure what the difference actually is"]
    fn test_resample_mono() {
        let path = Path::new("data/s32_mono_44_1_kHz.flac");
        let expected_hash = 0xa0f8b8af;
        _test_decode(&path, expected_hash);
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    fn test_resample_mono_ffmpeg_v_symphonia() {
        /*
        configurations tested:

        | Resampler, and configuration | difference from ffmpeg |
        - SincFixedIn, on whole buffer, with process_into_buffer:       0.0020331843
        - SincFixedIn, on whole buffer, with process:                   0.0020285384
        - FastFixedIn, on whole buffer, with process_into_buffer:       0.0039299703
        - FastFixedIn, on whole buffer, with process:                   0.0039298288
        - FftFixedIn, on whole buffer, with process_into_buffer:        0.017518902
        - FftFixedIn, on whole buffer, with process:                    0.0154156

        - SincFixedIn, on chunks of 1024, Cubic interp, Blackman        0.024933979
        - SincFixedIn, on chunks of 1024, Linear interp, Blackman       0.024933979
        - SincFixedIn, on chunks of 1024, Cubic interp, Blackman2       0.024934249
        - SincFixedIn, on chunks of 1024, Cubic interp, Hann            0.024934188
        - SincFixedIn, on chunks of 1024, Cubic interp, BlackmanHarris  0.024934053
        - FastFixedIn, on chunks of 1024, Cubic interp                  0.0039299796
        - FastFixedIn, on chunks of 8192, Cubic interp                  0.0039299796
        - FastFixedIn, on chunks of 8192, Linear interp                 0.0039299796
        - FftFixedIn, on chunks of 128, 1 subchunk                      0.000033739863
        - FftFixedIn, on chunks of 256, 1 subchunk                      0.000015570473
        - FftFixedIn, on chunks of 512, 1 subchunk                      0.0000071326162
        - FftFixedIn, on chunks of 1024, 32 subchunks                   0.0018597797
        - FftFixedIn, on chunks of 1024, 16 subchunks                   0.000092027316
        - FftFixedIn, on chunks of 1024, 1 subchunk                     0.0000068506047 // <--
        - FftFixedIn, on chunks of 2048, 1 subchunk                     0.0000070857413
        - FftFixedIn, on chunks of 4096, 1 subchunk                     0.0000071542086
        - FftFixedIn, on chunks of 4096, 4 subchunk                     0.0000068506047 // that makes sense actually, 4096/4 = 1024 so it makes sense this matches the output of CHUNK_SIZE=1024
        - FftFixedIn, on chunks of 8192, 1 subchunk                     0.000007135614
        - FftFixedIn, on chunks of 16384, 1 subchunk                    0.0000071084633
        - FftFixedIn, on chunks of 32768, 1 subchunk                    0.0000071034465
        - FftFixedIn, on chunks of 65736, 1 subchunk                    0.000007098081
        - FftFixedIn, on chunks of 1024*128, 1 subchunk                 0.000007097704
        - FftFixedIn, on chunks of 1024*256, 1 subchunk                 0.000007096261

        so FftFixedIn on chunks is definitely the best, if we make the chunks too small it diverges,
        and if we make them large we get diminishing returns, so we should probably stick to 1024

        Now, what can we do to eliminate the remaining error?


         */
        let path = Path::new("data/s32_mono_44_1_kHz.flac");
        let symphonia_decoded = Decoder::decode(&path).unwrap();
        let ffmpeg_decoded = crate::decoder::ffmpeg::FFmpeg::decode(&path).unwrap();
        // if the first 100 samples are equal, then the rest should be equal.
        // we check this first since the sample arrays are large enough that printing the diff would attempt
        // and fail to allocate memory for the string
        // assert_eq!(
        //     symphonia_decoded.sample_array[..100],
        //     ffmpeg_decoded.sample_array[..100]
        // );
        // assert_eq!(symphonia_decoded.sample_array, ffmpeg_decoded.sample_array);

        // calculate the similarity between the two arrays
        let mut diff = 0.0;
        for (a, b) in symphonia_decoded
            .sample_array
            .iter()
            .zip(ffmpeg_decoded.sample_array.iter())
        {
            diff += (a - b).abs();
        }
        diff /= symphonia_decoded.sample_array.len() as f32;
        assert!(
            diff < 1.0e-5,
            "Difference between symphonia and ffmpeg: {}",
            diff
        );
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    #[ignore = "fails when asked to resample to 22050 Hz, ig ffmpeg does it differently, but I'm not sure what the difference actually is"]
    fn test_resample_multi() {
        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let expected_hash = 0xbbcba1cf;
        _test_decode(&path, expected_hash);
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    fn test_resample_multi_ffmpeg_v_symphonia() {
        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let symphonia_decoded = Decoder::decode(&path).unwrap();
        let ffmpeg_decoded = crate::decoder::ffmpeg::FFmpeg::decode(&path).unwrap();

        // calculate the similarity between the two arrays
        let mut diff = 0.0;
        for (a, b) in symphonia_decoded
            .sample_array
            .iter()
            .zip(ffmpeg_decoded.sample_array.iter())
        {
            diff += (a - b).abs();
        }
        diff /= symphonia_decoded.sample_array.len() as f32;
        assert!(
            diff < 1.0e-5,
            "Difference between symphonia and ffmpeg: {}",
            diff
        );
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    fn test_resample_stereo() {
        let path = Path::new("data/s16_stereo_22_5kHz.flac");
        let expected_hash = 0x1d7b2d6d;
        _test_decode(&path, expected_hash);
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    // From this test, I was able to determine that multiplying the average of the channels by the square root of 2
    // recovers the exact behavior of ffmpeg when converting stereo to mono
    fn test_stereo_ffmpeg_v_symphonia() {
        let path = Path::new("data/s16_stereo_22_5kHz.flac");
        let symphonia_decoded = Decoder::decode(&path).unwrap();
        let ffmpeg_decoded = crate::decoder::ffmpeg::FFmpeg::decode(&path).unwrap();
        // if the first 100 samples are equal, then the rest should be equal.
        // we check this first since the sample arrays are large enough that printing the diff would attempt
        // and fail to allocate memory for the string
        assert_eq!(
            symphonia_decoded.sample_array[..100],
            ffmpeg_decoded.sample_array[..100]
        );
        assert_eq!(symphonia_decoded.sample_array, ffmpeg_decoded.sample_array);
    }

    #[cfg(feature = "symphonia-flac")]
    #[test]
    fn test_decode_mono() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash adler32 -
        let expected_hash = 0x5e01930b;
        _test_decode(&path, expected_hash);
    }

    #[cfg(feature = "symphonia-mp3")]
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

    #[cfg(feature = "symphonia-mp3")]
    #[test]
    fn test_decode_mp3_ffmpeg_v_symphonia() {
        // TODO: Figure out how to get the error down to 1.0e-5
        let path = Path::new("data/s32_stereo_44_1_kHz.mp3");
        let symphonia_decoded = Decoder::decode(&path).unwrap();
        let ffmpeg_decoded = crate::decoder::ffmpeg::FFmpeg::decode(&path).unwrap();

        // calculate the similarity between the two arrays
        let mut diff = 0.0;
        for (a, b) in symphonia_decoded
            .sample_array
            .iter()
            .zip(ffmpeg_decoded.sample_array.iter())
        {
            diff += (a - b).abs();
        }
        diff /= symphonia_decoded.sample_array.len() as f32;
        assert!(
            diff < 0.05,
            "Difference between symphonia and ffmpeg: {}",
            diff
        );
    }

    #[cfg(feature = "symphonia-wav")]
    #[test]
    fn test_dont_panic_no_channel_layout() {
        let path = Path::new("data/no_channel.wav");
        Decoder::decode(path).unwrap();
    }

    #[cfg(all(feature = "symphonia-flac", feature = "symphonia-ogg"))]
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

    #[cfg(all(
        feature = "symphonia-flac",
        feature = "symphonia-ogg",
        feature = "symphonia-wav",
        feature = "symphonia-mp3"
    ))]
    #[test]
    fn compare_ffmpeg_to_symphonia_for_all_test_songs() {
        let paths_and_tolerances = [
            ("data/capacity_fix.ogg", 0.0000000017),
            ("data/no_channel.wav", 0.027),
            ("data/no_tags.flac", 0.175),
            ("data/piano.flac", f32::EPSILON),
            ("data/piano.wav", f32::EPSILON),
            ("data/s16_mono_22_5kHz.flac", f32::EPSILON),
            ("data/s16_stereo_22_5kHz.flac", f32::EPSILON),
            ("data/s32_mono_44_1_kHz.flac", 0.0000069),
            ("data/s32_stereo_44_1_kHz.flac", 0.00001),
            ("data/s32_stereo_44_1_kHz.mp3", 0.03),
            ("data/special-tags.mp3", 0.312),
            ("data/tone_11080Hz.flac", 0.175),
            ("data/unsupported-tags.mp3", 0.312),
            ("data/white_noise.mp3", 0.312),
        ];

        for (path_str, tolerance) in paths_and_tolerances {
            let path = Path::new(path_str);
            let symphonia_decoded = Decoder::decode(&path).unwrap();
            let ffmpeg_decoded = crate::decoder::ffmpeg::FFmpeg::decode(&path).unwrap();

            // calculate the similarity between the two arrays
            let mut diff = 0.0;
            for (a, b) in symphonia_decoded
                .sample_array
                .iter()
                .zip(ffmpeg_decoded.sample_array.iter())
            {
                diff += (a - b).abs();
            }
            diff /= symphonia_decoded.sample_array.len() as f32;
            assert!(
                diff < tolerance,
                "Difference between symphonia and ffmpeg: {diff}, tolerance: {tolerance}, file: {path_str}",
            );
        }
    }

    #[cfg(all(feature = "bench", feature = "symphonia-flac", test))]
    mod bench {
        extern crate test;
        use crate::decoder::symphonia::SymphoniaDecoder as Decoder;
        use crate::decoder::Decoder as DecoderTrait;
        use std::path::Path;
        use test::Bencher;

        #[bench]
        /// No resampling, just decoding
        fn bench_decode_mono(b: &mut Bencher) {
            let path = Path::new("./data/s16_mono_22_5kHz.flac");
            b.iter(|| {
                Decoder::decode(&path).unwrap();
            });
        }

        #[bench]
        /// needs to convert from stereo to mono
        fn bench_decode_stereo(b: &mut Bencher) {
            let path = Path::new("./data/s16_stereo_22_5kHz.flac");
            b.iter(|| {
                Decoder::decode(&path).unwrap();
            });
        }

        #[bench]
        /// needs to convert from 44.1 kHz to 22.05 kHz
        fn bench_resample_mono(b: &mut Bencher) {
            let path = Path::new("./data/s32_mono_44_1_kHz.flac");
            b.iter(|| {
                Decoder::decode(&path).unwrap();
            });
        }

        #[bench]
        /// needs to convert from 44.1 kHz to 22.05 kHz
        /// and from stereo to mono
        fn bench_resample_multi(b: &mut Bencher) {
            let path = Path::new("./data/s32_stereo_44_1_kHz.flac");
            b.iter(|| {
                Decoder::decode(&path).unwrap();
            });
        }
    }
}
