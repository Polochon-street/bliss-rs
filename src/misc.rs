//! Miscellaneous feature extraction module.
//!
//! Contains various descriptors that don't fit in one of the
//! existing categories.

use bliss_audio_aubio_rs::level_lin;
use ndarray::{arr1, Axis};

use super::utils::{mean, Normalize};

/**
 * Loudness (in dB) detection object.
 *
 * It indicates how "loud" a recording of a song is. For a given audio signal,
 * this value increases if the amplitude of the signal, and nothing else, is
 * increased.
 *
 * Of course, this makes this result dependent of the recording, meaning
 * the same song would yield different loudness on different recordings. Which
 * is exactly what we want, given that this is not a music theory project, but
 * one that aims at giving the best real-life results.
 *
 * Ranges between -90 dB (~silence) and 0 dB.
 *
 * (This is technically the sound pressure level of the track, but loudness is
 * way more visual)
 */
#[derive(Default)]
pub(crate) struct LoudnessDesc {
    pub values: Vec<f32>,
}

impl LoudnessDesc {
    pub const WINDOW_SIZE: usize = 1024;

    pub fn do_(&mut self, chunk: &[f32]) {
        let level = level_lin(chunk);
        self.values.push(level);
    }

    pub fn get_value(&mut self) -> Vec<f32> {
        let mut std_value = arr1(&self.values).std_axis(Axis(0), 0.).into_scalar();
        let mut mean_value = mean(&self.values);
        // Make sure the dB don't go less than -90dB
        if mean_value < 1e-9 {
            mean_value = 1e-9
        };
        if std_value < 1e-9 {
            std_value = 1e-9
        }
        vec![
            self.normalize(10.0 * mean_value.log10()),
            self.normalize(10.0 * std_value.log10()),
        ]
    }
}

impl Normalize for LoudnessDesc {
    const MAX_VALUE: f32 = 0.;
    const MIN_VALUE: f32 = -90.;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::ffmpeg::FFmpegDecoder as Decoder;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::Decoder as DecoderTrait;
    #[cfg(feature = "ffmpeg")]
    use std::path::Path;

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_loudness() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut loudness_desc = LoudnessDesc::default();
        for chunk in song.sample_array.chunks_exact(LoudnessDesc::WINDOW_SIZE) {
            loudness_desc.do_(&chunk);
        }
        let expected_values = vec![0.271263, 0.2577181];
        for (expected, actual) in expected_values.iter().zip(loudness_desc.get_value().iter()) {
            assert!(0.01 > (expected - actual).abs());
        }
    }

    #[test]
    fn test_loudness_boundaries() {
        let mut loudness_desc = LoudnessDesc::default();
        let silence_chunk = vec![0.; 1024];
        loudness_desc.do_(&silence_chunk);
        let expected_values = vec![-1., -1.];
        for (expected, actual) in expected_values.iter().zip(loudness_desc.get_value().iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }

        let mut loudness_desc = LoudnessDesc::default();
        let silence_chunk = vec![1.; 1024];
        loudness_desc.do_(&silence_chunk);
        let expected_values = vec![1., -1.];
        for (expected, actual) in expected_values.iter().zip(loudness_desc.get_value().iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }

        let mut loudness_desc = LoudnessDesc::default();
        let silence_chunk = vec![-1.; 1024];
        loudness_desc.do_(&silence_chunk);
        let expected_values = vec![1., -1.];
        for (expected, actual) in expected_values.iter().zip(loudness_desc.get_value().iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }
}
