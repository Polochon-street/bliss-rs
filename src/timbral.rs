//! Timbral feature extraction module.
//!
//! Contains functions to extract & summarize the zero-crossing rate,
//! spectral centroid, spectral flatness and spectral roll-off of
//! a given Song.

use bliss_audio_aubio_rs::vec::CVec;
use bliss_audio_aubio_rs::{bin_to_freq, PVoc, SpecDesc, SpecShape};
use ndarray::{arr1, Axis};

use super::utils::{geometric_mean, mean, number_crossings, Normalize};
use crate::{BlissError, BlissResult, SAMPLE_RATE};

/**
 * General object holding all the spectral descriptor.
 *
 * Holds 3 spectral descriptors together. It would be better conceptually
 * to have 3 different spectral descriptor objects, but this avoids re-computing
 * the same FFT three times.
 *
 * Current spectral descriptors are spectral centroid, spectral rolloff and
 * spectral flatness (see `values_object` for a further description of the
 * object.
 *
 * All descriptors are currently summarized by their mean only.
 */
pub(crate) struct SpectralDesc {
    phase_vocoder: PVoc,
    sample_rate: u32,

    centroid_aubio_desc: SpecDesc,
    rolloff_aubio_desc: SpecDesc,
    values_centroid: Vec<f32>,
    values_rolloff: Vec<f32>,
    values_flatness: Vec<f32>,
}

impl SpectralDesc {
    pub const WINDOW_SIZE: usize = 512;
    pub const HOP_SIZE: usize = SpectralDesc::WINDOW_SIZE / 4;

    /**
     * Compute score related to the
     * [spectral centroid](https://en.wikipedia.org/wiki/Spectral_centroid) values,
     * obtained after repeatedly calling `do_` on all of the song's chunks.
     *
     * Spectral centroid is used to determine the "brightness" of a sound, i.e.
     * how much high frequency there is in an audio signal.
     *
     * It of course depends of the instrument used: a piano-only track that makes
     * use of high frequencies will still score less than a song using a lot of
     * percussive sound, because the piano frequency range is lower.
     *
     * The value range is between 0 and `sample_rate / 2`.
     */
    pub fn get_centroid(&mut self) -> Vec<f32> {
        vec![
            self.normalize(mean(&self.values_centroid)),
            self.normalize(
                arr1(&self.values_centroid)
                    .std_axis(Axis(0), 0.)
                    .into_scalar(),
            ),
        ]
    }

    /**
     * Compute score related to the spectral roll-off values, obtained
     * after repeatedly calling `do_` on all of the song's chunks.
     *
     * Spectral roll-off is the bin frequency number below which a certain
     * percentage of the spectral energy is found, here, 95%.
     *
     * It can be used to distinguish voiced speech (low roll-off) and unvoiced
     * speech (high roll-off). It is also a good indication of the energy
     * repartition of a song.
     *
     * The value range is between 0 and `sample_rate / 2`
     */
    pub fn get_rolloff(&mut self) -> Vec<f32> {
        vec![
            self.normalize(mean(&self.values_rolloff)),
            self.normalize(
                arr1(&self.values_rolloff)
                    .std_axis(Axis(0), 0.)
                    .into_scalar(),
            ),
        ]
    }

    /**
     * Compute score related to the
     * [spectral flatness](https://en.wikipedia.org/wiki/Spectral_flatness) values,
     * obtained after repeatedly calling `do_` on all of the song's chunks.
     *
     * Spectral flatness is the ratio between the geometric mean of the spectrum
     * and its arithmetic mean.
     *
     * It is used to distinguish between tone-like and noise-like signals.
     * Tone-like audio is f.ex. a piano key, something that has one or more
     * specific frequencies, while (white) noise has an equal distribution
     * of intensity among all frequencies.
     *
     * The value range is between 0 and 1, since the geometric mean is always less
     * than the arithmetic mean.
     */
    pub fn get_flatness(&mut self) -> Vec<f32> {
        let max_value = 1.;
        let min_value = 0.;
        // Range is different from the other spectral algorithms, so normalizing
        // manually here.
        vec![
            2. * (mean(&self.values_flatness) - min_value) / (max_value - min_value) - 1.,
            2. * (arr1(&self.values_flatness)
                .std_axis(Axis(0), 0.)
                .into_scalar()
                - min_value)
                / (max_value - min_value)
                - 1.,
        ]
    }

    pub fn new(sample_rate: u32) -> BlissResult<Self> {
        Ok(SpectralDesc {
            centroid_aubio_desc: SpecDesc::new(SpecShape::Centroid, SpectralDesc::WINDOW_SIZE)
                .map_err(|e| {
                    BlissError::AnalysisError(format!(
                        "error while loading aubio centroid object: {e}",
                    ))
                })?,
            rolloff_aubio_desc: SpecDesc::new(SpecShape::Rolloff, SpectralDesc::WINDOW_SIZE)
                .map_err(|e| {
                    BlissError::AnalysisError(format!(
                        "error while loading aubio rolloff object: {e}",
                    ))
                })?,
            phase_vocoder: PVoc::new(SpectralDesc::WINDOW_SIZE, SpectralDesc::HOP_SIZE).map_err(
                |e| {
                    BlissError::AnalysisError(
                        format!("error while loading aubio pvoc object: {e}",),
                    )
                },
            )?,
            values_centroid: Vec::new(),
            values_rolloff: Vec::new(),
            values_flatness: Vec::new(),
            sample_rate,
        })
    }

    /**
     * Compute all the descriptors' value for the given chunk.
     *
     * After using this on all the song's chunks, you can call
     * `get_centroid`, `get_flatness` and `get_rolloff` to get the respective
     * descriptors' values.
     */
    pub fn do_(&mut self, chunk: &[f32]) -> BlissResult<()> {
        let mut fftgrain: Vec<f32> = vec![0.0; SpectralDesc::WINDOW_SIZE];
        self.phase_vocoder
            .do_(chunk, fftgrain.as_mut_slice())
            .map_err(|e| {
                BlissError::AnalysisError(format!("error while processing aubio pv object: {e}"))
            })?;

        let bin = self
            .centroid_aubio_desc
            .do_result(fftgrain.as_slice())
            .map_err(|e| {
                BlissError::AnalysisError(format!(
                    "error while processing aubio centroid object: {e}",
                ))
            })?;

        let freq = bin_to_freq(
            bin,
            self.sample_rate as f32,
            SpectralDesc::WINDOW_SIZE as f32,
        );
        self.values_centroid.push(freq);

        let mut bin = self
            .rolloff_aubio_desc
            .do_result(fftgrain.as_slice())
            .unwrap();

        // Until https://github.com/aubio/aubio/pull/318 is in
        if bin > SpectralDesc::WINDOW_SIZE as f32 / 2. {
            bin = SpectralDesc::WINDOW_SIZE as f32 / 2.;
        }

        let freq = bin_to_freq(
            bin,
            self.sample_rate as f32,
            SpectralDesc::WINDOW_SIZE as f32,
        );
        self.values_rolloff.push(freq);

        let cvec: CVec = fftgrain.as_slice().into();
        let geo_mean = geometric_mean(cvec.norm());
        if geo_mean == 0.0 {
            self.values_flatness.push(0.0);
            return Ok(());
        }
        let flatness = geo_mean / mean(cvec.norm());
        self.values_flatness.push(flatness);
        Ok(())
    }
}

impl Normalize for SpectralDesc {
    const MAX_VALUE: f32 = SAMPLE_RATE as f32 / 2.;
    const MIN_VALUE: f32 = 0.;
}

/**
 * [Zero-crossing rate](https://en.wikipedia.org/wiki/Zero-crossing_rate)
 * detection object.
 *
 * Zero-crossing rate is mostly used to detect percussive sounds in an audio
 * signal, as well as whether an audio signal contains speech or not.
 *
 * It is a good metric to differentiate between songs with people speaking clearly,
 * (e.g. slam) and instrumental songs.
 *
 * The value range is between 0 and 1.
 */
#[derive(Default)]
pub(crate) struct ZeroCrossingRateDesc {
    values: Vec<u32>,
    number_samples: usize,
}

impl ZeroCrossingRateDesc {
    #[allow(dead_code)]
    pub fn new(_sample_rate: u32) -> Self {
        ZeroCrossingRateDesc::default()
    }

    /// Count the number of zero-crossings for the current `chunk`.
    pub fn do_(&mut self, chunk: &[f32]) {
        self.values.push(number_crossings(chunk));
        self.number_samples += chunk.len();
    }

    /// Sum the number of zero-crossings witnessed and divide by
    /// the total number of samples.
    pub fn get_value(&mut self) -> f32 {
        self.normalize((self.values.iter().sum::<u32>()) as f32 / self.number_samples as f32)
    }
}

impl Normalize for ZeroCrossingRateDesc {
    const MAX_VALUE: f32 = 1.;
    const MIN_VALUE: f32 = 0.;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::ffmpeg::FFmpeg as Decoder;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::Decoder as DecoderTrait;
    #[cfg(feature = "ffmpeg")]
    use std::path::Path;

    #[test]
    fn test_zcr_boundaries() {
        let mut zcr_desc = ZeroCrossingRateDesc::default();
        let chunk = vec![0.; 1024];
        zcr_desc.do_(&chunk);
        assert_eq!(-1., zcr_desc.get_value());

        let one_chunk = vec![-1., 1.];
        let chunks = std::iter::repeat(one_chunk.iter())
            .take(512)
            .flatten()
            .cloned()
            .collect::<Vec<f32>>();
        let mut zcr_desc = ZeroCrossingRateDesc::default();
        zcr_desc.do_(&chunks);
        assert!(0.001 > (0.9980469 - zcr_desc.get_value()).abs());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_zcr() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut zcr_desc = ZeroCrossingRateDesc::default();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            zcr_desc.do_(&chunk);
        }
        assert!(0.001 > (-0.85036 - zcr_desc.get_value()).abs());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_flatness_boundaries() {
        let mut spectral_desc = SpectralDesc::new(10).unwrap();
        let chunk = vec![0.; 1024];

        let expected_values = vec![-1., -1.];
        spectral_desc.do_(&chunk).unwrap();
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_flatness().iter())
        {
            assert!(0.0000001 > (expected - actual).abs());
        }

        let song = Decoder::decode(Path::new("data/white_noise.mp3")).unwrap();
        let mut spectral_desc = SpectralDesc::new(22050).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        println!("{:?}", spectral_desc.get_flatness());
        // White noise - as close to 1 as possible
        let expected_values = vec![0.5785303, -0.9426308];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_flatness().iter())
        {
            assert!(0.001 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_flatness() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        // Spectral flatness mean value computed here with phase vocoder before normalization: 0.111949615
        // Essentia value with spectrum / hann window: 0.11197535695207445
        let expected_values = vec![-0.77610075, -0.8148179];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_flatness().iter())
        {
            assert!(0.01 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_roll_off_boundaries() {
        let mut spectral_desc = SpectralDesc::new(10).unwrap();
        let chunk = vec![0.; 512];

        let expected_values = vec![-1., -1.];
        spectral_desc.do_(&chunk).unwrap();
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_rolloff().iter())
        {
            assert!(0.0000001 > (expected - actual).abs());
        }

        let song = Decoder::decode(Path::new("data/tone_11080Hz.flac")).unwrap();
        let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        let expected_values = vec![0.9967681, -0.99615175];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_rolloff().iter())
        {
            assert!(0.0001 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_roll_off() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        let expected_values = vec![-0.6326486, -0.7260933];
        // Roll-off mean value computed here with phase vocoder before normalization: 2026.7644
        // Essentia value with spectrum / hann window: 1979.632683520047
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_rolloff().iter())
        {
            assert!(0.01 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_centroid() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        // Spectral centroid mean value computed here with phase vocoder before normalization: 1354.2273
        // Essentia value with spectrum / hann window: 1351
        let expected_values = vec![-0.75483, -0.87916887];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_centroid().iter())
        {
            assert!(0.0001 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_spectral_centroid_boundaries() {
        let mut spectral_desc = SpectralDesc::new(10).unwrap();
        let chunk = vec![0.; 512];

        spectral_desc.do_(&chunk).unwrap();
        let expected_values = vec![-1., -1.];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_centroid().iter())
        {
            assert!(0.0000001 > (expected - actual).abs());
        }
        let song = Decoder::decode(Path::new("data/tone_11080Hz.flac")).unwrap();
        let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(SpectralDesc::HOP_SIZE) {
            spectral_desc.do_(&chunk).unwrap();
        }
        let expected_values = vec![0.97266, -0.9609926];
        for (expected, actual) in expected_values
            .iter()
            .zip(spectral_desc.get_centroid().iter())
        {
            assert!(0.00001 > (expected - actual).abs());
        }
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    extern crate test;
    use crate::timbral::{SpectralDesc, ZeroCrossingRateDesc};
    use test::Bencher;

    #[bench]
    fn bench_spectral_desc(b: &mut Bencher) {
        let mut spectral_desc = SpectralDesc::new(10).unwrap();
        let chunk = vec![0.; 512];

        b.iter(|| {
            spectral_desc.do_(&chunk).unwrap();
        });
    }

    #[bench]
    fn bench_zcr_desc(b: &mut Bencher) {
        let mut zcr_desc = ZeroCrossingRateDesc::new(10);
        let chunk = vec![0.; 512];

        b.iter(|| {
            zcr_desc.do_(&chunk);
        });
    }
}
