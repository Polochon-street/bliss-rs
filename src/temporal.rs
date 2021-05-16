//! Temporal feature extraction module.
//!
//! Contains functions to extract & summarize the temporal aspects
//! of a given Song.

use crate::utils::Normalize;
use crate::BlissError;
use bliss_audio_aubio_rs::{OnsetMode, Tempo};
use log::warn;
use ndarray::arr1;
use ndarray_stats::interpolate::Midpoint;
use ndarray_stats::Quantile1dExt;
use noisy_float::prelude::*;

/**
 * Beats per minutes ([BPM](https://en.wikipedia.org/wiki/Tempo#Measurement))
 * detection object.
 *
 * It indicates the (subjective) "speed" of a music piece. The higher the BPM,
 * the "quicker" the song will feel.
 *
 * It uses `WPhase`, a phase-deviation onset detection function to perform
 * onset detection; it proved to be the best for finding out the BPM of a panel
 * of songs I had, but it could very well be replaced by something better in the
 * future.
 *
 * Ranges from 0 (theoretically...) to 206 BPM. (Even though aubio apparently
 * has trouble to identify tempo > 190 BPM - did not investigate too much)
 *
 */
pub(crate) struct BPMDesc {
    aubio_obj: Tempo,
    bpms: Vec<f32>,
}

// TODO>1.0 use the confidence value to discard this descriptor if confidence
// is too low.
impl BPMDesc {
    pub const WINDOW_SIZE: usize = 512;
    pub const HOP_SIZE: usize = BPMDesc::WINDOW_SIZE / 2;

    pub fn new(sample_rate: u32) -> Result<Self, BlissError> {
        Ok(BPMDesc {
            aubio_obj: Tempo::new(
                OnsetMode::SpecFlux,
                BPMDesc::WINDOW_SIZE,
                BPMDesc::HOP_SIZE,
                sample_rate,
            )
            .map_err(|e| {
                BlissError::AnalysisError(format!(
                    "error while loading aubio tempo object: {}",
                    e.to_string()
                ))
            })?,
            bpms: Vec::new(),
        })
    }

    pub fn do_(&mut self, chunk: &[f32]) -> Result<(), BlissError> {
        let result = self.aubio_obj.do_result(chunk).map_err(|e| {
            BlissError::AnalysisError(format!(
                "aubio error while computing tempo {}",
                e.to_string()
            ))
        })?;

        if result > 0. {
            self.bpms.push(self.aubio_obj.get_bpm());
        }
        Ok(())
    }

    /**
     * Compute score related to tempo.
     * Right now, basically returns the song's BPM.
     *
     * - `song` Song to compute score from
     */
    pub fn get_value(&mut self) -> f32 {
        if self.bpms.is_empty() {
            warn!("Set tempo value to zero because no beats were found.");
            return -1.;
        }
        let median = arr1(&self.bpms)
            .mapv(n32)
            .quantile_mut(n64(0.5), &Midpoint)
            .unwrap();
        self.normalize(median.into())
    }
}

impl Normalize for BPMDesc {
    // See aubio/src/tempo/beattracking.c:387
    // Should really be 413, needs testing
    const MAX_VALUE: f32 = 206.;
    const MIN_VALUE: f32 = 0.;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Song, SAMPLE_RATE};

    #[test]
    fn test_tempo_real() {
        let song = Song::decode("data/s16_mono_22_5kHz.flac").unwrap();
        let mut tempo_desc = BPMDesc::new(SAMPLE_RATE).unwrap();
        for chunk in song.sample_array.chunks_exact(BPMDesc::HOP_SIZE) {
            tempo_desc.do_(&chunk).unwrap();
        }
        assert!(0.01 > (0.378605 - tempo_desc.get_value()).abs());
    }

    #[test]
    fn test_tempo_artificial() {
        let mut tempo_desc = BPMDesc::new(22050).unwrap();
        // This gives one beat every second, so 60 BPM
        let mut one_chunk = vec![0.; 22000];
        one_chunk.append(&mut vec![1.; 100]);
        let chunks = std::iter::repeat(one_chunk.iter())
            .take(100)
            .flatten()
            .cloned()
            .collect::<Vec<f32>>();
        for chunk in chunks.chunks_exact(BPMDesc::HOP_SIZE) {
            tempo_desc.do_(&chunk).unwrap();
        }

        // -0.41 is 60 BPM normalized
        assert!(0.01 > (-0.416853 - tempo_desc.get_value()).abs());
    }

    #[test]
    fn test_tempo_boundaries() {
        let mut tempo_desc = BPMDesc::new(10).unwrap();
        let silence_chunk = vec![0.; 1024];
        tempo_desc.do_(&silence_chunk).unwrap();
        assert_eq!(-1., tempo_desc.get_value());

        let mut tempo_desc = BPMDesc::new(22050).unwrap();
        // The highest value I could obtain was with these params, even though
        // apparently the higher bound is 206 BPM, but here I found ~189 BPM.
        let mut one_chunk = vec![0.; 6989];
        one_chunk.append(&mut vec![1.; 20]);
        let chunks = std::iter::repeat(one_chunk.iter())
            .take(500)
            .flatten()
            .cloned()
            .collect::<Vec<f32>>();
        for chunk in chunks.chunks_exact(BPMDesc::HOP_SIZE) {
            tempo_desc.do_(&chunk).unwrap();
        }
        // 0.86 is 192BPM normalized
        assert!(0.01 > (0.86 - tempo_desc.get_value()).abs());
    }
}
