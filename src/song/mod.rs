//! Song decoding / analysis module.
//!
//! Use decoding, and features-extraction functions from other modules
//! e.g. tempo features, spectral features, etc to build a Song and its
//! corresponding Analysis. For the nitty-gritty decoding details, see
//! the [decoder] module.
//!
//! For implementation of plug-ins for already existing audio players,
//! a look at Library is instead recommended.

#[cfg(feature = "ffmpeg")]
extern crate ffmpeg_next as ffmpeg;
extern crate ndarray;

use crate::chroma::ChromaDesc;
use crate::cue::CueInfo;
use crate::misc::LoudnessDesc;
use crate::temporal::BPMDesc;
use crate::timbral::{SpectralDesc, ZeroCrossingRateDesc};
use crate::{BlissError, BlissResult, FeaturesVersion, SAMPLE_RATE};
use core::ops::Index;
use ndarray::{arr1, Array1};
use std::fmt;
use std::num::NonZeroUsize;

use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use strum::IntoEnumIterator;
use strum_macros::{EnumCount, EnumIter};

pub mod decoder;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, Debug, PartialEq, Clone)]
/// Simple object used to represent a Song, with its path, analysis, and
/// other metadata (artist, genre...)
pub struct Song {
    /// Song's provided file path
    pub path: PathBuf,
    /// Song's artist, read from the metadata
    pub artist: Option<String>,
    /// Song's title, read from the metadata
    pub title: Option<String>,
    /// Song's album name, read from the metadata
    pub album: Option<String>,
    /// Song's album's artist name, read from the metadata
    pub album_artist: Option<String>,
    /// Song's tracked number, read from the metadata
    pub track_number: Option<i32>,
    /// Song's disc number, read from the metadata
    pub disc_number: Option<i32>,
    /// Song's genre, read from the metadata
    pub genre: Option<String>,
    /// bliss analysis results
    pub analysis: Analysis,
    /// The song's duration
    pub duration: Duration,
    /// Version of the features the song was analyzed with.
    /// A simple integer that is bumped every time a breaking change
    /// is introduced in the features.
    pub features_version: FeaturesVersion,
    /// Populated only if the song was extracted from a larger audio file,
    /// through the use of a CUE sheet.
    /// By default, such a song's path would be
    /// `path/to/cue_file.wav/CUE_TRACK00<track_number>`. Using this field,
    /// you can change `song.path` to fit your needs.
    pub cue_info: Option<CueInfo>,
}

impl AsRef<Song> for Song {
    fn as_ref(&self) -> &Song {
        self
    }
}

/// Indexes different fields of an [Analysis](Song::analysis).
///
/// * Example:
/// ```no_run
/// use bliss_audio::{AnalysisIndex, BlissResult, Song};
///
/// fn main() -> BlissResult<()> {
///     // Should be an actual track loaded with a Decoder, but using an empty
///     // song for simplicity's sake
///     let song = Song::default();
///     println!("{}", song.analysis[AnalysisIndex::Tempo]);
///     Ok(())
/// }
/// ```
/// Prints the tempo value of an analysis.
///
/// Note that this should mostly be used for debugging / distance metric
/// customization purposes.
#[derive(Debug, EnumIter, EnumCount)]
pub enum AnalysisIndex {
    /// The song's tempo.
    Tempo,
    /// The song's zero-crossing rate.
    Zcr,
    /// The mean of the song's spectral centroid.
    MeanSpectralCentroid,
    /// The standard deviation of the song's spectral centroid.
    StdDeviationSpectralCentroid,
    /// The mean of the song's spectral rolloff.
    MeanSpectralRolloff,
    /// The standard deviation of the song's spectral rolloff.
    StdDeviationSpectralRolloff,
    /// The mean of the song's spectral flatness.
    MeanSpectralFlatness,
    /// The standard deviation of the song's spectral flatness.
    StdDeviationSpectralFlatness,
    /// The mean of the song's loudness.
    MeanLoudness,
    /// The standard deviation of the song's loudness.
    StdDeviationLoudness,
    /// The proportion of pitch class set 1 (IC1) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma1,
    /// The proportion of pitch class set 2 (IC2) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma2,
    /// The proportion of pitch class set 3 (IC3) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma3,
    /// The proportion of pitch class set 4 (IC4) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma4,
    /// The proportion of pitch class set 5 (IC5) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma5,
    /// The proportion of pitch class set 6 (IC6) compared to the 6 other pitch class sets,
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma6,
    /// The proportion of major triads in the song, compared to the other triads.
    Chroma7,
    /// The proportion of minor triads in the song, compared to the other triads.
    Chroma8,
    /// The proportion of diminished triads in the song, compared to the other triads.
    Chroma9,
    /// The proportion of augmented triads in the song, compared to the other triads.
    Chroma10,
    /// The L2-norm of the IC1-6 (see above).
    Chroma11,
    /// The L2-norm of the IC7-10 (see above).
    Chroma12,
    /// The ratio of the L2-norm of IC7-10 and IC1-6 (proportion of triads vs dyads).
    Chroma13,
}

impl AnalysisIndex {
    /// The features version associated with this analysis index.
    pub const FEATURES_VERSION: FeaturesVersion = FeaturesVersion::LATEST;
}

#[derive(Debug, EnumIter, EnumCount)]
pub enum AnalysisIndexv1 {
    /// The song's tempo.
    Tempo,
    /// The song's zero-crossing rate.
    Zcr,
    /// The mean of the song's spectral centroid.
    MeanSpectralCentroid,
    /// The standard deviation of the song's spectral centroid.
    StdDeviationSpectralCentroid,
    /// The mean of the song's spectral rolloff.
    MeanSpectralRolloff,
    /// The standard deviation of the song's spectral rolloff.
    StdDeviationSpectralRolloff,
    /// The mean of the song's spectral flatness.
    MeanSpectralFlatness,
    /// The standard deviation of the song's spectral flatness.
    StdDeviationSpectralFlatness,
    /// The mean of the song's loudness.
    MeanLoudness,
    /// The standard deviation of the song's loudness.
    StdDeviationLoudness,
    /// The raw value of pitch class set 1 (IC1)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma1,
    /// The raw value of pitch class set 2 (IC2)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma2,
    /// The raw value of pitch class set 3 (IC3)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma3,
    /// The raw value of pitch class set 4 (IC4)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma4,
    /// The raw value of pitch class set 5 (IC5)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma5,
    /// The raw value of pitch class set 6 (IC6)
    /// per this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
    Chroma6,
    /// The proportion of major triads in the song, compared to all the other chroma features
    /// (stays between -0.98 and -0.99) - use the latest features version to avoid this)
    Chroma7,
    /// The proportion of minor triads in the song, compared to all the other chroma features
    /// (stays between -0.98 and -0.99) - use the latest features version to avoid this)
    Chroma8,
    /// The proportion of diminished triads in the song, compared to all the other chroma features
    /// (stays between -0.98 and -0.99) - use the latest features version to avoid this)
    Chroma9,
    /// The proportion of augmented triads in the song, compared to all the other chroma features
    /// (stays between -0.98 and -0.99) - use the latest features version to avoid this)
    Chroma10,
}

impl AnalysisIndexv1 {
    /// The features version associated with this analysis index.
    pub const FEATURES_VERSION: FeaturesVersion = FeaturesVersion::Version1;
}
/// The number of features used in the latest `Analysis` version.
pub const NUMBER_FEATURES: usize = FeaturesVersion::LATEST.feature_count();

/// Object holding the results of the song's analysis.
///
/// Only use it if you want to have an in-depth look of what is
/// happening behind the scene, or make a distance metric yourself.
///
/// Under the hood, it is just an array of f32 holding different numeric
/// features.
///
/// For more info on the different features, build the
/// documentation with private items included using
/// `cargo doc --document-private-items`, and / or read up
/// [this document](https://lelele.io/thesis.pdf), that contains a description
/// on most of the features, except the chroma ones, which are documented
/// directly in this code.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, PartialEq, Clone)]
pub struct Analysis {
    pub(crate) internal_analysis: Vec<f32>,
    // Version of the features the song was analyzed with.
    /// It is bumped every time a change is introduced in the
    /// features that makes them incompatible with previous versions.
    pub features_version: FeaturesVersion,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
/// Various options bliss should be aware of while performing the analysis
/// of a song.
pub struct AnalysisOptions {
    /// The version of the features that should be used for analysis.
    /// Should be kept as the default [FeaturesVersion::LATEST](bliss_audio::FeaturesVersion::LATEST).
    pub features_version: FeaturesVersion,
    /// The number of computer cores that should be used when performing the
    /// analysis of multiple songs.
    pub number_cores: NonZeroUsize,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        let cores = thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap());
        AnalysisOptions {
            features_version: FeaturesVersion::LATEST,
            number_cores: cores,
        }
    }
}

// TODO: group these if this makes sense?
impl Index<AnalysisIndex> for Analysis {
    type Output = f32;

    fn index(&self, index: AnalysisIndex) -> &f32 {
        if self.features_version != AnalysisIndex::FEATURES_VERSION {
            panic!("Tried to index features with incompatible indexes");
        }
        &self.internal_analysis[index as usize]
    }
}

impl Index<AnalysisIndexv1> for Analysis {
    type Output = f32;

    fn index(&self, index: AnalysisIndexv1) -> &f32 {
        if self.features_version != AnalysisIndexv1::FEATURES_VERSION {
            panic!("Tried to index features with incompatible indexes");
        }
        &self.internal_analysis[index as usize]
    }
}

impl fmt::Debug for Analysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = if self.features_version.feature_count() != self.internal_analysis.len() {
            String::from("?")
        } else {
            (self.features_version as u16).to_string()
        };
        let mut debug_struct = f.debug_struct(&format!("Analysis (Version {version})"));
        // If all is good, keep on printing.
        if self.features_version.feature_count() == self.internal_analysis.len() {
            if self.features_version == FeaturesVersion::Version1 {
                for feature in AnalysisIndexv1::iter() {
                    debug_struct.field(&format!("{feature:?}"), &self[feature]);
                }
            } else {
                for feature in AnalysisIndex::iter() {
                    debug_struct.field(&format!("{feature:?}"), &self[feature]);
                }
            }
        }

        debug_struct.finish()?;
        f.write_str(&format!(" /* {:?} */", &self.as_vec()))
    }
}

impl Analysis {
    /// Create a new Analysis object.
    ///
    /// Usually not needed, unless you have already computed and stored
    /// features somewhere, and need to recreate a Song with an already
    /// existing Analysis yourself.
    pub fn new(analysis: Vec<f32>, features_version: FeaturesVersion) -> BlissResult<Analysis> {
        if analysis.len() != features_version.feature_count() {
            return Err(BlissError::ProviderError(format!(
                "Feature count {} does not match the expected version feature count {}",
                analysis.len(),
                features_version.feature_count()
            )));
        }
        Ok(Analysis {
            internal_analysis: analysis,
            features_version,
        })
    }

    /// Return an ndarray `Array1` representing the analysis' features.
    ///
    /// Particularly useful if you want to make a custom distance metric.
    pub fn as_arr1(&self) -> Array1<f32> {
        arr1(&self.internal_analysis)
    }

    /// Return a `Vec<f32>` representing the analysis' features.
    ///
    /// Particularly useful if you want iterate through the values to store
    /// them somewhere.
    pub fn as_vec(&self) -> Vec<f32> {
        self.internal_analysis.to_vec()
    }
}

impl Song {
    /**
     * Analyze a song decoded in `sample_array`. This function should NOT
     * be used manually, unless you want to explore analyzing a sample array you
     * already decoded yourself. Most people will want to use
     * [Decoder::song_from_path](crate::decoder::Decoder::song_from_path)
     * instead to just analyze a file from its path.
     *
     * The current implementation doesn't make use of it,
     * but the song can also be streamed wrt.
     * each descriptor (with the exception of the chroma descriptor which
     * yields worse results when streamed).
     *
     * Useful in the rare cases where the full song is not
     * completely available.
     *
     * If you *do* want to use this with a song already decoded by yourself,
     * the sample format of `sample_array` should be f32le, one channel, and
     * the sampling rate 22050 Hz. Anything other than that will yield wrong
     * results.
     * To double-check that your sample array has the right format, you could run
     * `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le -f hash -hash addler32 -`,
     * which will give you the addler32 checksum of the sample array if the song
     * has been decoded properly. You can then compute the addler32 checksum of your sample
     * array (see `_test_decode` in the tests) and make sure both are the same.
     *
     * (Running `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le` will simply give
     * you the raw sample array as it should look like, if you're not into computing checksums)
     **/
    pub fn analyze(sample_array: &[f32]) -> BlissResult<Analysis> {
        Self::analyze_with_options(sample_array, &AnalysisOptions::default())
    }

    /**
     * This function is the same as [Song::analyze], but allows to compute an
     * analysis using old features_version. Do not use, unless for backwards
     * compatibility.
     **/
    pub fn analyze_with_options(
        sample_array: &[f32],
        analysis_options: &AnalysisOptions,
    ) -> BlissResult<Analysis> {
        let largest_window = vec![
            BPMDesc::WINDOW_SIZE,
            ChromaDesc::WINDOW_SIZE,
            SpectralDesc::WINDOW_SIZE,
            LoudnessDesc::WINDOW_SIZE,
        ]
        .into_iter()
        .max()
        .unwrap();
        if sample_array.len() < largest_window {
            return Err(BlissError::AnalysisError(String::from(
                "empty or too short song.",
            )));
        }

        thread::scope(|s| -> BlissResult<Analysis> {
            let child_tempo = s.spawn(|| -> BlissResult<f32> {
                let mut tempo_desc = BPMDesc::new(SAMPLE_RATE)?;
                let windows = sample_array
                    .windows(BPMDesc::WINDOW_SIZE)
                    .step_by(BPMDesc::HOP_SIZE);

                for window in windows {
                    tempo_desc.do_(window)?;
                }
                Ok(tempo_desc.get_value())
            });

            let child_chroma = s.spawn(|| -> BlissResult<Vec<f32>> {
                let mut chroma_desc = ChromaDesc::new(SAMPLE_RATE, 12);
                chroma_desc.do_(sample_array)?;
                if analysis_options.features_version == FeaturesVersion::Version1 {
                    Ok(chroma_desc.get_values_version_1()?)
                } else {
                    Ok(chroma_desc.get_values()?)
                }
            });

            #[allow(clippy::type_complexity)]
            let child_timbral = s.spawn(|| -> BlissResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                let mut spectral_desc = SpectralDesc::new(SAMPLE_RATE)?;
                let windows = sample_array
                    .windows(SpectralDesc::WINDOW_SIZE)
                    .step_by(SpectralDesc::HOP_SIZE);
                for window in windows {
                    spectral_desc.do_(window)?;
                }
                let centroid = spectral_desc.get_centroid();
                let rolloff = spectral_desc.get_rolloff();
                let flatness = spectral_desc.get_flatness();
                Ok((centroid, rolloff, flatness))
            });

            let child_zcr = s.spawn(|| -> BlissResult<f32> {
                let mut zcr_desc = ZeroCrossingRateDesc::default();
                zcr_desc.do_(sample_array);
                Ok(zcr_desc.get_value())
            });

            let child_loudness = s.spawn(|| -> BlissResult<Vec<f32>> {
                let mut loudness_desc = LoudnessDesc::default();
                let windows = sample_array.chunks(LoudnessDesc::WINDOW_SIZE);

                for window in windows {
                    loudness_desc.do_(window);
                }
                Ok(loudness_desc.get_value())
            });

            // Non-streaming approach for that one
            let tempo = child_tempo.join().unwrap()?;
            let chroma = child_chroma.join().unwrap()?;
            let (centroid, rolloff, flatness) = child_timbral.join().unwrap()?;
            let loudness = child_loudness.join().unwrap()?;
            let zcr = child_zcr.join().unwrap()?;

            let mut result = vec![tempo, zcr];
            result.extend_from_slice(&centroid);
            result.extend_from_slice(&rolloff);
            result.extend_from_slice(&flatness);
            result.extend_from_slice(&loudness);
            result.extend_from_slice(&chroma);
            if result.len() != analysis_options.features_version.feature_count() {
                return Err(BlissError::AnalysisError(
                    "Too many or too little features were provided at the end of
                        the analysis."
                        .to_string(),
                ));
            };
            Analysis::new(result, analysis_options.features_version)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::decoder::ffmpeg::FFmpegDecoder as Decoder;
    #[cfg(feature = "ffmpeg")]
    use crate::decoder::Decoder as DecoderTrait;
    #[cfg(feature = "ffmpeg")]
    use crate::FeaturesVersion;
    use pretty_assertions::assert_eq;
    #[cfg(feature = "ffmpeg")]
    use std::path::Path;

    #[test]
    fn test_analysis_too_small() {
        let error = Song::analyze(&[0.]).unwrap_err();
        assert_eq!(
            error,
            BlissError::AnalysisError(String::from("empty or too short song."))
        );

        let error = Song::analyze(&[]).unwrap_err();
        assert_eq!(
            error,
            BlissError::AnalysisError(String::from("empty or too short song."))
        );
    }

    const SONG_AND_EXPECTED_ANALYSIS: (&str, [f32; NUMBER_FEATURES]) = (
        "data/s16_mono_22_5kHz.flac",
        [
            0.3846389,
            -0.849141,
            -0.75481045,
            -0.8790748,
            -0.63258266,
            -0.7258959,
            -0.7757379,
            -0.8146726,
            0.2716726,
            0.25779057,
            -0.34292513,
            -0.62803423,
            -0.28095096,
            0.08686459,
            0.24446082,
            -0.5723257,
            0.23292065,
            0.19981146,
            -0.58594406,
            -0.06784296,
            -0.06000763,
            -0.58485717,
            -0.07880378,
        ],
    );

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze() {
        let (song, expected_analysis) = SONG_AND_EXPECTED_ANALYSIS;
        let song = Decoder::song_from_path(Path::new(song)).unwrap();
        for (x, y) in song.analysis.as_vec().iter().zip(expected_analysis) {
            assert!(1e-5 > (x - y).abs());
        }
        assert_eq!(FeaturesVersion::LATEST, song.features_version);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze_with_options() {
        let (song, expected_analysis) = (
            "data/s16_mono_22_5kHz.flac",
            [
                0.3846389,
                -0.849141,
                -0.75481045,
                -0.8790748,
                -0.63258266,
                -0.7258959,
                -0.7757379,
                -0.8146726,
                0.2716726,
                0.25779057,
                -0.35661936,
                -0.63578653,
                -0.29593682,
                0.06421304,
                0.21852458,
                -0.581239,
                -0.9466835,
                -0.9481153,
                -0.9820945,
                -0.95968974,
            ],
        );
        let song = Decoder::song_from_path_with_options(
            Path::new(song),
            AnalysisOptions {
                features_version: FeaturesVersion::Version1,
                ..Default::default()
            },
        )
        .unwrap();
        for (x, y) in song.analysis.as_vec().iter().zip(expected_analysis) {
            assert!(1e-5 > (x - y).abs());
        }
        assert_eq!(FeaturesVersion::Version1, song.features_version);
    }

    #[test]
    #[cfg(feature = "symphonia-flac")]
    fn test_analyze_with_symphonia() {
        use crate::decoder::symphonia::SymphoniaDecoder;

        let (song, expected_analysis) = SONG_AND_EXPECTED_ANALYSIS;
        let song = SymphoniaDecoder::song_from_path(Path::new(song)).unwrap();

        for (x, y) in song.analysis.as_vec().iter().zip(expected_analysis) {
            assert!(1e-5 > (x - y).abs(), "{}", (x - y).abs());
        }
        assert_eq!(FeaturesVersion::LATEST, song.features_version);
    }

    #[test]
    #[cfg(feature = "symphonia-flac")]
    fn test_analyze_resampled_with_symphonia() {
        use crate::decoder::symphonia::SymphoniaDecoder;

        let (song, expected_analysis) = (
            "data/s32_stereo_44_1_kHz.flac",
            [
                0.38463664,
                -0.85172224,
                -0.7607465,
                -0.8857495,
                -0.63906085,
                -0.73908424,
                -0.7890965,
                -0.8191868,
                0.33856833,
                0.3246863,
                -0.34292227,
                -0.62803173,
                -0.2809453,
                0.08687115,
                0.2444489,
                -0.5723239,
                0.23292565,
                0.19979525,
                -0.58593845,
                -0.06783122,
                -0.060014784,
                -0.5848569,
                -0.07879859,
            ],
        );

        let song = SymphoniaDecoder::song_from_path(Path::new(song)).unwrap();

        for (x, y) in song.analysis.as_vec().iter().zip(expected_analysis) {
            assert!(0.1 > (x - y).abs(), "{}", (x - y).abs());
        }
        assert_eq!(FeaturesVersion::LATEST, song.features_version);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_index_analysis() {
        let song = Decoder::song_from_path("data/s16_mono_22_5kHz.flac").unwrap();
        assert_eq!(song.analysis[AnalysisIndex::Tempo], 0.3846389);
        assert_eq!(song.analysis[AnalysisIndex::Chroma10], -0.06784296);
    }

    #[test]
    fn test_index_analysis_old_version() {
        let analysis = Analysis::new(
            vec![1.; FeaturesVersion::Version1.feature_count()],
            FeaturesVersion::Version1,
        )
        .unwrap();
        assert_eq!(analysis[AnalysisIndexv1::Tempo], 1.);
        assert_eq!(analysis[AnalysisIndexv1::Chroma10], 1.);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_debug_analysis() {
        let song = Decoder::song_from_path("data/s16_mono_22_5kHz.flac").unwrap();
        assert_eq!(
            "Analysis (Version 2) { Tempo: 0.3846389, Zcr: -0.849141, MeanSpectralCentroid: -0.75481045, StdDeviationSpectralCentroid: -0.8790748, MeanSpectralRolloff: -0.63258266, StdDeviationSpectralRolloff: -0.7258959, MeanSpectralFlatness: -0.7757379, StdDeviationSpectralFlatness: -0.8146726, MeanLoudness: 0.2716726, StdDeviationLoudness: 0.25779057, Chroma1: -0.34292513, Chroma2: -0.62803423, Chroma3: -0.28095096, Chroma4: 0.08686459, Chroma5: 0.24446082, Chroma6: -0.5723257, Chroma7: 0.23292065, Chroma8: 0.19981146, Chroma9: -0.58594406, Chroma10: -0.06784296, Chroma11: -0.06000763, Chroma12: -0.58485717, Chroma13: -0.07880378 } /* [0.3846389, -0.849141, -0.75481045, -0.8790748, -0.63258266, -0.7258959, -0.7757379, -0.8146726, 0.2716726, 0.25779057, -0.34292513, -0.62803423, -0.28095096, 0.08686459, 0.24446082, -0.5723257, 0.23292065, 0.19981146, -0.58594406, -0.06784296, -0.06000763, -0.58485717, -0.07880378] */",
            format!("{:?}", song.analysis),
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_debug_analysis_v1() {
        let song = Decoder::song_from_path_with_options(
            "data/s16_mono_22_5kHz.flac",
            AnalysisOptions {
                features_version: FeaturesVersion::Version1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            "Analysis (Version 1) { Tempo: 0.3846389, Zcr: -0.849141, MeanSpectralCentroid: -0.75481045, StdDeviationSpectralCentroid: -0.8790748, MeanSpectralRolloff: -0.63258266, StdDeviationSpectralRolloff: -0.7258959, MeanSpectralFlatness: -0.7757379, StdDeviationSpectralFlatness: -0.8146726, MeanLoudness: 0.2716726, StdDeviationLoudness: 0.25779057, Chroma1: -0.35661936, Chroma2: -0.63578653, Chroma3: -0.29593682, Chroma4: 0.06421304, Chroma5: 0.21852458, Chroma6: -0.581239, Chroma7: -0.9466835, Chroma8: -0.9481153, Chroma9: -0.9820945, Chroma10: -0.95968974 } /* [0.3846389, -0.849141, -0.75481045, -0.8790748, -0.63258266, -0.7258959, -0.7757379, -0.8146726, 0.2716726, 0.25779057, -0.35661936, -0.63578653, -0.29593682, 0.06421304, 0.21852458, -0.581239, -0.9466835, -0.9481153, -0.9820945, -0.95968974] */",
            format!("{:?}", song.analysis),
        );
    }

    #[test]
    fn test_new_analysis_wrong_number_features() {
        assert!(Analysis::new(vec![1.], FeaturesVersion::Version2).is_err());
    }

    #[test]
    fn test_debug_analysis_wrong_number_fields() {
        let analysis = Analysis {
            internal_analysis: vec![0.; 10],
            features_version: FeaturesVersion::Version1,
        };
        assert_eq!(
            "Analysis (Version ?) /* [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] */",
            format!("{:?}", analysis)
        );
    }

    #[test]
    #[should_panic(expected = "incompatible indexes")]
    fn test_analysis_index_with_wrong_version() {
        let analysis = Analysis::new(
            vec![0.; FeaturesVersion::Version1.feature_count()],
            FeaturesVersion::Version1,
        )
        .unwrap();
        analysis[AnalysisIndex::Chroma13];
    }
}
