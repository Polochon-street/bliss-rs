#![cfg_attr(feature = "bench", feature(test))]
mod chroma;
pub mod library;
mod misc;
mod song;
mod temporal;
mod timbral;
mod utils;

extern crate crossbeam;
extern crate num_cpus;
use thiserror::Error;

const CHANNELS: u16 = 1;
const SAMPLE_RATE: u32 = 22050;

#[derive(Default, Debug, PartialEq, Clone)]
/// Simple object used to represent a Song, with its path, analysis, and
/// other metadata (artist, genre...)
pub struct Song {
    pub path: String,
    pub artist: String,
    pub title: String,
    pub album: String,
    pub track_number: String,
    pub genre: String,
    /// Vec containing analysis, in order: tempo, zero-crossing rate,
    /// mean spectral centroid, std deviation spectral centroid,
    /// mean spectral rolloff, std deviation spectral rolloff
    /// mean spectral_flatness, std deviation spectral flatness,
    /// mean loudness, std deviation loudness
    /// chroma interval feature 1 to 10
    pub analysis: Vec<f32>,
}

#[derive(Error, Debug, PartialEq)]
pub enum BlissError {
    #[error("Error happened while decoding file – {0}")]
    DecodingError(String),
    #[error("Error happened while analyzing file – {0}")]
    AnalysisError(String),
    #[error("Error happened with the music library provider - {0}")]
    ProviderError(String),
}

/// Simple function to bulk analyze a set of songs represented by their
/// absolute paths.
///
/// When making an extension for an audio player, prefer
/// implementing the `Library` trait.
#[doc(hidden)]
pub fn bulk_analyse(paths: Vec<String>) -> Vec<Result<Song, BlissError>> {
    let mut songs = Vec::with_capacity(paths.len());
    let num_cpus = num_cpus::get();

    crossbeam::scope(|s| {
        let mut handles = Vec::with_capacity(paths.len() / num_cpus);
        let mut chunk_number = paths.len() / num_cpus;
        if chunk_number == 0 {
            chunk_number = paths.len();
        }
        for chunk in paths.chunks(chunk_number) {
            handles.push(s.spawn(move |_| {
                let mut result = Vec::with_capacity(chunk.len());
                for path in chunk {
                    let song = Song::new(&path);
                    result.push(song);
                }
                result
            }));
        }

        for handle in handles {
            songs.extend(handle.join().unwrap());
        }
    })
    .unwrap();

    songs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulk_analyse() {
        let results = bulk_analyse(vec![
            String::from("data/s16_mono_22_5kHz.flac"),
            String::from("data/s16_mono_22_5kHz.flac"),
            String::from("nonexistent"),
            String::from("data/s16_stereo_22_5kHz.flac"),
            String::from("nonexistent"),
            String::from("nonexistent"),
            String::from("nonexistent"),
            String::from("nonexistent"),
            String::from("nonexistent"),
            String::from("nonexistent"),
            String::from("nonexistent"),
        ]);
        let mut errored_songs: Vec<String> = results
            .iter()
            .filter_map(|x| x.as_ref().err().map(|x| x.to_string()))
            .collect();
        errored_songs.sort_by(|a, b| a.cmp(b));

        let mut analysed_songs: Vec<String> = results
            .iter()
            .filter_map(|x| x.as_ref().ok().map(|x| x.path.to_string()))
            .collect();
        analysed_songs.sort_by(|a, b| a.cmp(b));

        assert_eq!(
            vec![
                String::from(
                    "Error happened while decoding file – while opening format: ffmpeg::Error(2: No such file or directory)."
                );
                8
            ],
            errored_songs
        );
        assert_eq!(
            vec![
                String::from("data/s16_mono_22_5kHz.flac"),
                String::from("data/s16_mono_22_5kHz.flac"),
                String::from("data/s16_stereo_22_5kHz.flac"),
            ],
            analysed_songs,
        );
    }
}
