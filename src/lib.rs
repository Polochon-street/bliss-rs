//! # bliss audio library
//!
//! bliss is a library for making "smart" audio playlists.
//!
//! The core of the library is the `Song` object, which relates to a
//! specific analyzed song and contains its path, title, analysis, and
//! other metadata fields (album, genre...).
//! Analyzing a song is as simple as running `Song::new("/path/to/song")`.
//!
//! The [analysis](Song::analysis) field of each song is an array of f32, which makes the
//! comparison between songs easy, by just using euclidean distance (see
//! [distance](Song::distance) for instance).
//!
//! Once several songs have been analyzed, making a playlist from one Song
//! is as easy as computing distances between that song and the rest, and ordering
//! the songs by distance, ascending.
//!
//! It is also convenient to make plug-ins for existing audio players.
//! It should be as easy as implementing the necessary traits for [Library].
//! A reference implementation for the MPD player is available
//! [here](https://github.com/Polochon-street/blissify-rs)
//!
//! # Examples
//!
//! ## Analyze & compute the distance between two songs
//! ```no_run
//! use bliss_audio::{BlissError, Song};
//! 
//! fn main() -> Result<(), BlissError> {
//!     let song1 = Song::new("/path/to/song1")?;
//!     let song2 = Song::new("/path/to/song2")?;
//!
//!     println!("Distance between song1 and song2 is {}", song1.distance(&song2));
//!     Ok(())
//! }
//! ```
//! 
//! ### Make a playlist from a song
//! ```no_run
//! use bliss_audio::{BlissError, Song};
//! use ndarray::{arr1, Array};
//! use noisy_float::prelude::n32;
//! 
//! fn main() -> Result<(), BlissError> {
//!     let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
//!     let mut songs: Vec<Song> = paths
//!         .iter()
//!         .map(|path| Song::new(path))
//!         .collect::<Result<Vec<Song>, BlissError>>()?;
//! 
//!     // Assuming there is a first song
//!     let first_song = songs.first().unwrap().to_owned();
//!
//!     songs.sort_by_cached_key(|song| n32(first_song.distance(&song)));
//!     println!(
//!         "Playlist is: {:?}",
//!         songs
//!             .iter()
//!             .map(|song| &song.path)
//!             .collect::<Vec<&String>>()
//!     );
//!     Ok(())
//! }
//! ```
#![cfg_attr(feature = "bench", feature(test))]
#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
mod chroma;
mod library;
mod misc;
mod song;
mod temporal;
mod timbral;
mod utils;

extern crate crossbeam;
extern crate num_cpus;
#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;
use thiserror::Error;

pub use library::Library;
pub use song::{Analysis, AnalysisIndex, Song};

const CHANNELS: u16 = 1;
const SAMPLE_RATE: u32 = 22050;

#[derive(Error, Clone, Debug, PartialEq)]
/// Umbrella type for bliss error types
pub enum BlissError {
    #[error("error happened while decoding file – {0}")]
    /// An error happened while decoding an (audio) file
    DecodingError(String),
    #[error("error happened while analyzing file – {0}")]
    /// An error happened during the analysis of the samples by bliss
    AnalysisError(String),
    #[error("error happened with the music library provider - {0}")]
    /// An error happened with the music library provider.
    /// Useful to report errors when you implement the [Library] trait.
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
    fn test_send_song() {
        fn assert_send<T: Send>() {}
        assert_send::<Song>();
    }

    #[test]
    fn test_sync_song() {
        fn assert_sync<T: Send>() {}
        assert_sync::<Song>();
    }

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
                    "error happened while decoding file – while opening format: ffmpeg::Error(2: No such file or directory)."
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
