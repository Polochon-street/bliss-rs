//! # bliss audio library
//!
//! bliss is a library for making "smart" audio playlists.
//!
//! The core of the library is the `Song` object, which relates to a
//! specific analyzed song and contains its path, title, analysis, and
//! other metadata fields (album, genre...).
//! Analyzing a song is as simple as running `Song::from_path("/path/to/song")`.
//!
//! The [analysis](Song::analysis) field of each song is an array of f32, which
//! makes the comparison between songs easy, by just using e.g. euclidean
//! distance (see [distance](Song::distance) for instance).
//!
//! Once several songs have been analyzed, making a playlist from one Song
//! is as easy as computing distances between that song and the rest, and ordering
//! the songs by distance, ascending.
//!
//! # Examples
//!
//! ## Analyze & compute the distance between two songs
//! ```no_run
//! use bliss_audio::{BlissResult, Song};
//!
//! fn main() -> BlissResult<()> {
//!     let song1 = Song::from_path("/path/to/song1")?;
//!     let song2 = Song::from_path("/path/to/song2")?;
//!
//!     println!("Distance between song1 and song2 is {}", song1.distance(&song2));
//!     Ok(())
//! }
//! ```
//!
//! ### Make a playlist from a song
//! ```no_run
//! use bliss_audio::{BlissResult, Song};
//! use noisy_float::prelude::n32;
//!
//! fn main() -> BlissResult<()> {
//!     let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
//!     let mut songs: Vec<Song> = paths
//!         .iter()
//!         .map(|path| Song::from_path(path))
//!         .collect::<BlissResult<Vec<Song>>>()?;
//!
//!     // Assuming there is a first song
//!     let first_song = songs.first().unwrap().to_owned();
//!
//!     songs.sort_by_cached_key(|song| n32(first_song.distance(&song)));
//!     println!(
//!         "Playlist is: {:?}",
//!         songs
//!             .iter()
//!             .map(|song| song.path.to_string_lossy().to_string())
//!             .collect::<Vec<String>>()
//!     );
//!     Ok(())
//! }
//! ```
#![cfg_attr(feature = "bench", feature(test))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
mod chroma;
mod misc;
pub mod playlist;
mod song;
mod temporal;
mod timbral;
mod utils;

extern crate crossbeam;
extern crate num_cpus;
#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;
use log::info;
use std::sync::mpsc;
use std::thread;
use thiserror::Error;

pub use song::{Analysis, AnalysisIndex, BlissCue, BlissCueFile, Song, NUMBER_FEATURES};

const CHANNELS: u16 = 1;
const SAMPLE_RATE: u32 = 22050;
/// Stores the current version of bliss-rs' features.
/// It is bumped every time one or more feature is added, updated or removed,
/// so plug-ins can rescan libraries when there is a major change.
pub const FEATURES_VERSION: u16 = 1;

#[derive(Error, Clone, Debug, PartialEq)]
/// Umbrella type for bliss error types
pub enum BlissError {
    #[error("error happened while decoding file – {0}")]
    /// An error happened while decoding an (audio) file.
    DecodingError(String),
    #[error("error happened while analyzing file – {0}")]
    /// An error happened during the analysis of the song's samples by bliss.
    AnalysisError(String),
    #[error("error happened with the music library provider - {0}")]
    /// An error happened with the music library provider.
    /// Useful to report errors when you implement bliss for an audio player.
    ProviderError(String),
}

/// bliss error type
pub type BlissResult<T> = Result<T, BlissError>;

/// Analyze songs in `paths`, and return the analyzed [Song] objects through an
/// [mpsc::IntoIter]
///
/// Returns an iterator, whose items are a tuple made of
/// the song path (to display to the user in case the analysis failed),
/// and a Result<Song>.
///
/// * Example:
/// ```no_run
/// use bliss_audio::{analyze_paths, BlissResult};
///
/// fn main() -> BlissResult<()> {
///     let paths = vec![String::from("/path/to/song1"), String::from("/path/to/song2")];
///     for (path, result) in analyze_paths(paths) {
///         match result {
///             Ok(song) => println!("Do something with analyzed song {} with title {:?}", song.path.display(), song.title),
///             Err(e) => println!("Song at {} could not be analyzed. Failed with: {}", path, e),
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn analyze_paths(paths: Vec<String>) -> mpsc::IntoIter<(String, BlissResult<Song>)> {
    let num_cpus = num_cpus::get();

    #[allow(clippy::type_complexity)]
    let (tx, rx): (
        mpsc::Sender<(String, BlissResult<Song>)>,
        mpsc::Receiver<(String, BlissResult<Song>)>,
    ) = mpsc::channel();
    if paths.is_empty() {
        return rx.into_iter();
    }
    let mut handles = Vec::new();
    let mut chunk_length = paths.len() / num_cpus;
    if chunk_length == 0 {
        chunk_length = paths.len();
    }

    for chunk in paths.chunks(chunk_length) {
        let tx_thread = tx.clone();
        let owned_chunk = chunk.to_owned();
        let child = thread::spawn(move || {
            for path in owned_chunk {
                info!("Analyzing file '{}'", path);
                let song = Song::from_path(&path);
                tx_thread.send((path.to_string(), song)).unwrap();
            }
        });
        handles.push(child);
    }

    rx.into_iter()
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
    fn test_analyze_paths() {
        let paths = vec![
            String::from("./data/s16_mono_22_5kHz.flac"),
            String::from("./data/white_noise.flac"),
            String::from("definitely-not-existing.foo"),
            String::from("not-existing.foo"),
        ];
        let mut results = analyze_paths(paths)
            .map(|x| match &x.1 {
                Ok(s) => (true, s.path.to_string_lossy().to_string()),
                Err(_) => (false, x.0.to_owned()),
            })
            .collect::<Vec<_>>();
        results.sort();
        assert_eq!(
            results,
            vec![
                (false, String::from("definitely-not-existing.foo")),
                (false, String::from("not-existing.foo")),
                (true, String::from("./data/s16_mono_22_5kHz.flac")),
                (true, String::from("./data/white_noise.flac")),
            ],
        );
    }
}
