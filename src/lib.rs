//! # bliss audio library
//!
//! bliss is a library for making "smart" audio playlists.
//!
//! The core of the library is the [Song] object, which relates to a
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
//! ### Analyze & compute the distance between two songs
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
//! ### Make a playlist from a song, discarding failed songs
//! ```no_run
//! use bliss_audio::{analyze_paths, BlissResult, Song};
//! use noisy_float::prelude::n32;
//!
//! fn main() -> BlissResult<()> {
//!     let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
//!     let mut songs: Vec<Song> = analyze_paths(&paths)
//!         .filter_map(|(_, s)| s.ok())
//!         .collect();
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
pub mod cue;
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
use crate::cue::BlissCue;
use log::info;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use thiserror::Error;

pub use song::{Analysis, AnalysisIndex, Song, NUMBER_FEATURES};

const CHANNELS: u16 = 1;
const SAMPLE_RATE: u32 = 22050;
/// Stores the current version of bliss-rs' features.
/// It is bumped every time one or more feature is added, updated or removed,
/// so plug-ins can rescan libraries when there is a major change.
pub const FEATURES_VERSION: u16 = 1;

#[derive(Error, Clone, Debug, PartialEq)]
/// Umbrella type for bliss error types
pub enum BlissError {
    #[error("error happened while decoding file ??? {0}")]
    /// An error happened while decoding an (audio) file.
    DecodingError(String),
    #[error("error happened while analyzing file ??? {0}")]
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
/// [mpsc::IntoIter].
///
/// Returns an iterator, whose items are a tuple made of
/// the song path (to display to the user in case the analysis failed),
/// and a Result<Song>.
///
/// # Note
///
/// This function also works with CUE files - it finds the audio files
/// mentionned in the CUE sheet, and then runs the analysis on each song
/// defined by it, returning a proper [Song] object for each one of them.
///
/// Make sure that you don't submit both the audio file along with the CUE
/// sheet if your library uses them, otherwise the audio file will be
/// analyzed as one, single, long song. For instance, with a CUE sheet named
/// `cue-file.cue` with the corresponding audio files `album-1.wav` and
/// `album-2.wav` defined in the CUE sheet, you would just pass `cue-file.cue`
/// to `analyze_paths`, and it will return [Song]s from both files, with
/// more information about which file it is extracted from in the
/// [cue info field](Song::cue_info).
///
/// # Example:
/// ```no_run
/// use bliss_audio::{analyze_paths, BlissResult};
///
/// fn main() -> BlissResult<()> {
///     let paths = vec![String::from("/path/to/song1"), String::from("/path/to/song2")];
///     for (path, result) in analyze_paths(&paths) {
///         match result {
///             Ok(song) => println!("Do something with analyzed song {} with title {:?}", song.path.display(), song.title),
///             Err(e) => println!("Song at {} could not be analyzed. Failed with: {}", path, e),
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn analyze_paths<P: Into<PathBuf>, F: IntoIterator<Item = P>>(
    paths: F,
) -> mpsc::IntoIter<(String, BlissResult<Song>)> {
    let num_cpus = num_cpus::get();
    let paths: Vec<PathBuf> = paths.into_iter().map(|p| p.into()).collect();
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
                info!("Analyzing file '{:?}'", path);
                if let Some(extension) = Path::new(&path).extension() {
                    let extension = extension.to_string_lossy().to_lowercase();
                    if extension == "cue" {
                        match BlissCue::songs_from_path(&path) {
                            Ok(songs) => {
                                for song in songs {
                                    tx_thread
                                        .send((path.to_string_lossy().to_string(), song))
                                        .unwrap();
                                }
                            }
                            Err(e) => tx_thread
                                .send((path.to_string_lossy().to_string(), Err(e)))
                                .unwrap(),
                        };
                        continue;
                    }
                }
                let song = Song::from_path(&path);
                tx_thread
                    .send((path.to_string_lossy().to_string(), song))
                    .unwrap();
            }
        });
        handles.push(child);
    }

    rx.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(test)]
    use pretty_assertions::assert_eq;

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
            String::from("./data/testcue.cue"),
            String::from("./data/white_noise.flac"),
            String::from("definitely-not-existing.foo"),
            String::from("not-existing.foo"),
        ];
        let mut results = analyze_paths(&paths)
            .map(|x| match &x.1 {
                Ok(s) => (true, s.path.to_string_lossy().to_string(), None),
                Err(e) => (false, x.0.to_owned(), Some(e.to_string())),
            })
            .collect::<Vec<_>>();
        results.sort();
        assert_eq!(
            results,
            vec![
                (
                    false,
                    String::from("./data/testcue.cue"),
                    Some(String::from(
                        "error happened while decoding file ??? while \
                            opening format for file './data/not-existing.wav': \
                            ffmpeg::Error(2: No such file or directory)."
                    ),),
                ),
                (
                    false,
                    String::from("definitely-not-existing.foo"),
                    Some(String::from(
                        "error happened while decoding file ??? while \
                            opening format for file 'definitely-not-existing\
                            .foo': ffmpeg::Error(2: No such file or directory)."
                    ),),
                ),
                (
                    false,
                    String::from("not-existing.foo"),
                    Some(String::from(
                        "error happened while decoding file ??? \
                            while opening format for file 'not-existing.foo': \
                            ffmpeg::Error(2: No such file or directory)."
                    ),),
                ),
                (true, String::from("./data/s16_mono_22_5kHz.flac"), None),
                (true, String::from("./data/testcue.flac/CUE_TRACK001"), None),
                (true, String::from("./data/testcue.flac/CUE_TRACK002"), None),
                (true, String::from("./data/testcue.flac/CUE_TRACK003"), None),
                (true, String::from("./data/white_noise.flac"), None),
            ],
        );
    }
}
