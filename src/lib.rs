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
//! distance (see [distance](playlist::euclidean_distance) for instance).
//!
//! Once several songs have been analyzed, making a playlist from one Song
//! is as easy as computing distances between that song and the rest, and ordering
//! the songs by distance, ascending.
//!
//! If you want to implement a bliss plugin for an already existing audio
//! player, the [library::Library] struct is a collection of goodies that should prove
//! useful (it contains utilities to store analyzed songs in a self-contained
//! database file, to make playlists directly from the database, etc).
//! [blissify](https://github.com/Polochon-street/blissify-rs/) for both
//! an example of how the [library::Library] struct works, and a real-life demo of bliss
//! implemented for [MPD](https://www.musicpd.org/).
//!
#![cfg_attr(
    feature = "ffmpeg",
    doc = r##"
# Examples

### Analyze & compute the distance between two songs

```no_run
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::decoder::ffmpeg::FFmpeg as Decoder;
use bliss_audio::playlist::euclidean_distance;
use bliss_audio::BlissResult;

fn main() -> BlissResult<()> {
    let song1 = Decoder::song_from_path("/path/to/song1")?;
    let song2 = Decoder::song_from_path("/path/to/song2")?;

    println!(
        "Distance between song1 and song2 is {}",
        euclidean_distance(&song1.analysis.as_arr1(), &song2.analysis.as_arr1())
    );
    Ok(())
}
```

### Make a playlist from a song, discarding failed songs
```no_run
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::decoder::ffmpeg::FFmpeg as Decoder;
use bliss_audio::{
    playlist::{closest_to_songs, euclidean_distance},
    BlissResult, Song,
};


fn main() -> BlissResult<()> {
    let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
    let mut songs: Vec<Song> = Decoder::analyze_paths(&paths).filter_map(|(_, s)| s.ok()).collect();

    // Assuming there is a first song
    let first_song = songs.first().unwrap().to_owned();

    closest_to_songs(&[first_song], &mut songs, &euclidean_distance);

    println!("Playlist is:");
    for song in songs {
        println!("{}", song.path.display());
    }
    Ok(())
}
```
"##
)]
#![cfg_attr(feature = "bench", feature(test))]
#![warn(missing_docs)]
mod chroma;
pub mod cue;
#[cfg(feature = "library")]
pub mod library;
mod misc;
pub mod playlist;
mod song;
mod temporal;
mod timbral;
mod utils;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use thiserror::Error;

pub use song::{decoder, Analysis, AnalysisIndex, Song, NUMBER_FEATURES};

#[allow(dead_code)]
/// The number of channels the raw samples must have to be analyzed by bliss-rs
/// and give correct results.
const CHANNELS: u16 = 1;
/// The sample rate the raw samples must have to be analyzed by bliss-rs
/// and give correct results.
const SAMPLE_RATE: u32 = 22050;
/// Stores the current version of bliss-rs' features.
/// It is bumped every time one or more feature is added, updated or removed,
/// so plug-ins can rescan libraries when there is a major change.
pub const FEATURES_VERSION: u16 = 1;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
/// Umbrella type for bliss error types
pub enum BlissError {
    #[error("error happened while decoding file - {0}")]
    /// An error happened while decoding an (audio) file.
    DecodingError(String),
    #[error("error happened while analyzing file - {0}")]
    /// An error happened during the analysis of the song's samples by bliss.
    AnalysisError(String),
    #[error("error happened with the music library provider - {0}")]
    /// An error happened with the music library provider.
    /// Useful to report errors when you implement bliss for an audio player.
    ProviderError(String),
    // #[cfg(feature = "symphonia")]
    #[error("error happened while decoding file - {0}")]
    /// An error happened while decoding an (audio) file.
    SymphoniaDecoderError(String),
}

/// bliss error type
pub type BlissResult<T> = Result<T, BlissError>;

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
}
