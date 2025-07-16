//! Module holding all the nitty-gritty decoding details.
//!
//! Contains the code that uses ffmpeg to decode songs in the [ffmpeg]
//! submodule.
//!
//! Also holds the `Decoder` trait, that you can use to decode songs
//! with the ffmpeg struct that implements that trait, or implement it for
//! other decoders if you do not wish to use ffmpeg, but something else
//! (GStreamer, symphonia...). Using the [ffmpeg] struct as a reference
//! to implement other decoders is probably a good starting point.
use log::info;

use crate::{cue::BlissCue, BlissError, BlissResult, Song, FEATURES_VERSION};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

#[derive(Default, Debug)]
/// A struct used to represent a song that has been decoded, but not analyzed yet.
///
/// Most users will not need to use it, as most users won't implement
/// their decoders, but rely on `ffmpeg` to decode songs, and use `FFmpegDecoder::song_from_path`.
///
/// Since it contains the fully decoded song inside of
/// `PreAnalyzedSong::sample_array`, it can be very large. Users should
/// convert it to a `Song` as soon as possible, since it is this
/// structure's only reason to be.
pub struct PreAnalyzedSong {
    /// Song's provided file path
    pub path: PathBuf,
    /// Song's artist, read from the metadata
    pub artist: Option<String>,
    /// Song's album's artist name, read from the metadata
    pub album_artist: Option<String>,
    /// Song's title, read from the metadata
    pub title: Option<String>,
    /// Song's album name, read from the metadata
    pub album: Option<String>,
    /// Song's tracked number, read from the metadata
    pub track_number: Option<i32>,
    /// Song's disc number, read from the metadata
    pub disc_number: Option<i32>,
    /// Song's genre, read from the metadata
    pub genre: Option<String>,
    /// The song's duration
    pub duration: Duration,
    /// An array of the song's decoded sample which should be,
    /// prior to analysis, resampled to f32le, one channel, with a sampling rate
    /// of 22050 Hz. Anything other than that will yield wrong results.
    /// To double-check that your sample array has the right format, you could run
    /// `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le -f hash -hash addler32 -`,
    /// which will give you the addler32 checksum of the sample array if the song
    /// has been decoded properly. You can then compute the addler32 checksum of your sample
    /// array (see `_test_decode` in the tests) and make sure both are the same.
    ///
    /// (Running `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le` will simply give
    /// you the raw sample array as it should look like, if you're not into computing checksums)
    pub sample_array: Vec<f32>,
}

impl TryFrom<PreAnalyzedSong> for Song {
    type Error = BlissError;

    fn try_from(raw_song: PreAnalyzedSong) -> BlissResult<Song> {
        Ok(Song {
            path: raw_song.path,
            artist: raw_song.artist,
            album_artist: raw_song.album_artist,
            title: raw_song.title,
            album: raw_song.album,
            track_number: raw_song.track_number,
            disc_number: raw_song.disc_number,
            genre: raw_song.genre,
            duration: raw_song.duration,
            analysis: Song::analyze(&raw_song.sample_array)?,
            features_version: FEATURES_VERSION,
            cue_info: None,
        })
    }
}

/// Trait used to implement your own decoder.
///
/// The `decode` function should be implemented so that it
/// decodes and resample a song to one channel with a sampling rate of 22050 Hz
/// and a f32le layout.
/// Once it is implemented, several functions
/// to perform analysis from path(s) are available, such as
/// [song_from_path](Decoder::song_from_path) and
/// [analyze_paths](Decoder::analyze_paths).
///
/// For a reference on how to implement that trait, look at the
/// [FFmpeg](ffmpeg::FFmpegDecoder) decoder
pub trait Decoder {
    /// A function that should decode and resample a song, optionally
    /// extracting the song's metadata such as the artist, the album, etc.
    ///
    /// The output sample array should be resampled to f32le, one channel, with a sampling rate
    /// of 22050 Hz. Anything other than that will yield wrong results.
    /// To double-check that your sample array has the right format, you could run
    /// `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le -f hash -hash addler32 -`,
    /// which will give you the addler32 checksum of the sample array if the song
    /// has been decoded properly. You can then compute the addler32 checksum of your sample
    /// array (see `_test_decode` in the tests) and make sure both are the same.
    ///
    /// (Running `ffmpeg -i path_to_your_song.flac -ar 22050 -ac 1 -c:a pcm_f32le` will simply give
    /// you the raw sample array as it should look like, if you're not into computing checksums)
    fn decode(path: &Path) -> BlissResult<PreAnalyzedSong>;

    /// Returns a decoded [Song] given a file path, or an error if the song
    /// could not be analyzed for some reason.
    ///
    /// # Arguments
    ///
    /// * `path` - A [Path] holding a valid file path to a valid audio file.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file path is invalid, if
    /// the file path points to a file containing no or corrupted audio stream,
    /// or if the analysis could not be conducted to the end for some reason.
    ///
    /// The error type returned should give a hint as to whether it was a
    /// decoding ([DecodingError](BlissError::DecodingError)) or an analysis
    /// ([AnalysisError](BlissError::AnalysisError)) error.
    fn song_from_path<P: AsRef<Path>>(path: P) -> BlissResult<Song> {
        Self::decode(path.as_ref())?.try_into()
    }

    /// Analyze songs in `paths` using multiple threads, and return the
    /// analyzed [Song] objects through an [mpsc::IntoIter].
    ///
    /// Returns an iterator, whose items are a tuple made of
    /// the song path (to display to the user in case the analysis failed),
    /// and a `Result<Song>`.
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
    /// This example uses FFmpeg to decode songs by default, but it is possible to
    /// implement another decoder and replace `use bliss_audio::decoder::ffmpeg::FFmpegDecoder as Decoder;`
    /// by a custom decoder.
    ///
    #[cfg_attr(
        feature = "ffmpeg",
        doc = r##"
# Example

```no_run
use bliss_audio::{BlissResult};
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::decoder::ffmpeg::FFmpegDecoder as Decoder;

fn main() -> BlissResult<()> {
    let paths = vec![String::from("/path/to/song1"), String::from("/path/to/song2")];
    for (path, result) in Decoder::analyze_paths(&paths) {
        match result {
            Ok(song) => println!("Do something with analyzed song {} with title {:?}", song.path.display(), song.title),
            Err(e) => println!("Song at {} could not be analyzed. Failed with: {}", path.display(), e),
        }
    }
    Ok(())
}
```"##
    )]
    fn analyze_paths<P: Into<PathBuf>, F: IntoIterator<Item = P>>(
        paths: F,
    ) -> mpsc::IntoIter<(PathBuf, BlissResult<Song>)> {
        let cores = thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap());
        Self::analyze_paths_with_cores(paths, cores)
    }

    /// Analyze songs in `paths`, and return the analyzed [Song] objects through an
    /// [mpsc::IntoIter]. `number_cores` sets the number of cores the analysis
    /// will use, capped by your system's capacity. Most of the time, you want to
    /// use the simpler `analyze_paths` functions, which autodetects the number
    /// of cores in your system.
    ///
    /// Return an iterator, whose items are a tuple made of
    /// the song path (to display to the user in case the analysis failed),
    /// and a `Result<Song>`.
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
    #[cfg_attr(
        feature = "ffmpeg",
        doc = r##"
# Example

```no_run
use bliss_audio::BlissResult;
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::decoder::ffmpeg::FFmpegDecoder as Decoder;

fn main() -> BlissResult<()> {
    let paths = vec![String::from("/path/to/song1"), String::from("/path/to/song2")];
    for (path, result) in Decoder::analyze_paths(&paths) {
        match result {
            Ok(song) => println!("Do something with analyzed song {} with title {:?}", song.path.display(), song.title),
            Err(e) => println!("Song at {} could not be analyzed. Failed with: {}", path.display(), e),
        }
    }
    Ok(())
}
```"##
    )]
    fn analyze_paths_with_cores<P: Into<PathBuf>, F: IntoIterator<Item = P>>(
        paths: F,
        number_cores: NonZeroUsize,
    ) -> mpsc::IntoIter<(PathBuf, BlissResult<Song>)> {
        let mut cores = thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap());
        // If the number of cores that we have is greater than the number of cores
        // that the user asked, comply with the user - otherwise we set a number
        // that's too great.
        if cores > number_cores {
            cores = number_cores;
        }
        let paths: Vec<PathBuf> = paths.into_iter().map(|p| p.into()).collect();
        #[allow(clippy::type_complexity)]
        let (tx, rx): (
            mpsc::Sender<(PathBuf, BlissResult<Song>)>,
            mpsc::Receiver<(PathBuf, BlissResult<Song>)>,
        ) = mpsc::channel();
        if paths.is_empty() {
            return rx.into_iter();
        }
        let mut handles = Vec::new();
        let mut chunk_length = paths.len() / cores;
        if chunk_length == 0 {
            chunk_length = paths.len();
        }
        for chunk in paths.chunks(chunk_length) {
            let tx_thread = tx.clone();
            let owned_chunk = chunk.to_owned();
            let child = thread::spawn(move || {
                for path in owned_chunk {
                    info!("Analyzing file '{path:?}'");
                    if let Some(extension) = Path::new(&path).extension() {
                        let extension = extension.to_string_lossy().to_lowercase();
                        if extension == "cue" {
                            match BlissCue::<Self>::songs_from_path(&path) {
                                Ok(songs) => {
                                    for song in songs {
                                        tx_thread.send((path.to_owned(), song)).unwrap();
                                    }
                                }
                                Err(e) => tx_thread.send((path.to_owned(), Err(e))).unwrap(),
                            };
                            continue;
                        }
                    }
                    let song = Self::song_from_path(&path);
                    tx_thread.send((path.to_owned(), song)).unwrap();
                }
            });
            handles.push(child);
        }

        rx.into_iter()
    }
}

#[cfg(feature = "symphonia")]
pub mod symphonia;

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
