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
/// their decoders, but rely on `ffmpeg` to decode songs, and use `FFmpeg::song_from_path`.
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
/// [FFmpeg](ffmpeg::FFmpeg) decoder
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

    /// Analyze songs in `paths`, and return the analyzed [Song] objects through an
    /// [mpsc::IntoIter].
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
    /// implement another decoder and replace `use bliss_audio::decoder::ffmpeg::FFmpeg as Decoder;`
    /// by a custom decoder.
    ///
    #[cfg_attr(
        feature = "ffmpeg",
        doc = r##"
# Example

```no_run
use bliss_audio::{BlissResult};
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::decoder::ffmpeg::FFmpeg as Decoder;

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
use bliss_audio::decoder::ffmpeg::FFmpeg as Decoder;

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
                    info!("Analyzing file '{:?}'", path);
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

#[cfg(feature = "ffmpeg")]
/// The default decoder module. It uses [ffmpeg](https://ffmpeg.org/) in
/// order to decode and resample songs. A very good choice for 99% of
/// the users.
pub mod ffmpeg {
    use super::{Decoder, PreAnalyzedSong};
    use crate::{BlissError, BlissResult, CHANNELS, SAMPLE_RATE};
    use ::log::warn;
    use ffmpeg_next;
    use ffmpeg_next::codec::threading::{Config, Type as ThreadingType};
    use ffmpeg_next::util::channel_layout::ChannelLayout;
    use ffmpeg_next::util::error::Error;
    use ffmpeg_next::util::error::EINVAL;
    use ffmpeg_next::util::format::sample::{Sample, Type};
    use ffmpeg_next::util::frame::audio::Audio;
    use ffmpeg_next::util::log;
    use ffmpeg_next::util::log::level::Level;
    use ffmpeg_next::{media, util};
    use std::sync::mpsc;
    use std::sync::mpsc::Receiver;
    use std::thread;
    use std::time::Duration;

    use std::path::Path;

    /// The actual FFmpeg decoder.
    ///
    /// To use it, one might `use FFmpeg as Decoder;`,
    /// `use super::decoder::Decoder as DecoderTrait;`, and then use
    /// `Decoder::song_from_path`
    pub struct FFmpeg;

    struct SendChannelLayout(ChannelLayout);
    // Safe because the other thread just reads the channel layout
    unsafe impl Send for SendChannelLayout {}

    impl FFmpeg {
        fn resample_frame(
            rx: Receiver<Audio>,
            in_codec_format: Sample,
            sent_in_channel_layout: SendChannelLayout,
            in_rate: u32,
            mut sample_array: Vec<f32>,
            empty_in_channel_layout: bool,
        ) -> BlissResult<Vec<f32>> {
            let in_channel_layout = sent_in_channel_layout.0;
            let mut resample_context = ffmpeg_next::software::resampling::context::Context::get(
                in_codec_format,
                in_channel_layout,
                in_rate,
                Sample::F32(Type::Packed),
                ffmpeg_next::util::channel_layout::ChannelLayout::MONO,
                SAMPLE_RATE,
            )
            .map_err(|e| {
                BlissError::DecodingError(format!(
                    "while trying to allocate resampling context: {e:?}",
                ))
            })?;

            let mut resampled = ffmpeg_next::frame::Audio::empty();
            let mut something_happened = false;
            for mut decoded in rx.iter() {
                #[cfg(not(feature = "ffmpeg_7_0"))]
                let is_channel_layout_empty = decoded.channel_layout() == ChannelLayout::empty();
                #[cfg(feature = "ffmpeg_7_0")]
                let is_channel_layout_empty = decoded.channel_layout().is_empty();

                // If the decoded layout is empty, it means we forced the
                // "in_channel_layout" to something default, not that
                // the format is wrong.
                if empty_in_channel_layout && is_channel_layout_empty {
                    decoded.set_channel_layout(in_channel_layout);
                } else if in_codec_format != decoded.format()
                    || (in_channel_layout != decoded.channel_layout())
                    || in_rate != decoded.rate()
                {
                    warn!("received decoded packet with wrong format; file might be corrupted.");
                    continue;
                }
                something_happened = true;
                resampled = ffmpeg_next::frame::Audio::empty();
                resample_context
                    .run(&decoded, &mut resampled)
                    .map_err(|e| {
                        BlissError::DecodingError(format!("while trying to resample song: {e:?}"))
                    })?;
                FFmpeg::push_to_sample_array(&resampled, &mut sample_array);
            }
            if !something_happened {
                return Ok(sample_array);
            }
            // TODO when ffmpeg-next will be active again: shouldn't we allocate
            // `resampled` again?
            loop {
                match resample_context.flush(&mut resampled).map_err(|e| {
                    BlissError::DecodingError(format!("while trying to resample song: {e:?}"))
                })? {
                    Some(_) => {
                        FFmpeg::push_to_sample_array(&resampled, &mut sample_array);
                    }
                    None => {
                        if resampled.samples() == 0 {
                            break;
                        }
                        FFmpeg::push_to_sample_array(&resampled, &mut sample_array);
                    }
                };
            }
            Ok(sample_array)
        }

        fn push_to_sample_array(frame: &ffmpeg_next::frame::Audio, sample_array: &mut Vec<f32>) {
            if frame.samples() == 0 {
                return;
            }
            // Account for the padding
            let actual_size = util::format::sample::Buffer::size(
                Sample::F32(Type::Packed),
                CHANNELS,
                frame.samples(),
                false,
            );
            let f32_frame: Vec<f32> = frame.data(0)[..actual_size]
                .chunks_exact(4)
                .map(|x| {
                    let mut a: [u8; 4] = [0; 4];
                    a.copy_from_slice(x);
                    f32::from_le_bytes(a)
                })
                .collect();
            sample_array.extend_from_slice(&f32_frame);
        }
    }

    impl Decoder for FFmpeg {
        fn decode(path: &Path) -> BlissResult<PreAnalyzedSong> {
            ffmpeg_next::init().map_err(|e| {
                BlissError::DecodingError(format!(
                    "ffmpeg init error while decoding file '{}': {:?}.",
                    path.display(),
                    e
                ))
            })?;
            log::set_level(Level::Quiet);
            let mut song = PreAnalyzedSong {
                path: path.into(),
                ..Default::default()
            };
            let mut ictx = ffmpeg_next::format::input(&path).map_err(|e| {
                BlissError::DecodingError(format!(
                    "while opening format for file '{}': {:?}.",
                    path.display(),
                    e
                ))
            })?;
            let (mut decoder, stream, expected_sample_number) = {
                let input = ictx.streams().best(media::Type::Audio).ok_or_else(|| {
                    BlissError::DecodingError(format!(
                        "No audio stream found for file '{}'.",
                        path.display()
                    ))
                })?;
                let mut context =
                    ffmpeg_next::codec::context::Context::from_parameters(input.parameters())
                        .map_err(|e| {
                            BlissError::DecodingError(format!(
                                "Could not load the codec context for file '{}': {:?}",
                                path.display(),
                                e
                            ))
                        })?;
                context.set_threading(Config {
                    kind: ThreadingType::Frame,
                    count: 0,
                    #[cfg(not(feature = "ffmpeg_6_0"))]
                    safe: true,
                });
                let decoder = context.decoder().audio().map_err(|e| {
                    BlissError::DecodingError(format!(
                        "when finding decoder for file '{}': {:?}.",
                        path.display(),
                        e
                    ))
                })?;

                // Add SAMPLE_RATE to have one second margin to avoid reallocating if
                // the duration is slightly more than estimated
                // TODO>1.0 another way to get the exact number of samples is to decode
                // everything once, compute the real number of samples from that,
                // allocate the array with that number, and decode again. Check
                // what's faster between reallocating, and just have one second
                // leeway.
                let expected_sample_number = (SAMPLE_RATE as f32 * input.duration() as f32
                    / input.time_base().denominator() as f32)
                    .ceil()
                    + SAMPLE_RATE as f32;
                (decoder, input.index(), expected_sample_number)
            };
            let sample_array: Vec<f32> = Vec::with_capacity(expected_sample_number as usize);
            if let Some(title) = ictx.metadata().get("title") {
                song.title = match title {
                    "" => None,
                    t => Some(t.to_string()),
                };
            };
            if let Some(artist) = ictx.metadata().get("artist") {
                song.artist = match artist {
                    "" => None,
                    a => Some(a.to_string()),
                };
            };
            if let Some(album) = ictx.metadata().get("album") {
                song.album = match album {
                    "" => None,
                    a => Some(a.to_string()),
                };
            };
            if let Some(genre) = ictx.metadata().get("genre") {
                song.genre = match genre {
                    "" => None,
                    g => Some(g.to_string()),
                };
            };
            if let Some(track_number) = ictx.metadata().get("track") {
                song.track_number = match track_number {
                    "" => None,
                    t => t.parse::<i32>().ok(),
                };
            };
            if let Some(disc_number) = ictx.metadata().get("disc") {
                song.disc_number = match disc_number {
                    "" => None,
                    t => {
                        // TODO fix the case where the CD is like "01/04" in tags
                        t.parse::<i32>().ok()
                    }
                };
            };
            if let Some(album_artist) = ictx.metadata().get("album_artist") {
                song.album_artist = match album_artist {
                    "" => None,
                    t => Some(t.to_string()),
                };
            };

            #[cfg(not(feature = "ffmpeg_7_0"))]
            let is_channel_layout_empty = decoder.channel_layout() == ChannelLayout::empty();
            #[cfg(feature = "ffmpeg_7_0")]
            let is_channel_layout_empty = decoder.channel_layout().is_empty();

            let (empty_in_channel_layout, in_channel_layout) = {
                if is_channel_layout_empty {
                    (true, ChannelLayout::default(decoder.channels().into()))
                } else {
                    (false, decoder.channel_layout())
                }
            };
            decoder.set_channel_layout(in_channel_layout);

            let in_channel_layout_to_send = SendChannelLayout(in_channel_layout);

            let (tx, rx) = mpsc::channel();
            let in_codec_format = decoder.format();
            let in_codec_rate = decoder.rate();
            let child = thread::spawn(move || {
                FFmpeg::resample_frame(
                    rx,
                    in_codec_format,
                    in_channel_layout_to_send,
                    in_codec_rate,
                    sample_array,
                    empty_in_channel_layout,
                )
            });
            for (s, packet) in ictx.packets() {
                if s.index() != stream {
                    continue;
                }
                match decoder.send_packet(&packet) {
                    Ok(_) => (),
                    Err(Error::Other { errno: EINVAL }) => {
                        return Err(BlissError::DecodingError(format!(
                            "wrong codec opened for file '{}.",
                            path.display(),
                        )))
                    }
                    Err(Error::Eof) => {
                        warn!(
                            "Premature EOF reached while decoding file '{}'.",
                            path.display()
                        );
                        drop(tx);
                        song.sample_array = child.join().unwrap()?;
                        return Ok(song);
                    }
                    Err(e) => warn!("error while decoding file '{}': {}", path.display(), e),
                };

                loop {
                    let mut decoded = ffmpeg_next::frame::Audio::empty();
                    match decoder.receive_frame(&mut decoded) {
                        Ok(_) => {
                            tx.send(decoded).map_err(|e| {
                            BlissError::DecodingError(format!(
                                "while sending decoded frame to the resampling thread for file '{}': {:?}",
                                path.display(),
                                e,
                            ))
                        })?;
                        }
                        Err(_) => break,
                    }
                }
            }

            // Flush the stream
            let packet = ffmpeg_next::codec::packet::Packet::empty();
            match decoder.send_packet(&packet) {
                Ok(_) => (),
                Err(Error::Other { errno: EINVAL }) => {
                    return Err(BlissError::DecodingError(format!(
                        "wrong codec opened for file '{}'.",
                        path.display()
                    )))
                }
                Err(Error::Eof) => {
                    warn!(
                        "Premature EOF reached while decoding file '{}'.",
                        path.display()
                    );
                    drop(tx);
                    song.sample_array = child.join().unwrap()?;
                    return Ok(song);
                }
                Err(e) => warn!("error while decoding {}: {}", path.display(), e),
            };

            loop {
                let mut decoded = ffmpeg_next::frame::Audio::empty();
                match decoder.receive_frame(&mut decoded) {
                    Ok(_) => {
                        tx.send(decoded).map_err(|e| {
                        BlissError::DecodingError(format!(
                            "while sending decoded frame to the resampling thread for file '{}': {:?}",
                            path.display(),
                            e
                        ))
                    })?;
                    }
                    Err(_) => break,
                }
            }

            drop(tx);
            song.sample_array = child.join().unwrap()?;
            let duration_seconds = song.sample_array.len() as f32 / SAMPLE_RATE as f32;
            song.duration = Duration::from_nanos((duration_seconds * 1e9_f32).round() as u64);
            Ok(song)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::decoder::ffmpeg::FFmpeg as Decoder;
        use crate::decoder::Decoder as DecoderTrait;
        use crate::BlissError;
        use crate::SAMPLE_RATE;
        use adler32::RollingAdler32;
        use pretty_assertions::assert_eq;
        use std::path::Path;

        fn _test_decode(path: &Path, expected_hash: u32) {
            let song = Decoder::decode(path).unwrap();
            let mut hasher = RollingAdler32::new();
            for sample in song.sample_array.iter() {
                hasher.update_buffer(&sample.to_le_bytes());
            }

            assert_eq!(expected_hash, hasher.hash());
        }

        #[test]
        fn test_tags() {
            let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
            assert_eq!(song.artist, Some(String::from("David TMX")));
            assert_eq!(
                song.album_artist,
                Some(String::from("David TMX - Album Artist"))
            );
            assert_eq!(song.title, Some(String::from("Renaissance")));
            assert_eq!(song.album, Some(String::from("Renaissance")));
            assert_eq!(song.track_number, Some(2));
            assert_eq!(song.disc_number, Some(1));
            assert_eq!(song.genre, Some(String::from("Pop")));
            // Test that there is less than 10ms of difference between what
            // the song advertises and what we compute.
            assert!((song.duration.as_millis() as f32 - 11070.).abs() < 10.);
        }

        #[test]
        fn test_empty_tags() {
            let song = Decoder::decode(Path::new("data/no_tags.flac")).unwrap();
            assert_eq!(song.artist, None);
            assert_eq!(song.title, None);
            assert_eq!(song.album, None);
            assert_eq!(song.track_number, None);
            assert_eq!(song.disc_number, None);
            assert_eq!(song.genre, None);
        }

        #[test]
        fn test_resample_multi() {
            let path = Path::new("data/s32_stereo_44_1_kHz.flac");
            let expected_hash = 0xbbcba1cf;
            _test_decode(&path, expected_hash);
        }

        #[test]
        fn test_resample_stereo() {
            let path = Path::new("data/s16_stereo_22_5kHz.flac");
            let expected_hash = 0x1d7b2d6d;
            _test_decode(&path, expected_hash);
        }

        #[test]
        fn test_decode_mono() {
            let path = Path::new("data/s16_mono_22_5kHz.flac");
            // Obtained through
            // ffmpeg -i data/s16_mono_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le
            // -f hash -hash addler32 -
            let expected_hash = 0x5e01930b;
            _test_decode(&path, expected_hash);
        }

        #[test]
        fn test_decode_mp3() {
            let path = Path::new("data/s32_stereo_44_1_kHz.mp3");
            // Obtained through
            // ffmpeg -i data/s16_mono_22_5kHz.mp3 -ar 22050 -ac 1 -c:a pcm_f32le
            // -f hash -hash addler32 -
            let expected_hash = 0x69ca6906;
            _test_decode(&path, expected_hash);
        }

        #[test]
        #[cfg(feature = "ffmpeg")]
        fn test_dont_panic_no_channel_layout() {
            let path = Path::new("data/no_channel.wav");
            Decoder::decode(&path).unwrap();
        }

        #[test]
        fn test_decode_right_capacity_vec() {
            let path = Path::new("data/s16_mono_22_5kHz.flac");
            let song = Decoder::decode(&path).unwrap();
            let sample_array = song.sample_array;
            assert_eq!(
                sample_array.len() + SAMPLE_RATE as usize,
                sample_array.capacity()
            );

            let path = Path::new("data/s32_stereo_44_1_kHz.flac");
            let song = Decoder::decode(&path).unwrap();
            let sample_array = song.sample_array;
            assert_eq!(
                sample_array.len() + SAMPLE_RATE as usize,
                sample_array.capacity()
            );

            let path = Path::new("data/capacity_fix.ogg");
            let song = Decoder::decode(&path).unwrap();
            let sample_array = song.sample_array;
            assert!(sample_array.len() as f32 / sample_array.capacity() as f32 > 0.90);
            assert!(sample_array.len() as f32 / (sample_array.capacity() as f32) < 1.);
        }

        #[test]
        fn test_decode_errors() {
            assert_eq!(
            Decoder::decode(Path::new("nonexistent")).unwrap_err(),
            BlissError::DecodingError(String::from(
                "while opening format for file 'nonexistent': ffmpeg::Error(2: No such file or directory)."
            )),
        );
            assert_eq!(
                Decoder::decode(Path::new("data/picture.png")).unwrap_err(),
                BlissError::DecodingError(String::from(
                    "No audio stream found for file 'data/picture.png'."
                )),
            );
        }

        #[test]
        fn test_decode_wav() {
            let expected_hash = 0xde831e82;
            _test_decode(Path::new("data/piano.wav"), expected_hash);
        }
    }

    #[cfg(all(feature = "bench", feature = "ffmpeg", test))]
    mod bench {
        extern crate test;
        use crate::decoder::ffmpeg::FFmpeg as Decoder;
        use crate::decoder::Decoder as DecoderTrait;
        use std::path::Path;
        use test::Bencher;

        #[bench]
        fn bench_resample_multi(b: &mut Bencher) {
            let path = Path::new("./data/s32_stereo_44_1_kHz.flac");
            b.iter(|| {
                Decoder::decode(&path).unwrap();
            });
        }
    }
}
