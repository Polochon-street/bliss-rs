//! Song decoding / analysis module.
//!
//! Use decoding, and features-extraction functions from other modules
//! e.g. tempo features, spectral features, etc to build a Song and its
//! corresponding Analysis.
//!
//! For implementation of plug-ins for already existing audio players,
//! a look at Library is instead recommended.

extern crate ffmpeg_next as ffmpeg;
extern crate ndarray;

use crate::chroma::ChromaDesc;
use crate::cue::CueInfo;
use crate::misc::LoudnessDesc;
use crate::temporal::BPMDesc;
use crate::timbral::{SpectralDesc, ZeroCrossingRateDesc};
use crate::{BlissError, BlissResult, SAMPLE_RATE};
use crate::{CHANNELS, FEATURES_VERSION};
use ::log::warn;
use core::ops::Index;
use ffmpeg_next::codec::threading::{Config, Type as ThreadingType};
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::error::Error;
use ffmpeg_next::util::error::EINVAL;
use ffmpeg_next::util::format::sample::{Sample, Type};
use ffmpeg_next::util::frame::audio::Audio;
use ffmpeg_next::util::log;
use ffmpeg_next::util::log::level::Level;
use ffmpeg_next::{media, util};
use ndarray::{arr1, Array1};
use std::convert::TryInto;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};

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
    /// TODO normalize this into an integer
    pub track_number: Option<String>,
    /// Song's genre, read from the metadata (`""` if empty)
    pub genre: Option<String>,
    /// bliss analysis results
    pub analysis: Analysis,
    /// The song's duration
    pub duration: Duration,
    /// Version of the features the song was analyzed with.
    /// A simple integer that is bumped every time a breaking change
    /// is introduced in the features.
    pub features_version: u16,
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

#[derive(Debug, EnumIter, EnumCount)]
/// Indexes different fields of an [Analysis](Song::analysis).
///
/// * Example:
/// ```no_run
/// use bliss_audio::{AnalysisIndex, BlissResult, Song};
///
/// fn main() -> BlissResult<()> {
///     let song = Song::from_path("path/to/song")?;
///     println!("{}", song.analysis[AnalysisIndex::Tempo]);
///     Ok(())
/// }
/// ```
///
/// Prints the tempo value of an analysis.
///
/// Note that this should mostly be used for debugging / distance metric
/// customization purposes.
#[allow(missing_docs)]
pub enum AnalysisIndex {
    Tempo,
    Zcr,
    MeanSpectralCentroid,
    StdDeviationSpectralCentroid,
    MeanSpectralRolloff,
    StdDeviationSpectralRolloff,
    MeanSpectralFlatness,
    StdDeviationSpectralFlatness,
    MeanLoudness,
    StdDeviationLoudness,
    Chroma1,
    Chroma2,
    Chroma3,
    Chroma4,
    Chroma5,
    Chroma6,
    Chroma7,
    Chroma8,
    Chroma9,
    Chroma10,
}
/// The number of features used in `Analysis`
pub const NUMBER_FEATURES: usize = AnalysisIndex::COUNT;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, PartialEq, Clone, Copy)]
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
pub struct Analysis {
    pub(crate) internal_analysis: [f32; NUMBER_FEATURES],
}

impl Index<AnalysisIndex> for Analysis {
    type Output = f32;

    fn index(&self, index: AnalysisIndex) -> &f32 {
        &self.internal_analysis[index as usize]
    }
}

impl fmt::Debug for Analysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("Analysis");
        for feature in AnalysisIndex::iter() {
            debug_struct.field(&format!("{feature:?}"), &self[feature]);
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
    pub fn new(analysis: [f32; NUMBER_FEATURES]) -> Analysis {
        Analysis {
            internal_analysis: analysis,
        }
    }

    /// Return an ndarray `Array1` representing the analysis' features.
    ///
    /// Particularly useful if you want to make a custom distance metric.
    pub fn as_arr1(&self) -> Array1<f32> {
        arr1(&self.internal_analysis)
    }

    /// Return a Vec<f32> representing the analysis' features.
    ///
    /// Particularly useful if you want iterate through the values to store
    /// them somewhere.
    pub fn as_vec(&self) -> Vec<f32> {
        self.internal_analysis.to_vec()
    }
}

// Safe because the other thread just reads the channel layout
struct SendChannelLayout(ChannelLayout);
unsafe impl Send for SendChannelLayout {}

impl Song {
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
    pub fn from_path<P: AsRef<Path>>(path: P) -> BlissResult<Self> {
        let raw_song = Song::decode(path.as_ref())?;

        Ok(Song {
            path: raw_song.path,
            artist: raw_song.artist,
            album_artist: raw_song.album_artist,
            title: raw_song.title,
            album: raw_song.album,
            track_number: raw_song.track_number,
            genre: raw_song.genre,
            duration: raw_song.duration,
            analysis: Song::analyze(&raw_song.sample_array)?,
            features_version: FEATURES_VERSION,
            cue_info: None,
        })
    }

    /**
     * Analyze a song decoded in `sample_array`. This function should NOT
     * be used manually, unless you want to explore analyzing a sample array you
     * already decoded yourself. Most people will want to use
     * [Song::from_path](Song::from_path) instead to just analyze a file from
     * its path.
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
     * the sampling rate 22050 Hz. Anything other than that will yield aberrant
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
            let child_tempo = s.spawn(|| {
                let mut tempo_desc = BPMDesc::new(SAMPLE_RATE)?;
                let windows = sample_array
                    .windows(BPMDesc::WINDOW_SIZE)
                    .step_by(BPMDesc::HOP_SIZE);

                for window in windows {
                    tempo_desc.do_(window)?;
                }
                Ok(tempo_desc.get_value())
            });

            let child_chroma = s.spawn(|| {
                let mut chroma_desc = ChromaDesc::new(SAMPLE_RATE, 12);
                chroma_desc.do_(sample_array)?;
                Ok(chroma_desc.get_values())
            });

            #[allow(clippy::type_complexity)]
            let child_timbral = s.spawn(|| {
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

            let child_zcr = s.spawn(|| {
                let mut zcr_desc = ZeroCrossingRateDesc::default();
                zcr_desc.do_(sample_array);
                Ok(zcr_desc.get_value())
            });

            let child_loudness = s.spawn(|| {
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
            let array: [f32; NUMBER_FEATURES] = result.try_into().map_err(|_| {
                BlissError::AnalysisError(
                    "Too many or too little features were provided at the end of
                        the analysis."
                        .to_string(),
                )
            })?;
            Ok(Analysis::new(array))
        })
    }

    pub(crate) fn decode(path: &Path) -> BlissResult<InternalSong> {
        ffmpeg::init().map_err(|e| {
            BlissError::DecodingError(format!(
                "ffmpeg init error while decoding file '{}': {:?}.",
                path.display(),
                e
            ))
        })?;
        log::set_level(Level::Quiet);
        let mut song = InternalSong {
            path: path.into(),
            ..Default::default()
        };
        let mut ictx = ffmpeg::format::input(&path).map_err(|e| {
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
            let mut context = ffmpeg::codec::context::Context::from_parameters(input.parameters())
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
                t => Some(t.to_string()),
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
            resample_frame(
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
                let mut decoded = ffmpeg::frame::Audio::empty();
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
        let packet = ffmpeg::codec::packet::Packet::empty();
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
            let mut decoded = ffmpeg::frame::Audio::empty();
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
        // Waiting here ensures that ChannelLayout doesn't get dropped in the middle
        song.sample_array = child.join().unwrap()?;
        let duration_seconds = song.sample_array.len() as f32 / SAMPLE_RATE as f32;
        song.duration = Duration::from_nanos((duration_seconds * 1e9_f32).round() as u64);
        Ok(song)
    }
}

#[derive(Default, Debug)]
pub(crate) struct InternalSong {
    pub path: PathBuf,
    pub artist: Option<String>,
    pub album_artist: Option<String>,
    pub title: Option<String>,
    pub album: Option<String>,
    pub track_number: Option<String>,
    pub genre: Option<String>,
    pub duration: Duration,
    pub sample_array: Vec<f32>,
}

fn resample_frame(
    rx: Receiver<Audio>,
    in_codec_format: Sample,
    sent_in_channel_layout: SendChannelLayout,
    in_rate: u32,
    mut sample_array: Vec<f32>,
    empty_in_channel_layout: bool,
) -> BlissResult<Vec<f32>> {
    let in_channel_layout = sent_in_channel_layout.0;
    let mut resample_context = ffmpeg::software::resampling::context::Context::get(
        in_codec_format,
        in_channel_layout,
        in_rate,
        Sample::F32(Type::Packed),
        ffmpeg::util::channel_layout::ChannelLayout::MONO,
        SAMPLE_RATE,
    )
    .map_err(|e| {
        BlissError::DecodingError(format!(
            "while trying to allocate resampling context: {e:?}",
        ))
    })?;

    let mut resampled = ffmpeg::frame::Audio::empty();
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
        resampled = ffmpeg::frame::Audio::empty();
        resample_context
            .run(&decoded, &mut resampled)
            .map_err(|e| {
                BlissError::DecodingError(format!("while trying to resample song: {e:?}"))
            })?;
        push_to_sample_array(&resampled, &mut sample_array);
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
                push_to_sample_array(&resampled, &mut sample_array);
            }
            None => {
                if resampled.samples() == 0 {
                    break;
                }
                push_to_sample_array(&resampled, &mut sample_array);
            }
        };
    }
    Ok(sample_array)
}

fn push_to_sample_array(frame: &ffmpeg::frame::Audio, sample_array: &mut Vec<f32>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use adler32::RollingAdler32;
    use pretty_assertions::assert_eq;
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

    #[test]
    fn test_analyze() {
        let song = Song::from_path(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let expected_analysis = vec![
            0.3846389,
            -0.849141,
            -0.75481045,
            -0.8790748,
            -0.63258266,
            -0.7258959,
            -0.775738,
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
        ];
        for (x, y) in song.analysis.as_vec().iter().zip(expected_analysis) {
            assert!(0.01 > (x - y).abs());
        }
        assert_eq!(FEATURES_VERSION, song.features_version);
    }

    fn _test_decode(path: &Path, expected_hash: u32) {
        let song = Song::decode(path).unwrap();
        let mut hasher = RollingAdler32::new();
        for sample in song.sample_array.iter() {
            hasher.update_buffer(&sample.to_le_bytes());
        }

        assert_eq!(expected_hash, hasher.hash());
    }

    #[test]
    fn test_tags() {
        let song = Song::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        assert_eq!(song.artist, Some(String::from("David TMX")));
        assert_eq!(
            song.album_artist,
            Some(String::from("David TMX - Album Artist"))
        );
        assert_eq!(song.title, Some(String::from("Renaissance")));
        assert_eq!(song.album, Some(String::from("Renaissance")));
        assert_eq!(song.track_number, Some(String::from("02")));
        assert_eq!(song.genre, Some(String::from("Pop")));
        // Test that there is less than 10ms of difference between what
        // the song advertises and what we compute.
        assert!((song.duration.as_millis() as f32 - 11070.).abs() < 10.);
    }

    #[test]
    fn test_empty_tags() {
        let song = Song::decode(Path::new("data/no_tags.flac")).unwrap();
        assert_eq!(song.artist, None);
        assert_eq!(song.title, None);
        assert_eq!(song.album, None);
        assert_eq!(song.track_number, None);
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
    fn test_dont_panic_no_channel_layout() {
        let path = Path::new("data/no_channel.wav");
        Song::decode(&path).unwrap();
    }

    #[test]
    fn test_decode_right_capacity_vec() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        let song = Song::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len() + SAMPLE_RATE as usize,
            sample_array.capacity()
        );

        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let song = Song::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len() + SAMPLE_RATE as usize,
            sample_array.capacity()
        );

        let path = Path::new("data/capacity_fix.ogg");
        let song = Song::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert!(sample_array.len() as f32 / sample_array.capacity() as f32 > 0.90);
        assert!(sample_array.len() as f32 / (sample_array.capacity() as f32) < 1.);
    }

    #[test]
    fn test_decode_errors() {
        assert_eq!(
            Song::decode(Path::new("nonexistent")).unwrap_err(),
            BlissError::DecodingError(String::from(
                "while opening format for file 'nonexistent': ffmpeg::Error(2: No such file or directory)."
            )),
        );
        assert_eq!(
            Song::decode(Path::new("data/picture.png")).unwrap_err(),
            BlissError::DecodingError(String::from(
                "No audio stream found for file 'data/picture.png'."
            )),
        );
    }

    #[test]
    fn test_index_analysis() {
        let song = Song::from_path("data/s16_mono_22_5kHz.flac").unwrap();
        assert_eq!(song.analysis[AnalysisIndex::Tempo], 0.3846389);
        assert_eq!(song.analysis[AnalysisIndex::Chroma10], -0.95968974);
    }

    #[test]
    fn test_decode_wav() {
        let expected_hash = 0xde831e82;
        _test_decode(Path::new("data/piano.wav"), expected_hash);
    }

    #[test]
    fn test_debug_analysis() {
        let song = Song::from_path("data/s16_mono_22_5kHz.flac").unwrap();
        assert_eq!(
            "Analysis { Tempo: 0.3846389, Zcr: -0.849141, MeanSpectralCentroid: \
            -0.75481045, StdDeviationSpectralCentroid: -0.8790748, MeanSpectralR\
            olloff: -0.63258266, StdDeviationSpectralRolloff: -0.7258959, MeanSp\
            ectralFlatness: -0.7757379, StdDeviationSpectralFlatness: -0.8146726\
            , MeanLoudness: 0.2716726, StdDeviationLoudness: 0.25779057, Chroma1\
            : -0.35661936, Chroma2: -0.63578653, Chroma3: -0.29593682, Chroma4: \
            0.06421304, Chroma5: 0.21852458, Chroma6: -0.581239, Chroma7: -0.946\
            6835, Chroma8: -0.9481153, Chroma9: -0.9820945, Chroma10: -0.95968974 } \
            /* [0.3846389, -0.849141, -0.75481045, -0.8790748, -0.63258266, -0.\
            7258959, -0.7757379, -0.8146726, 0.2716726, 0.25779057, -0.35661936, \
            -0.63578653, -0.29593682, 0.06421304, 0.21852458, -0.581239, -0.946\
            6835, -0.9481153, -0.9820945, -0.95968974] */",
            format!("{:?}", song.analysis),
        );
    }
}

#[cfg(all(feature = "bench", test))]
mod bench {
    extern crate test;
    use crate::Song;
    use std::path::Path;
    use test::Bencher;

    #[bench]
    fn bench_resample_multi(b: &mut Bencher) {
        let path = Path::new("./data/s32_stereo_44_1_kHz.flac");
        b.iter(|| {
            Song::decode(&path).unwrap();
        });
    }
}
