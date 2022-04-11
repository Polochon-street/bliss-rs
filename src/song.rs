//! Song decoding / analysis module.
//!
//! Use decoding, and features-extraction functions from other modules
//! e.g. tempo features, spectral features, etc to build a Song and its
//! corresponding Analysis.
//!
//! For implementation of plug-ins for already existing audio players,
//! a look at Library is instead recommended.

extern crate crossbeam;
extern crate ffmpeg_next as ffmpeg;
extern crate ndarray;
extern crate ndarray_npy;

use crate::chroma::ChromaDesc;
use crate::misc::LoudnessDesc;
#[cfg(doc)]
use crate::playlist;
use crate::playlist::{closest_to_first_song, dedup_playlist, euclidean_distance, DistanceMetric};
use crate::temporal::BPMDesc;
use crate::timbral::{SpectralDesc, ZeroCrossingRateDesc};
use crate::{BlissError, BlissResult, SAMPLE_RATE};
use crate::{CHANNELS, FEATURES_VERSION};
use ::log::warn;
use core::ops::Index;
use crossbeam::thread;
#[cfg(feature = "cue")]
use cue::cd::CD;
#[cfg(feature = "cue")]
use cue::cd_text::PTI;
use ffmpeg_next::codec::threading::{Config, Type as ThreadingType};
use ffmpeg_next::util;
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::error::Error;
use ffmpeg_next::util::error::EINVAL;
use ffmpeg_next::util::format::sample::{Sample, Type};
use ffmpeg_next::util::frame::audio::Audio;
use ffmpeg_next::util::log;
use ffmpeg_next::util::log::level::Level;
use ndarray::{arr1, Array1};
#[cfg(feature = "cue")]
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread as std_thread;
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
    /// Song's tracked number, read from the metadata
    pub track_number: Option<String>,
    /// Song's genre, read from the metadata (`""` if empty)
    pub genre: Option<String>,
    /// bliss analysis results
    pub analysis: Analysis,
    /// Version of the features the song was analyzed with.
    /// A simple integer that is bumped every time a breaking change
    /// is introduced in the features.
    pub features_version: u16,
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
    internal_analysis: [f32; NUMBER_FEATURES],
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
            debug_struct.field(&format!("{:?}", feature), &self[feature]);
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

    /// Compute distance between two analysis using a user-provided distance
    /// metric. You most likely want to use `song.custom_distance` directly
    /// rather than this function.
    ///
    /// For this function to be integrated properly with the rest
    /// of bliss' parts, it should be a valid distance metric, i.e.:
    /// 1. For X, Y real vectors, d(X, Y) = 0 ⇔ X = Y
    /// 2. For X, Y real vectors, d(X, Y) >= 0
    /// 3. For X, Y real vectors, d(X, Y) = d(Y, X)
    /// 4. For X, Y, Z real vectors d(X, Y) ≤ d(X + Z) + d(Z, Y)
    ///
    /// Note that almost all distance metrics you will find obey these
    /// properties, so don't sweat it too much.
    pub fn custom_distance(&self, other: &Self, distance: impl DistanceMetric) -> f32 {
        distance(&self.as_arr1(), &other.as_arr1())
    }
}

impl Song {
    #[allow(dead_code)]
    /// Compute the distance between the current song and any given
    /// Song.
    ///
    /// The smaller the number, the closer the songs; usually more useful
    /// if compared between several songs
    /// (e.g. if song1.distance(song2) < song1.distance(song3), then song1 is
    /// closer to song2 than it is to song3.
    ///
    /// Currently uses the euclidean distance, but this can change in an
    /// upcoming release if another metric performs better.
    pub fn distance(&self, other: &Self) -> f32 {
        self.analysis
            .custom_distance(&other.analysis, euclidean_distance)
    }

    /// Compute distance between two songs using a user-provided distance
    /// metric.
    ///
    /// For this function to be integrated properly with the rest
    /// of bliss' parts, it should be a valid distance metric, i.e.:
    /// 1. For X, Y real vectors, d(X, Y) = 0 ⇔ X = Y
    /// 2. For X, Y real vectors, d(X, Y) >= 0
    /// 3. For X, Y real vectors, d(X, Y) = d(Y, X)
    /// 4. For X, Y, Z real vectors d(X, Y) ≤ d(X + Z) + d(Z, Y)
    ///
    /// Note that almost all distance metrics you will find obey these
    /// properties, so don't sweat it too much.
    pub fn custom_distance(&self, other: &Self, distance: impl DistanceMetric) -> f32 {
        self.analysis.custom_distance(&other.analysis, distance)
    }

    /// Orders songs in `pool` by proximity to `self`, using the distance
    /// metric `distance` to compute the order.
    /// Basically return a playlist from songs in `pool`, starting
    /// from `self`, using `distance` (some distance metrics can
    /// be found in the [playlist] module).
    ///
    /// Note that contrary to [Song::closest_from_pool], `self` is NOT added
    /// to the beginning of the returned vector.
    ///
    /// No deduplication is ran either; if you're looking for something easy
    /// that works "out of the box", use [Song::closest_from_pool].
    pub fn closest_from_pool_custom(
        &self,
        pool: Vec<Self>,
        distance: impl DistanceMetric,
    ) -> Vec<Self> {
        let mut pool = pool;
        closest_to_first_song(self, &mut pool, distance);
        pool
    }

    /// Order songs in `pool` by proximity to `self`.
    /// Convenience method to return a playlist from songs in `pool`,
    /// starting from `self`.
    ///
    /// The distance is already chosen, deduplication is ran, and the first song
    /// is added to the top of the playlist, to make everything easier.
    ///
    /// If you want more control over which distance metric is chosen,
    /// run deduplication manually, etc, use [Song::closest_from_pool_custom].
    pub fn closest_from_pool(&self, pool: Vec<Self>) -> Vec<Self> {
        let mut playlist = vec![self.to_owned()];
        playlist.extend_from_slice(&pool);
        closest_to_first_song(self, &mut playlist, euclidean_distance);
        dedup_playlist(&mut playlist, None);
        playlist
    }

    /// Return a decoded [Song] given a file path, or an error if the song
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
            title: raw_song.title,
            album: raw_song.album,
            track_number: raw_song.track_number,
            genre: raw_song.genre,
            analysis: Song::analyze(raw_song.sample_array)?,
            features_version: FEATURES_VERSION,
        })
    }

    #[cfg(feature = "cue")]
    /// Return a list of [Song]s that are listed in a CUE sheet, decoding the
    /// tracks found in the corresponding audio file.
    ///
    /// # Arguments
    ///
    /// * `path` - A [Path] holding a valid file path to a proper CUE sheet.
    ///
    /// # Returns
    ///
    /// Return a list of songs, with populated tags and paths formatted like
    /// `path_to_audio_file.flac/CUE_TRACK001`.
    ///
    /// # Errors
    ///
    /// This function will return an error directly if reading the CUE sheet
    /// did not work for any reason, as well as individual errors if they
    /// happen for specific tracks.
    pub fn from_cue<P: AsRef<Path>>(cue_path: P) -> BlissResult<Vec<BlissResult<Self>>> {
        let cd = CD::parse_file(cue_path.as_ref().to_path_buf()).map_err(|e| {
            BlissError::DecodingError(format!("could not parse CUE file: {:?}.", e))
        })?;

        // Apparently one CUE file can have several filepaths.
        let mut tracks = cd.tracks();
        tracks.sort_by_key(|s| s.get_filename());
        tracks.dedup_by_key(|s| s.get_filename());

        // Hold a HashMap like
        // {"path/to/file1": sample_array1, "path/to/file2": sample_array2}
        let mut raw_whole_albums = HashMap::new();
        for path in tracks.iter().map(|s| s.get_filename()) {
            let path = match cue_path.as_ref().parent() {
                Some(parent) => format!("{}/{}", parent.to_string_lossy(), path),
                None => path,
            };
            raw_whole_albums.insert(path.to_owned(), Song::decode(&PathBuf::from(path))?);
        }

        let mut songs = Vec::new();
        for (index, track) in cd.tracks().iter().enumerate() {
            let path = match cue_path.as_ref().parent() {
                Some(parent) => format!("{}/{}", parent.to_string_lossy(), track.get_filename()),
                None => track.get_filename(),
            };
            let whole_sample_array = &raw_whole_albums[&path].sample_array;
            // Start time, in seconds
            let start = track.get_start() as f32 / 75.;
            // End time, in seconds
            let end = start + track.get_length() as f32 / 75.;

            // If track.get_length() is -1, then it means it's the last song,
            // so act accordingly.
            let analysis = if end > start {
                Song::analyze(
                    whole_sample_array[(start * SAMPLE_RATE as f32) as usize
                        ..(end * SAMPLE_RATE as f32) as usize]
                        .to_vec(),
                )
            } else {
                Song::analyze(whole_sample_array[(start * SAMPLE_RATE as f32) as usize..].to_vec())
            };
            let cd_text = track.get_cdtext();
            let song = if let Ok(a) = analysis {
                Ok(Song {
                    path: PathBuf::from(format!("{}/CUE_TRACK{:03}", path, index + 1)),
                    artist: cd_text.read(PTI::Performer),
                    title: cd_text.read(PTI::Title),
                    album: cd.get_cdtext().read(PTI::Title),
                    genre: cd.get_cdtext().read(PTI::Genre),
                    analysis: a,
                    features_version: FEATURES_VERSION,
                    track_number: Some(format!("{:02}", index + 1)),
                })
            } else {
                Err(analysis.unwrap_err())
            };
            songs.push(song);
        }

        Ok(songs)
    }

    /**
     * Analyze a song decoded in `sample_array`, with one channel @ 22050 Hz.
     *
     * The current implementation doesn't make use of it,
     * but the song can also be streamed wrt.
     * each descriptor (with the exception of the chroma descriptor which
     * yields worse results when streamed).
     *
     * Useful in the rare cases where the full song is not
     * completely available.
     **/
    fn analyze(sample_array: Vec<f32>) -> BlissResult<Analysis> {
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

        thread::scope(|s| {
            let child_tempo: thread::ScopedJoinHandle<'_, BlissResult<f32>> = s.spawn(|_| {
                let mut tempo_desc = BPMDesc::new(SAMPLE_RATE)?;
                let windows = sample_array
                    .windows(BPMDesc::WINDOW_SIZE)
                    .step_by(BPMDesc::HOP_SIZE);

                for window in windows {
                    tempo_desc.do_(window)?;
                }
                Ok(tempo_desc.get_value())
            });

            let child_chroma: thread::ScopedJoinHandle<'_, BlissResult<Vec<f32>>> = s.spawn(|_| {
                let mut chroma_desc = ChromaDesc::new(SAMPLE_RATE, 12);
                chroma_desc.do_(&sample_array)?;
                Ok(chroma_desc.get_values())
            });

            #[allow(clippy::type_complexity)]
            let child_timbral: thread::ScopedJoinHandle<
                '_,
                BlissResult<(Vec<f32>, Vec<f32>, Vec<f32>)>,
            > = s.spawn(|_| {
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

            let child_zcr: thread::ScopedJoinHandle<'_, BlissResult<f32>> = s.spawn(|_| {
                let mut zcr_desc = ZeroCrossingRateDesc::default();
                zcr_desc.do_(&sample_array);
                Ok(zcr_desc.get_value())
            });

            let child_loudness: thread::ScopedJoinHandle<'_, BlissResult<Vec<f32>>> =
                s.spawn(|_| {
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
        .unwrap()
    }

    pub(crate) fn decode(path: &Path) -> BlissResult<InternalSong> {
        ffmpeg::init()
            .map_err(|e| BlissError::DecodingError(format!("ffmpeg init error: {:?}.", e)))?;
        log::set_level(Level::Quiet);
        let mut song = InternalSong {
            path: path.into(),
            ..Default::default()
        };
        let mut format = ffmpeg::format::input(&path)
            .map_err(|e| BlissError::DecodingError(format!("while opening format: {:?}.", e)))?;
        let (mut codec, stream, expected_sample_number) = {
            let stream = format
                .streams()
                .find(|s| s.parameters().medium() == ffmpeg::media::Type::Audio)
                .ok_or_else(|| BlissError::DecodingError(String::from("No audio stream found.")))?;
            let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| {
                    BlissError::DecodingError(format!("Could not load the codec context: {:?}", e))
                })?;
            context.set_threading(Config {
                kind: ThreadingType::Frame,
                count: 0,
                safe: true,
            });
            let codec = context
                .decoder()
                .audio()
                .map_err(|e| BlissError::DecodingError(format!("when finding codec: {:?}.", e)))?;
            // Add SAMPLE_RATE to have one second margin to avoid reallocating if
            // the duration is slightly more than estimated
            // TODO>1.0 another way to get the exact number of samples is to decode
            // everything once, compute the real number of samples from that,
            // allocate the array with that number, and decode again. Check
            // what's faster between reallocating, and just have one second
            // leeway.
            let expected_sample_number = (SAMPLE_RATE as f32 * stream.duration() as f32
                / stream.time_base().denominator() as f32)
                .ceil()
                + SAMPLE_RATE as f32;
            (codec, stream.index(), expected_sample_number)
        };
        let sample_array: Vec<f32> = Vec::with_capacity(expected_sample_number as usize);
        if let Some(title) = format.metadata().get("title") {
            song.title = match title {
                "" => None,
                t => Some(t.to_string()),
            };
        };
        if let Some(artist) = format.metadata().get("artist") {
            song.artist = match artist {
                "" => None,
                a => Some(a.to_string()),
            };
        };
        if let Some(album) = format.metadata().get("album") {
            song.album = match album {
                "" => None,
                a => Some(a.to_string()),
            };
        };
        if let Some(genre) = format.metadata().get("genre") {
            song.genre = match genre {
                "" => None,
                g => Some(g.to_string()),
            };
        };
        if let Some(track_number) = format.metadata().get("track") {
            song.track_number = match track_number {
                "" => None,
                t => Some(t.to_string()),
            };
        };
        let in_channel_layout = {
            if codec.channel_layout() == ChannelLayout::empty() {
                ChannelLayout::default(codec.channels().into())
            } else {
                codec.channel_layout()
            }
        };
        codec.set_channel_layout(in_channel_layout);

        let (tx, rx) = mpsc::channel();
        let in_codec_format = codec.format();
        let in_codec_rate = codec.rate();
        let child = std_thread::spawn(move || {
            resample_frame(
                rx,
                in_codec_format,
                in_channel_layout,
                in_codec_rate,
                sample_array,
            )
        });
        for (s, packet) in format.packets() {
            if s.index() != stream {
                continue;
            }
            match codec.send_packet(&packet) {
                Ok(_) => (),
                Err(Error::Other { errno: EINVAL }) => {
                    return Err(BlissError::DecodingError(String::from(
                        "wrong codec opened.",
                    )))
                }
                Err(Error::Eof) => {
                    warn!("Premature EOF reached while decoding.");
                    drop(tx);
                    song.sample_array = child.join().unwrap()?;
                    return Ok(song);
                }
                Err(e) => warn!("error while decoding {}: {}", path.display(), e),
            };

            loop {
                let mut decoded = ffmpeg::frame::Audio::empty();
                match codec.receive_frame(&mut decoded) {
                    Ok(_) => {
                        tx.send(decoded).map_err(|e| {
                            BlissError::DecodingError(format!(
                                "while sending decoded frame to the resampling thread: {:?}",
                                e
                            ))
                        })?;
                    }
                    Err(_) => break,
                }
            }
        }

        // Flush the stream
        let packet = ffmpeg::codec::packet::Packet::empty();
        match codec.send_packet(&packet) {
            Ok(_) => (),
            Err(Error::Other { errno: EINVAL }) => {
                return Err(BlissError::DecodingError(String::from(
                    "wrong codec opened.",
                )))
            }
            Err(Error::Eof) => {
                warn!("Premature EOF reached while decoding.");
                drop(tx);
                song.sample_array = child.join().unwrap()?;
                return Ok(song);
            }
            Err(e) => warn!("error while decoding {}: {}", path.display(), e),
        };

        loop {
            let mut decoded = ffmpeg::frame::Audio::empty();
            match codec.receive_frame(&mut decoded) {
                Ok(_) => {
                    tx.send(decoded).map_err(|e| {
                        BlissError::DecodingError(format!(
                            "while sending decoded frame to the resampling thread: {:?}",
                            e
                        ))
                    })?;
                }
                Err(_) => break,
            }
        }

        drop(tx);
        song.sample_array = child.join().unwrap()?;
        Ok(song)
    }
}

#[derive(Default, Debug)]
pub(crate) struct InternalSong {
    pub path: PathBuf,
    pub artist: Option<String>,
    pub title: Option<String>,
    pub album: Option<String>,
    pub track_number: Option<String>,
    pub genre: Option<String>,
    pub sample_array: Vec<f32>,
}

fn resample_frame(
    rx: Receiver<Audio>,
    in_codec_format: Sample,
    in_channel_layout: ChannelLayout,
    in_rate: u32,
    mut sample_array: Vec<f32>,
) -> BlissResult<Vec<f32>> {
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
            "while trying to allocate resampling context: {:?}",
            e
        ))
    })?;
    let mut resampled = ffmpeg::frame::Audio::empty();
    for decoded in rx.iter() {
        resampled = ffmpeg::frame::Audio::empty();
        resample_context
            .run(&decoded, &mut resampled)
            .map_err(|e| {
                BlissError::DecodingError(format!("while trying to resample song: {:?}", e))
            })?;
        push_to_sample_array(&resampled, &mut sample_array);
    }
    // TODO when ffmpeg-next will be active again: shouldn't we allocate
    // `resampled` again?
    loop {
        match resample_context.flush(&mut resampled).map_err(|e| {
            BlissError::DecodingError(format!("while trying to resample song: {:?}", e))
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
    use ripemd160::{Digest, Ripemd160};
    use std::path::Path;

    #[test]
    #[cfg(feature = "cue")]
    fn test_cue_analysis() {
        let songs = Song::from_cue(Path::new("data/testcue.cue")).unwrap();
        let expected = vec![
            Ok(Song {
                path: Path::new("data/testcue.wav/CUE_TRACK001").to_path_buf(),
                analysis: Analysis {
                    internal_analysis: [
                        0.38463724,
                        -0.85219246,
                        -0.761946,
                        -0.8904667,
                        -0.63892543,
                        -0.73945934,
                        -0.8004017,
                        -0.8237293,
                        0.33865356,
                        0.32481194,
                        -0.35692245,
                        -0.6355889,
                        -0.29584837,
                        0.06431806,
                        0.21875131,
                        -0.58104205,
                        -0.9466792,
                        -0.94811195,
                        -0.9820919,
                        -0.9596871,
                    ],
                },
                album: Some(String::from("Album for CUE test")),
                artist: Some(String::from("David TMX")),
                title: Some(String::from("Renaissance")),
                genre: Some(String::from("Random")),
                track_number: Some(String::from("01")),
                features_version: FEATURES_VERSION,
                ..Default::default()
            }),
            Ok(Song {
                path: Path::new("data/testcue.wav/CUE_TRACK002").to_path_buf(),
                analysis: Analysis {
                    internal_analysis: [
                        0.18622077,
                        -0.5989029,
                        -0.5554645,
                        -0.6343865,
                        -0.24163479,
                        -0.25766593,
                        -0.40616858,
                        -0.23334873,
                        0.76875293,
                        0.7785741,
                        -0.5075115,
                        -0.5272629,
                        -0.56706166,
                        -0.568486,
                        -0.5639081,
                        -0.5706943,
                        -0.96501005,
                        -0.96501285,
                        -0.9649896,
                        -0.96498996,
                    ],
                },
                features_version: FEATURES_VERSION,
                album: Some(String::from("Album for CUE test")),
                artist: Some(String::from("Polochon_street")),
                title: Some(String::from("Piano")),
                genre: Some(String::from("Random")),
                track_number: Some(String::from("02")),
                ..Default::default()
            }),
            Ok(Song {
                path: Path::new("data/testcue.wav/CUE_TRACK003").to_path_buf(),
                analysis: Analysis {
                    internal_analysis: [
                        0.0024261475,
                        0.9874661,
                        0.97330654,
                        -0.9724426,
                        0.99678576,
                        -0.9961549,
                        -0.9840142,
                        -0.9269961,
                        0.7498772,
                        0.22429907,
                        -0.8355152,
                        -0.9977258,
                        -0.9977849,
                        -0.997785,
                        -0.99778515,
                        -0.997785,
                        -0.99999976,
                        -0.99999976,
                        -0.99999976,
                        -0.99999976,
                    ],
                },
                album: Some(String::from("Album for CUE test")),
                artist: Some(String::from("Polochon_street")),
                title: Some(String::from("Tone")),
                genre: Some(String::from("Random")),
                track_number: Some(String::from("03")),
                features_version: FEATURES_VERSION,
                ..Default::default()
            }),
        ];
        assert_eq!(expected, songs);
    }

    #[test]
    fn test_analysis_too_small() {
        let error = Song::analyze(vec![0.]).unwrap_err();
        assert_eq!(
            error,
            BlissError::AnalysisError(String::from("empty or too short song."))
        );

        let error = Song::analyze(vec![]).unwrap_err();
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

    fn _test_decode(path: &Path, expected_hash: &[u8]) {
        let song = Song::decode(path).unwrap();
        let mut hasher = Ripemd160::new();
        for sample in song.sample_array.iter() {
            hasher.update(sample.to_le_bytes().to_vec());
        }

        assert_eq!(expected_hash, hasher.finalize().as_slice());
    }

    #[test]
    fn test_tags() {
        let song = Song::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        assert_eq!(song.artist, Some(String::from("David TMX")));
        assert_eq!(song.title, Some(String::from("Renaissance")));
        assert_eq!(song.album, Some(String::from("Renaissance")));
        assert_eq!(song.track_number, Some(String::from("02")));
        assert_eq!(song.genre, Some(String::from("Pop")));
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
        let expected_hash = [
            0xc5, 0xf8, 0x23, 0xce, 0x63, 0x2c, 0xf4, 0xa0, 0x72, 0x66, 0xbb, 0x49, 0xad, 0x84,
            0xb6, 0xea, 0x48, 0x48, 0x9c, 0x50,
        ];
        _test_decode(&path, &expected_hash);
    }

    #[test]
    fn test_resample_stereo() {
        let path = Path::new("data/s16_stereo_22_5kHz.flac");
        let expected_hash = [
            0x24, 0xed, 0x45, 0x58, 0x06, 0xbf, 0xfb, 0x05, 0x57, 0x5f, 0xdc, 0x4d, 0xb4, 0x9b,
            0xa5, 0x2b, 0x05, 0x56, 0x10, 0x4f,
        ];
        _test_decode(&path, &expected_hash);
    }

    #[test]
    fn test_decode_mono() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash ripemd160 -
        let expected_hash = [
            0x9d, 0x95, 0xa5, 0xf2, 0xd2, 0x9c, 0x68, 0xe8, 0x8a, 0x70, 0xcd, 0xf3, 0x54, 0x2c,
            0x5b, 0x45, 0x98, 0xb4, 0xf3, 0xb4,
        ];
        _test_decode(&path, &expected_hash);
    }

    #[test]
    fn test_decode_mp3() {
        let path = Path::new("data/s32_stereo_44_1_kHz.mp3");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.mp3 -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash ripemd160 -
        let expected_hash = [
            0x28, 0x25, 0x6b, 0x7b, 0x6e, 0x37, 0x1c, 0xcf, 0xc7, 0x06, 0xdf, 0x62, 0x8c, 0x0e,
            0x91, 0xf7, 0xd6, 0x1f, 0xac, 0x5b,
        ];
        _test_decode(&path, &expected_hash);
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
    fn test_analysis_distance() {
        let mut a = Song::default();
        a.analysis = Analysis::new([
            0.16391512, 0.11326739, 0.96868552, 0.8353934, 0.49867523, 0.76532606, 0.63448005,
            0.82506196, 0.71457147, 0.62395476, 0.69680329, 0.9855766, 0.41369333, 0.13900452,
            0.68001012, 0.11029723, 0.97192943, 0.57727861, 0.07994821, 0.88993185,
        ]);

        let mut b = Song::default();
        b.analysis = Analysis::new([
            0.5075758, 0.36440256, 0.28888011, 0.43032829, 0.62387977, 0.61894916, 0.99676086,
            0.11913155, 0.00640396, 0.15943407, 0.33829514, 0.34947174, 0.82927523, 0.18987604,
            0.54437275, 0.22076826, 0.91232151, 0.29233168, 0.32846024, 0.04522147,
        ]);
        assert_eq!(a.distance(&b), 1.9469079)
    }

    #[test]
    fn test_analysis_distance_indiscernible() {
        let mut a = Song::default();
        a.analysis = Analysis::new([
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ]);
        assert_eq!(a.distance(&a), 0.)
    }

    #[test]
    fn test_decode_errors() {
        assert_eq!(
            Song::decode(Path::new("nonexistent")).unwrap_err(),
            BlissError::DecodingError(String::from(
                "while opening format: ffmpeg::Error(2: No such file or directory)."
            )),
        );
        assert_eq!(
            Song::decode(Path::new("data/picture.png")).unwrap_err(),
            BlissError::DecodingError(String::from("No audio stream found.")),
        );
    }

    #[test]
    fn test_index_analysis() {
        let song = Song::from_path("data/s16_mono_22_5kHz.flac").unwrap();
        assert_eq!(song.analysis[AnalysisIndex::Tempo], 0.3846389);
        assert_eq!(song.analysis[AnalysisIndex::Chroma10], -0.95968974);
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

    fn dummy_distance(_: &Array1<f32>, _: &Array1<f32>) -> f32 {
        0.
    }

    #[test]
    fn test_custom_distance() {
        let mut a = Song::default();
        a.analysis = Analysis::new([
            0.16391512, 0.11326739, 0.96868552, 0.8353934, 0.49867523, 0.76532606, 0.63448005,
            0.82506196, 0.71457147, 0.62395476, 0.69680329, 0.9855766, 0.41369333, 0.13900452,
            0.68001012, 0.11029723, 0.97192943, 0.57727861, 0.07994821, 0.88993185,
        ]);

        let mut b = Song::default();
        b.analysis = Analysis::new([
            0.5075758, 0.36440256, 0.28888011, 0.43032829, 0.62387977, 0.61894916, 0.99676086,
            0.11913155, 0.00640396, 0.15943407, 0.33829514, 0.34947174, 0.82927523, 0.18987604,
            0.54437275, 0.22076826, 0.91232151, 0.29233168, 0.32846024, 0.04522147,
        ]);
        assert_eq!(a.custom_distance(&b, dummy_distance), 0.);
    }

    #[test]
    fn test_closest_from_pool() {
        let song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let first_song_dupe = Song {
            path: Path::new("path-to-dupe").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.9, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.5, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fifth_song = Song {
            path: Path::new("path-to-fifth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let songs = vec![
            song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist = song.closest_from_pool(songs.to_owned());
        assert_eq!(
            playlist,
            vec![
                song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
                third_song.to_owned(),
            ],
        );
        let playlist = song.closest_from_pool_custom(songs, euclidean_distance);
        assert_eq!(
            playlist,
            vec![
                song,
                first_song_dupe,
                second_song,
                fourth_song,
                fifth_song,
                third_song
            ],
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
