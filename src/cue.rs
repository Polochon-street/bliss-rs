//! CUE-handling module.
//!
//! Using [BlissCue::songs_from_path] is most likely what you want.
//!
//! There are two main structures in this module, [BlissCue], used
//! to extract and analyze songs from a cue file through [BlissCue::songs_from_path],
//! and [CueInfo], which is a struct stored in [Song] to keep track of the CUE information
//! the song was extracted from.

use crate::song::decoder::Decoder as DecoderTrait;
use crate::{Analysis, BlissError, BlissResult, Song, FEATURES_VERSION, SAMPLE_RATE};
use rcue::cue::{Cue, Track};
use rcue::parser::parse_from_file;
use std::path::Path;
use std::time::Duration;
use std::{marker::PhantomData, path::PathBuf};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, Debug, PartialEq, Eq, Clone)]
/// A struct populated when the corresponding [Song] has been extracted from an
/// audio file split with the help of a CUE sheet.
/// It is mostly used to sit in [Song::cue_info], in order to know which file this song
/// was extracted from, and from which CUE file the information was retrieved.
pub struct CueInfo {
    /// The path of the original CUE sheet, e.g. `/path/to/album_name.cue`.
    pub cue_path: PathBuf,
    /// The path of the audio file the song was extracted from, e.g.
    /// `/path/to/album_name.wav`. Used because one CUE sheet can refer to
    /// several audio files.
    pub audio_file_path: PathBuf,
}

/// A struct to handle CUEs with bliss.
/// Use either [analyze_paths](crate::decoder::Decoder::analyze_paths), which takes care of CUE files
/// automatically, or [songs_from_path](BlissCue::songs_from_path) to return a list
/// of [Song]s from CUE files.
pub struct BlissCue<D: ?Sized> {
    cue: Cue,
    cue_path: PathBuf,
    decoder: PhantomData<D>,
}

#[allow(missing_docs)]
#[derive(Default, Debug, PartialEq, Clone)]
struct BlissCueFile {
    sample_array: Vec<f32>,
    album: Option<String>,
    artist: Option<String>,
    genre: Option<String>,
    tracks: Vec<Track>,
    cue_path: PathBuf,
    audio_file_path: PathBuf,
}

impl<D: ?Sized + DecoderTrait> BlissCue<D> {
    /// Analyze songs from a CUE file, extracting individual [Song] objects
    /// for each individual song.
    ///
    /// Each returned [Song] has a populated [cue_info](Song::cue_info) object, that can be
    /// be used to retrieve which CUE sheet was used to extract it, as well
    /// as the corresponding audio file.
    pub fn songs_from_path<P: AsRef<Path>>(path: P) -> BlissResult<Vec<BlissResult<Song>>> {
        let cue: BlissCue<D> = BlissCue::from_path(&path)?;
        let cue_files = cue.files();
        let mut songs = Vec::new();
        for cue_file in cue_files.into_iter() {
            match cue_file {
                Ok(f) => {
                    if !f.sample_array.is_empty() {
                        songs.extend_from_slice(&f.get_songs());
                    } else {
                        songs.push(Err(BlissError::DecodingError(
                            "empty audio file associated to CUE sheet".into(),
                        )));
                    }
                }
                Err(e) => songs.push(Err(e)),
            }
        }
        Ok(songs)
    }

    // Extract a BlissCue from a given path.
    fn from_path<P: AsRef<Path>>(path: P) -> BlissResult<Self> {
        let cue = parse_from_file(&path.as_ref().to_string_lossy(), false).map_err(|e| {
            BlissError::DecodingError(format!(
                "when opening CUE file '{:?}': {:?}",
                path.as_ref(),
                e
            ))
        })?;
        Ok(Self {
            cue,
            cue_path: path.as_ref().to_owned(),
            decoder: PhantomData,
        })
    }

    // List all BlissCueFile from a BlissCue.
    fn files(&self) -> Vec<BlissResult<BlissCueFile>> {
        let mut cue_files = Vec::new();
        for cue_file in self.cue.files.iter() {
            let audio_file_path = match &self.cue_path.parent() {
                Some(parent) => parent.join(Path::new(&cue_file.file)),
                None => PathBuf::from(cue_file.file.to_owned()),
            };
            let genre = self
                .cue
                .comments
                .iter()
                .find(|(c, _)| c == "GENRE")
                .map(|(_, v)| v.to_owned());
            let raw_song = D::decode(Path::new(&audio_file_path));
            if let Ok(song) = raw_song {
                let bliss_cue_file = BlissCueFile {
                    sample_array: song.sample_array,
                    genre,
                    artist: self.cue.performer.to_owned(),
                    album: self.cue.title.to_owned(),
                    tracks: cue_file.tracks.to_owned(),
                    audio_file_path,
                    cue_path: self.cue_path.to_owned(),
                };
                cue_files.push(Ok(bliss_cue_file))
            } else {
                cue_files.push(Err(raw_song.unwrap_err()));
            }
        }
        cue_files
    }
}

impl BlissCueFile {
    fn create_song(
        &self,
        analysis: BlissResult<Analysis>,
        current_track: &Track,
        duration: Duration,
        index: usize,
    ) -> BlissResult<Song> {
        if let Ok(a) = analysis {
            let song = Song {
                path: PathBuf::from(format!(
                    "{}/CUE_TRACK{:03}",
                    self.cue_path.to_string_lossy(),
                    index,
                )),
                album: self.album.to_owned(),
                artist: current_track.performer.to_owned(),
                album_artist: self.artist.to_owned(),
                analysis: a,
                duration,
                genre: self.genre.to_owned(),
                title: current_track.title.to_owned(),
                track_number: current_track.no.parse::<i32>().ok(),
                features_version: FEATURES_VERSION,
                cue_info: Some(CueInfo {
                    cue_path: self.cue_path.to_owned(),
                    audio_file_path: self.audio_file_path.to_owned(),
                }),
            };
            Ok(song)
        } else {
            Err(analysis.unwrap_err())
        }
    }

    // Get all songs from a BlissCueFile, using Song::analyze, each song being
    // located using the sample_array and the timestamp delimiter.
    fn get_songs(&self) -> Vec<BlissResult<Song>> {
        let mut songs = Vec::new();
        for (index, tuple) in (self.tracks[..]).windows(2).enumerate() {
            let (current_track, next_track) = (tuple[0].to_owned(), tuple[1].to_owned());
            if let Some((_, start_current)) = current_track.indices.first() {
                if let Some((_, end_current)) = next_track.indices.first() {
                    let start_current = (start_current.as_secs_f32() * SAMPLE_RATE as f32) as usize;
                    let end_current = (end_current.as_secs_f32() * SAMPLE_RATE as f32) as usize;
                    let duration = Duration::from_secs_f32(
                        (end_current - start_current) as f32 / SAMPLE_RATE as f32,
                    );
                    let analysis = Song::analyze(&self.sample_array[start_current..end_current]);

                    let song = self.create_song(analysis, &current_track, duration, index + 1);
                    songs.push(song);
                }
            }
        }
        // Take care of the last track, since the windows iterator doesn't.
        if let Some(last_track) = self.tracks.last() {
            if let Some((_, start_current)) = last_track.indices.first() {
                let start_current = (start_current.as_secs_f32() * SAMPLE_RATE as f32) as usize;
                let duration = Duration::from_secs_f32(
                    (self.sample_array.len() - start_current) as f32 / SAMPLE_RATE as f32,
                );
                let analysis = Song::analyze(&self.sample_array[start_current..]);
                let song = self.create_song(analysis, last_track, duration, self.tracks.len());
                songs.push(song);
            }
        }
        songs
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ffmpeg")]
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::decoder::ffmpeg::FFmpeg;
    #[cfg(feature = "ffmpeg")]
    use pretty_assertions::assert_eq;

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_empty_cue() {
        let songs = BlissCue::<FFmpeg>::songs_from_path("data/empty.cue").unwrap();
        let error = songs[0].to_owned().unwrap_err();
        assert_eq!(
            error,
            BlissError::DecodingError("empty audio file associated to CUE sheet".to_string())
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_cue_analysis() {
        let songs = BlissCue::<FFmpeg>::songs_from_path("data/testcue.cue").unwrap();
        let expected = vec![
            Ok(Song {
                path: Path::new("data/testcue.cue/CUE_TRACK001").to_path_buf(),
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
                track_number: Some(1),
                features_version: FEATURES_VERSION,
                album_artist: Some(String::from("Polochon_street")),
                duration: Duration::from_secs_f32(11.066666603),
                cue_info: Some(CueInfo {
                    cue_path: PathBuf::from("data/testcue.cue"),
                    audio_file_path: PathBuf::from("data/testcue.flac"),
                }),
                ..Default::default()
            }),
            Ok(Song {
                path: Path::new("data/testcue.cue/CUE_TRACK002").to_path_buf(),
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
                track_number: Some(2),
                album_artist: Some(String::from("Polochon_street")),
                duration: Duration::from_secs_f64(5.853333473),
                cue_info: Some(CueInfo {
                    cue_path: PathBuf::from("data/testcue.cue"),
                    audio_file_path: PathBuf::from("data/testcue.flac"),
                }),
                ..Default::default()
            }),
            Ok(Song {
                path: Path::new("data/testcue.cue/CUE_TRACK003").to_path_buf(),
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
                track_number: Some(3),
                features_version: FEATURES_VERSION,
                album_artist: Some(String::from("Polochon_street")),
                duration: Duration::from_secs_f32(5.586666584),
                cue_info: Some(CueInfo {
                    cue_path: PathBuf::from("data/testcue.cue"),
                    audio_file_path: PathBuf::from("data/testcue.flac"),
                }),
                ..Default::default()
            }),
            Err(BlissError::DecodingError(String::from(
                "while opening format for file 'data/not-existing.wav': \
                ffmpeg::Error(2: No such file or directory).",
            ))),
        ];
        assert_eq!(expected, songs);
    }
}
