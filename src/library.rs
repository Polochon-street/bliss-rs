//! Module containing the Library trait, useful to get started to implement
//! a plug-in for an audio player.
//!
//! Looking at the [reference implementation for
//! MPD](https://github.com/Polochon-street/blissify-rs) could also be useful.
#[cfg(doc)]
use crate::distance;
use crate::distance::{closest_to_first_song, euclidean_distance, DistanceMetric};
use crate::{BlissError, BlissResult, Song};
use log::{debug, error, info};
use ndarray::{Array, Array2, Axis};
use noisy_float::prelude::n32;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

/// Library trait to make creating plug-ins for existing audio players easier.
pub trait Library {
    /// Return the absolute path of all the songs in an
    /// audio player's music library.
    fn get_songs_paths(&self) -> BlissResult<Vec<String>>;
    /// Store an analyzed Song object in some (cold) storage, e.g.
    /// a database, a file...
    fn store_song(&mut self, song: &Song) -> BlissResult<()>;
    /// Log and / or store that an error happened while trying to decode and
    /// analyze a song.
    fn store_error_song(&mut self, song_path: String, error: BlissError) -> BlissResult<()>;
    /// Retrieve a list of all the stored Songs.
    ///
    /// This should work only after having run `analyze_library` at least
    /// once.
    fn get_stored_songs(&self) -> BlissResult<Vec<Song>>;

    /// Return a list of `number_albums` albums that are similar
    /// to `album`, discarding songs that don't belong to an album.
    ///
    /// # Arguments
    ///
    /// * `album` - The album the playlist will be built from.
    /// * `number_albums` - The number of albums to queue.
    ///
    /// # Returns
    ///
    /// A vector of songs, including `first_song`, that you
    /// most likely want to plug in your audio player by using something like
    /// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
    fn playlist_from_songs_album(
        &self,
        first_album: &str,
        playlist_length: usize,
    ) -> BlissResult<Vec<Song>> {
        let songs = self.get_stored_songs()?;
        let mut albums_analysis: HashMap<&str, Array2<f32>> = HashMap::new();
        let mut albums = Vec::new();

        for song in &songs {
            if let Some(album) = &song.album {
                if let Some(analysis) = albums_analysis.get_mut(album as &str) {
                    analysis
                        .push_row(song.analysis.as_arr1().view())
                        .map_err(|e| {
                            BlissError::ProviderError(format!("while computing distances: {}", e))
                        })?;
                } else {
                    let mut array = Array::zeros((1, song.analysis.as_arr1().len()));
                    array.assign(&song.analysis.as_arr1());
                    albums_analysis.insert(album, array);
                }
            }
        }
        let mut first_analysis = None;
        for (album, analysis) in albums_analysis.iter() {
            let mean_analysis = analysis
                .mean_axis(Axis(0))
                .ok_or_else(|| BlissError::ProviderError(String::from("Mean of empty slice")))?;
            let album = album.to_owned();
            albums.push((album, mean_analysis.to_owned()));
            if album == first_album {
                first_analysis = Some(mean_analysis);
            }
        }

        if first_analysis.is_none() {
            return Err(BlissError::ProviderError(format!(
                "Could not find album \"{}\".",
                first_album
            )));
        }
        albums.sort_by_key(|(_, analysis)| {
            n32(euclidean_distance(
                first_analysis.as_ref().unwrap(),
                analysis,
            ))
        });
        let albums = albums.get(..playlist_length).unwrap_or(&albums);
        let mut playlist = Vec::new();
        for (album, _) in albums {
            let mut al = songs
                .iter()
                .filter(|s| s.album.is_some() && s.album.as_ref().unwrap() == &album.to_string())
                .map(|s| s.to_owned())
                .collect::<Vec<Song>>();
            al.sort_by(|s1, s2| {
                let track_number1 = s1
                    .track_number
                    .to_owned()
                    .unwrap_or_else(|| String::from(""));
                let track_number2 = s2
                    .track_number
                    .to_owned()
                    .unwrap_or_else(|| String::from(""));
                if let Ok(x) = track_number1.parse::<i32>() {
                    if let Ok(y) = track_number2.parse::<i32>() {
                        return x.cmp(&y);
                    }
                }
                s1.track_number.cmp(&s2.track_number)
            });
            playlist.extend_from_slice(&al);
        }
        Ok(playlist)
    }

    /// Return a list of `playlist_length` songs that are similar
    /// to ``first_song``, deduplicating identical songs.
    ///
    /// # Arguments
    ///
    /// * `first_song` - The song the playlist will be built from.
    /// * `playlist_length` - The playlist length. If there are not enough
    /// songs in the library, it will be truncated to the size of the library.
    ///
    /// # Returns
    ///
    /// A vector of `playlist_length` songs, including `first_song`, that you
    /// most likely want to plug in your audio player by using something like
    /// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
    // TODO return an iterator and not a Vec
    fn playlist_from_song(
        &self,
        first_song: Song,
        playlist_length: usize,
    ) -> BlissResult<Vec<Song>> {
        let playlist = self.playlist_from_song_custom(
            first_song,
            playlist_length,
            euclidean_distance,
            closest_to_first_song,
        )?;

        debug!(
            "Playlist created: {}",
            playlist
                .iter()
                .map(|s| format!("{:?}", &s))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        Ok(playlist)
    }

    /// Return a list of songs that are similar to ``first_song``, using a
    /// custom distance metric and deduplicating indentical songs.
    ///
    /// # Arguments
    ///
    /// * `first_song` - The song the playlist will be built from.
    /// * `playlist_length` - The playlist length. If there are not enough
    /// songs in the library, it will be truncated to the size of the library.
    /// * `distance` - a user-supplied valid distance metric, either taken
    /// from the [distance](distance) module, or made from scratch.
    ///
    /// # Returns
    ///
    /// A vector of `playlist_length` Songs, including `first_song`, that you
    /// most likely want to plug in your audio player by using something like
    /// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
    ///
    /// # Custom distance example
    ///
    /// ```
    /// use ndarray::Array1;
    ///
    /// fn manhattan_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    ///     (a - b).mapv(|x| x.abs()).sum()
    /// }
    /// ```
    fn playlist_from_song_custom_distance(
        &self,
        first_song: Song,
        playlist_length: usize,
        distance: impl DistanceMetric,
    ) -> BlissResult<Vec<Song>> {
        let playlist = self.playlist_from_song_custom(
            first_song,
            playlist_length,
            distance,
            closest_to_first_song,
        )?;

        debug!(
            "Playlist created: {}",
            playlist
                .iter()
                .map(|s| format!("{:?}", &s))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        Ok(playlist)
    }

    /// Return a playlist of songs, starting with `first_song`, sorted using
    /// the custom `sort` function, and the custom `distance` metric.
    ///
    /// # Arguments
    ///
    /// * `first_song` - The song the playlist will be built from.
    /// * `playlist_length` - The playlist length. If there are not enough
    ///    songs in the library, it will be truncated to the size of the library.
    /// * `distance` - a user-supplied valid distance metric, either taken
    ///    from the [distance](distance) module, or made from scratch.
    /// * `sort` - a user-supplied sorting function that uses the `distance`
    ///    metric, either taken from the [distance module](distance), or made
    ///    from scratch.
    ///
    /// # Returns
    ///
    /// A vector of `playlist_length` Songs, including `first_song`, that you
    /// most likely want to plug in your audio player by using something like
    /// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
    fn playlist_from_song_custom<F, G>(
        &self,
        first_song: Song,
        playlist_length: usize,
        distance: G,
        mut sort: F,
    ) -> BlissResult<Vec<Song>>
    where
        F: FnMut(&Song, &mut Vec<Song>, G),
        G: DistanceMetric,
    {
        let mut songs = self.get_stored_songs()?;
        sort(&first_song, &mut songs, distance);
        Ok(songs
            .into_iter()
            .take(playlist_length)
            .collect::<Vec<Song>>())
    }

    /// Analyze and store songs in `paths`, using `store_song` and
    /// `store_error_song` implementations.
    ///
    /// note: this is mostly useful for updating a song library. for the first
    /// run, you probably want to use `analyze_library`.
    fn analyze_paths(&mut self, paths: Vec<String>) -> BlissResult<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let num_cpus = num_cpus::get();

        #[allow(clippy::type_complexity)]
        let (tx, rx): (
            Sender<(String, BlissResult<Song>)>,
            Receiver<(String, BlissResult<Song>)>,
        ) = mpsc::channel();
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
                drop(tx_thread);
            });
            handles.push(child);
        }
        drop(tx);

        for (path, song) in rx.iter() {
            // A storage fail should just warn the user, but not abort the whole process
            match song {
                Ok(song) => {
                    self.store_song(&song).unwrap_or_else(|e| {
                        error!("Error while storing song '{}': {}", song.path.display(), e)
                    });
                    info!(
                        "Analyzed and stored song '{}' successfully.",
                        song.path.display()
                    )
                }
                Err(e) => {
                    self.store_error_song(path.to_string(), e.to_owned())
                        .unwrap_or_else(|e| {
                            error!("Error while storing errored song '{}': {}", path, e)
                        });
                    error!(
                        "Analysis of song '{}': {} failed. Error has been stored.",
                        path, e
                    )
                }
            }
        }

        for child in handles {
            child
                .join()
                .map_err(|_| BlissError::AnalysisError("in analysis".to_string()))?;
        }
        Ok(())
    }

    /// Analyzes a song library, using `get_songs_paths`, `store_song` and
    /// `store_error_song` implementations.
    fn analyze_library(&mut self) -> BlissResult<()> {
        let paths = self
            .get_songs_paths()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        self.analyze_paths(paths)?;
        Ok(())
    }

    /// Analyze an entire library using `get_songs_paths`, but instead of
    /// storing songs using [store_song](Library::store_song)
    /// and [store_error_song](Library::store_error_song).
    ///
    /// Returns an iterable [Receiver], whose items are a tuple made of
    /// the song path (to display to the user in case the analysis failed),
    /// and a Result<Song>.
    fn analyze_library_streaming(&mut self) -> BlissResult<Receiver<(String, BlissResult<Song>)>> {
        let paths = self
            .get_songs_paths()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        analyze_paths_streaming(paths)
    }
}

/// Analyze songs in `paths`, and return the analyzed [Song] objects through a
/// [Receiver].
///
/// Returns an iterable [Receiver], whose items are a tuple made of
/// the song path (to display to the user in case the analysis failed),
/// and a Result<Song>.
///
/// Note: this is mostly useful for updating a song library, while displaying
/// status to the user (since you have access to each song object). For the
/// first run, you probably want to use `analyze_library`.
///
/// * Example:
/// ```no_run
/// use bliss_audio::{library::analyze_paths_streaming, BlissResult};
///
/// fn main() -> BlissResult<()> {
///     let paths = vec![String::from("/path/to/song1"), String::from("/path/to/song2")];
///     let rx = analyze_paths_streaming(paths)?;
///     for (path, result) in rx.iter() {
///         match result {
///             Ok(song) => println!("Do something with analyzed song {} with title {:?}", song.path.display(), song.title),
///             Err(e) => println!("Song at {} could not be analyzed. Failed with: {}", path, e),
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn analyze_paths_streaming(
    paths: Vec<String>,
) -> BlissResult<Receiver<(String, BlissResult<Song>)>> {
    let num_cpus = num_cpus::get();

    #[allow(clippy::type_complexity)]
    let (tx, rx): (
        Sender<(String, BlissResult<Song>)>,
        Receiver<(String, BlissResult<Song>)>,
    ) = mpsc::channel();
    if paths.is_empty() {
        return Ok(rx);
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

    Ok(rx)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::song::Analysis;
    use ndarray::Array1;
    use std::path::Path;

    #[derive(Default)]
    struct TestLibrary {
        internal_storage: Vec<Song>,
        failed_files: Vec<(String, String)>,
    }

    impl Library for TestLibrary {
        fn get_songs_paths(&self) -> BlissResult<Vec<String>> {
            Ok(vec![
                String::from("./data/white_noise.flac"),
                String::from("./data/s16_mono_22_5kHz.flac"),
                String::from("not-existing.foo"),
                String::from("definitely-not-existing.foo"),
            ])
        }

        fn store_song(&mut self, song: &Song) -> BlissResult<()> {
            self.internal_storage.push(song.to_owned());
            Ok(())
        }

        fn store_error_song(&mut self, song_path: String, error: BlissError) -> BlissResult<()> {
            self.failed_files.push((song_path, error.to_string()));
            Ok(())
        }

        fn get_stored_songs(&self) -> BlissResult<Vec<Song>> {
            Ok(self.internal_storage.to_owned())
        }
    }

    #[derive(Default)]
    struct FailingLibrary;

    impl Library for FailingLibrary {
        fn get_songs_paths(&self) -> BlissResult<Vec<String>> {
            Err(BlissError::ProviderError(String::from(
                "Could not get songs path",
            )))
        }

        fn store_song(&mut self, _: &Song) -> BlissResult<()> {
            Ok(())
        }

        fn get_stored_songs(&self) -> BlissResult<Vec<Song>> {
            Err(BlissError::ProviderError(String::from(
                "Could not get stored songs",
            )))
        }

        fn store_error_song(&mut self, _: String, _: BlissError) -> BlissResult<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct FailingStorage;

    impl Library for FailingStorage {
        fn get_songs_paths(&self) -> BlissResult<Vec<String>> {
            Ok(vec![
                String::from("./data/white_noise.flac"),
                String::from("./data/s16_mono_22_5kHz.flac"),
                String::from("not-existing.foo"),
                String::from("definitely-not-existing.foo"),
            ])
        }

        fn store_song(&mut self, song: &Song) -> BlissResult<()> {
            Err(BlissError::ProviderError(format!(
                "Could not store song {}",
                song.path.display()
            )))
        }

        fn get_stored_songs(&self) -> BlissResult<Vec<Song>> {
            Ok(vec![])
        }

        fn store_error_song(&mut self, song_path: String, error: BlissError) -> BlissResult<()> {
            Err(BlissError::ProviderError(format!(
                "Could not store errored song: {}, with error: {}",
                song_path, error
            )))
        }
    }

    #[test]
    fn test_analyze_library_fail() {
        let mut test_library = FailingLibrary {};
        assert_eq!(
            test_library.analyze_library(),
            Err(BlissError::ProviderError(String::from(
                "error happened with the music library provider - Could not get songs path"
            ))),
        );
    }

    #[test]
    fn test_playlist_from_song_fail() {
        let test_library = FailingLibrary {};
        let song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            ..Default::default()
        };

        assert_eq!(
            test_library.playlist_from_song(song, 10),
            Err(BlissError::ProviderError(String::from(
                "Could not get stored songs"
            ))),
        );
    }

    #[test]
    fn test_analyze_library_fail_storage() {
        let mut test_library = FailingStorage {};

        // A storage fail should just warn the user, but not abort the whole process
        assert!(test_library.analyze_library().is_ok())
    }

    #[test]
    fn test_analyze_library_streaming() {
        let mut test_library = TestLibrary {
            internal_storage: vec![],
            failed_files: vec![],
        };
        let rx = test_library.analyze_library_streaming().unwrap();

        let mut result = rx.iter().collect::<Vec<(String, BlissResult<Song>)>>();
        result.sort_by_key(|k| k.0.to_owned());
        let expected = result
            .iter()
            .map(|x| match &x.1 {
                Ok(s) => (true, s.path.to_string_lossy().to_string()),
                Err(_) => (false, x.0.to_owned()),
            })
            .collect::<Vec<(bool, String)>>();
        assert_eq!(
            vec![
                (true, String::from("./data/s16_mono_22_5kHz.flac")),
                (true, String::from("./data/white_noise.flac")),
                (false, String::from("definitely-not-existing.foo")),
                (false, String::from("not-existing.foo")),
            ],
            expected,
        );
    }

    #[test]
    fn test_analyze_library() {
        let mut test_library = TestLibrary {
            internal_storage: vec![],
            failed_files: vec![],
        };
        test_library.analyze_library().unwrap();

        let mut failed_files = test_library
            .failed_files
            .iter()
            .map(|x| x.0.to_owned())
            .collect::<Vec<String>>();
        failed_files.sort();

        assert_eq!(
            failed_files,
            vec![
                String::from("definitely-not-existing.foo"),
                String::from("not-existing.foo"),
            ],
        );

        let mut songs = test_library
            .internal_storage
            .iter()
            .map(|x| x.path.to_str().unwrap().to_string())
            .collect::<Vec<String>>();
        songs.sort();

        assert_eq!(
            songs,
            vec![
                String::from("./data/s16_mono_22_5kHz.flac"),
                String::from("./data/white_noise.flac"),
            ],
        );
    }

    #[test]
    fn test_playlist_from_album() {
        let mut test_library = TestLibrary::default();
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            album: Some(String::from("Album")),
            track_number: Some(String::from("01")),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            album: Some(String::from("Another Album")),
            track_number: Some(String::from("10")),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            album: Some(String::from("Album")),
            track_number: Some(String::from("02")),
            ..Default::default()
        };

        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            album: Some(String::from("Another Album")),
            track_number: Some(String::from("01")),
            ..Default::default()
        };
        let fifth_song = Song {
            path: Path::new("path-to-fifth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            album: None,
            ..Default::default()
        };

        test_library.internal_storage = vec![
            first_song.to_owned(),
            fourth_song.to_owned(),
            third_song.to_owned(),
            second_song.to_owned(),
            fifth_song.to_owned(),
        ];
        assert_eq!(
            vec![first_song, third_song, fourth_song, second_song],
            test_library.playlist_from_songs_album("Album", 3).unwrap()
        );
    }

    #[test]
    fn test_playlist_from_song() {
        let mut test_library = TestLibrary::default();
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            ..Default::default()
        };

        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            ..Default::default()
        };

        test_library.internal_storage = vec![
            first_song.to_owned(),
            fourth_song.to_owned(),
            third_song.to_owned(),
            second_song.to_owned(),
        ];
        assert_eq!(
            vec![first_song.to_owned(), second_song, third_song],
            test_library.playlist_from_song(first_song, 3).unwrap()
        );
    }

    #[test]
    fn test_playlist_from_song_too_little_songs() {
        let mut test_library = TestLibrary::default();
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            ..Default::default()
        };

        test_library.internal_storage = vec![
            first_song.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
        ];
        assert_eq!(
            vec![first_song.to_owned(), second_song, third_song],
            test_library.playlist_from_song(first_song, 200).unwrap()
        );
    }

    #[test]
    fn test_analyze_empty_path() {
        let mut test_library = TestLibrary::default();
        assert!(test_library.analyze_paths(vec![]).is_ok());
    }

    fn custom_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        if a == b {
            return 0.;
        }
        1. / (a.first().unwrap() - b.first().unwrap()).abs()
    }

    #[test]
    fn test_playlist_from_song_custom_distance() {
        let mut test_library = TestLibrary::default();
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            ..Default::default()
        };

        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            ..Default::default()
        };

        test_library.internal_storage = vec![
            first_song.to_owned(),
            fourth_song.to_owned(),
            third_song.to_owned(),
            second_song.to_owned(),
        ];
        assert_eq!(
            vec![first_song.to_owned(), fourth_song, third_song],
            test_library
                .playlist_from_song_custom_distance(first_song, 3, custom_distance)
                .unwrap()
        );
    }

    fn custom_sort(_: &Song, songs: &mut Vec<Song>, _: impl DistanceMetric) {
        songs.sort_by_key(|song| song.path.to_owned());
    }

    #[test]
    fn test_playlist_from_song_custom() {
        let mut test_library = TestLibrary::default();
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            ..Default::default()
        };

        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            ..Default::default()
        };

        test_library.internal_storage = vec![
            first_song.to_owned(),
            fourth_song.to_owned(),
            third_song.to_owned(),
            second_song.to_owned(),
        ];
        assert_eq!(
            vec![first_song.to_owned(), fourth_song, second_song],
            test_library
                .playlist_from_song_custom(first_song, 3, custom_distance, custom_sort)
                .unwrap()
        );
    }
}
