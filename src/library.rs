//! Module containing the Library trait, useful to get started to implement
//! a plug-in for an audio player.
#[cfg(doc)]
use crate::distance;
use crate::distance::DistanceMetric;
use crate::{BlissError, BlissResult, Song};
use log::{debug, error, info};
use noisy_float::prelude::*;
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

    /// Return a list of songs that are similar to ``first_song``.
    ///
    /// # Arguments
    ///
    /// * `first_song` - The song the playlist will be built from.
    /// * `playlist_length` - The playlist length. If there are not enough
    /// songs in the library, it will be truncated to the size of the library.
    ///
    /// # Returns
    ///
    /// A vector of `playlist_length` Songs, including `first_song`, that you
    /// most likely want to plug in your audio player by using something like
    /// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
    fn playlist_from_song(
        &self,
        first_song: Song,
        playlist_length: usize,
    ) -> BlissResult<Vec<Song>> {
        let mut songs = self.get_stored_songs()?;
        songs.sort_by_cached_key(|song| n32(first_song.distance(&song)));

        let playlist = songs
            .into_iter()
            .take(playlist_length)
            .collect::<Vec<Song>>();
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
    /// custom distance metric.
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
        let mut songs = self.get_stored_songs()?;
        songs.sort_by_cached_key(|song| n32(first_song.custom_distance(&song, &distance)));

        let playlist = songs
            .into_iter()
            .take(playlist_length)
            .collect::<Vec<Song>>();
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

    /// Analyze and store songs in `paths`, using `store_song` and
    /// `store_error_song` implementations.
    ///
    /// Note: this is mostly useful for updating a song library. For the first
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
                    let song = Song::new(&path);
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
}
