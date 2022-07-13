//! Module containing utilities to manage a SQLite library of [Song]s.
use crate::analyze_paths;
use crate::playlist::euclidean_distance;
use anyhow::{bail, Context, Result};
#[cfg(not(test))]
use dirs::data_local_dir;
use indicatif::{ProgressBar, ProgressStyle};
use noisy_float::prelude::*;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::Row;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use crate::Song;
use crate::FEATURES_VERSION;
use crate::{Analysis, BlissError, NUMBER_FEATURES};
use rusqlite::Error as RusqliteError;
use std::convert::TryInto;
use std::time::Duration;

/// Configuration trait, used for instance to customize
/// the format in which the configuration file should be written.
pub trait AppConfigTrait: Serialize + Sized + DeserializeOwned {
    // Implementers have to provide these.
    /// This trait should return the [BaseConfig] from the parent,
    /// user-created `Config`.
    fn base_config(&self) -> &BaseConfig;

    // Default implementation to output the config as a JSON file.
    /// Convert the current config to a [String], to be written to
    /// a file.
    ///
    /// The default writes a JSON file, but any format can be used,
    /// using for example the various Serde libraries (`serde_yaml`, etc) -
    /// just overwrite this method.
    fn serialize_config(&self) -> Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// Default implementation to load a config from a JSON file.
    /// Reads from a string.
    ///
    /// If you change the serialization format to use something else
    /// than JSON, you need to also overwrite that function with the
    /// format you chose.
    fn deserialize_config(data: &str) -> Result<Self> {
        Ok(serde_json::from_str(data)?)
    }

    /// Load a config from the specified path, using `deserialize_config`.
    ///
    /// This method can be overriden in the very unlikely case
    /// the user wants to do something Serde cannot.
    fn from_path(path: &str) -> Result<Self> {
        let data = fs::read_to_string(path)?;
        Self::deserialize_config(&data)
    }

    // This default impl is what requires the `Serialize` supertrait
    /// Write the configuration to a file using
    /// [AppConfigTrait::serialize_config].
    ///
    /// This method can be overriden
    /// to not use [AppConfigTrait::serialize_config], in the very unlikely
    /// case the user wants to do something Serde cannot.
    fn write(&self) -> Result<()> {
        let serialized = self.serialize_config()?;
        fs::write(&self.base_config().config_path, serialized)?;
        Ok(())
    }
}

/// Actual configuration trait that will be used.
pub trait ConfigTrait: AppConfigTrait {
    /// Do some specific configuration things.
    fn do_config_things(&self) {
        let config = self.base_config();
        config.do_config_things()
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
/// The minimum configuration an application needs to work with
/// a [Library].
pub struct BaseConfig {
    config_path: PathBuf,
    database_path: PathBuf,
}

impl BaseConfig {
    pub(crate) fn get_default_data_folder() -> Result<PathBuf> {
        match env::var("XDG_DATA_HOME") {
            Ok(path) => Ok(Path::new(&path).join("bliss-rs")),
            Err(_) => {
                Ok(
                    data_local_dir()
                    .with_context(|| "No suitable path found to store bliss' song database. Consider specifying such a path.")?
                    .join("bliss-rs")
                )
            },
        }
    }

    /// Create a new, basic config. Upon calls of `Config.write()`, it will be
    /// written to `config_path`.
    //
    /// Any path omitted will instead default to a "clever" path using
    /// data directory inference.
    pub fn new(config_path: Option<PathBuf>, database_path: Option<PathBuf>) -> Result<Self> {
        let config_path = {
            // User provided a path; let the future file creation determine
            // whether it points to something valid or not
            if let Some(path) = config_path {
                path
            } else {
                Self::get_default_data_folder()?.join(Path::new("config.json"))
            }
        };

        let database_path = {
            if let Some(path) = database_path {
                path
            } else {
                Self::get_default_data_folder()?.join(Path::new("songs.db"))
            }
        };
        Ok(Self {
            config_path,
            database_path,
        })
    }

    fn do_config_things(&self) {}
}

impl<App: AppConfigTrait> ConfigTrait for App {}
impl AppConfigTrait for BaseConfig {
    fn base_config(&self) -> &BaseConfig {
        self
    }
}

/// A struct used to hold a collection of [Song]s, with convenience
/// methods to add, remove and update songs.
///
/// Provide it either the `BaseConfig`, or a `Config` extending
/// `BaseConfig`.
/// TODO code example
pub struct Library<Config> {
    /// The configuration struct, containing both information
    /// from `BaseConfig` as well as user-defined values.
    pub config: Config,
    /// SQL connection to the database.
    pub sqlite_conn: Arc<Mutex<Connection>>,
}

/// Struct holding both a Bliss song, as well as any extra info
/// that a user would want to store in the database related to that
/// song.
///
/// The only constraint is that `extra_info` must be serializable, so,
/// something like
/// ```no_compile
/// #[derive(Serialize)]
/// struct ExtraInfo {
///     ignore: bool,
///     unique_id: i64,
/// }
/// let extra_info = ExtraInfo { ignore: true, unique_id = 123 };
/// let song = LibrarySong { bliss_song: song, extra_info };
/// ```
/// is totally possible.
#[derive(Debug, PartialEq, Clone)]
pub struct LibrarySong<T: Serialize + DeserializeOwned> {
    /// Actual bliss song, containing the song's metadata, as well
    /// as the bliss analysis.
    pub bliss_song: Song,
    /// User-controlled information regarding that specific song.
    pub extra_info: T,
}

// TODO simple playlist
// TODO add logging statement
// TODO replace String by pathbufs / the ref thing
// TODO concrete examples
// TODO example LibrarySong without any extra_info
// TODO maybe return number of elements updated / deleted / whatev in analysis
//      functions?
// TODO manage bliss feature version
impl<Config: ConfigTrait> Library<Config> {
    /// Create a new [Library] object from the given [Config] struct,
    /// writing the configuration to the file given in
    /// `config.config_path`.
    ///
    /// This function should only be called once, when a user wishes to
    /// create a completely new "library".
    /// Otherwise, load an existing library file using [Library::from_config].
    pub fn new(config: Config) -> Result<Self> {
        let sqlite_conn = Connection::open(&config.base_config().database_path)?;
        sqlite_conn.execute(
            "
            create table if not exists song (
                id integer primary key,
                path text not null unique,
                duration float,
                album_artist text,
                artist text,
                title text,
                album text,
                track_number text,
                genre text,
                stamp timestamp default current_timestamp,
                version integer,
                analyzed boolean default false,
                extra_info json,
                error text
            );
            ",
            [],
        )?;
        sqlite_conn.execute("pragma foreign_keys = on;", [])?;
        sqlite_conn.execute(
            "
            create table if not exists feature (
                id integer primary key,
                song_id integer not null,
                feature real not null,
                feature_index integer not null,
                unique(song_id, feature_index),
                foreign key(song_id) references song(id) on delete cascade
            )
            ",
            [],
        )?;
        config.write()?;
        Ok(Library {
            config,
            sqlite_conn: Arc::new(Mutex::new(sqlite_conn)),
        })
    }

    /// Load a library from a configuration path.
    ///
    /// If no configuration path is provided, the path is
    /// set using default data folder path.
    pub fn from_config_path(config_path: Option<PathBuf>) -> Result<Self> {
        let config_path: Result<PathBuf> =
            config_path.map_or_else(|| Ok(BaseConfig::new(None, None)?.config_path), Ok);
        let config_path = config_path?;
        let data = fs::read_to_string(config_path)?;
        let config = Config::deserialize_config(&data)?;
        let sqlite_conn = Connection::open(&config.base_config().database_path)?;
        Ok(Library {
            config,
            sqlite_conn: Arc::new(Mutex::new(sqlite_conn)),
        })
    }

    /// Create a new [Library] object from a minimal configuration setup,
    /// writing it to `config_path`.
    pub fn new_from_base(
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
    ) -> Result<Self>
    where
        BaseConfig: Into<Config>,
    {
        let base = BaseConfig::new(config_path, database_path)?;
        let config = base.into();
        Self::new(config)
    }

    /// Build a playlist of `playlist_length` items from an already analyzed
    /// song in the library at `song_path`.
    ///
    /// It uses a simple euclidean distance between songs, and deduplicates songs
    /// that are too close.
    pub fn playlist_from<T: Serialize + DeserializeOwned>(
        &self,
        song_path: &str,
        playlist_length: usize,
    ) -> Result<Vec<LibrarySong<T>>> {
        let first_song: LibrarySong<T> = self.song_from_path(song_path)?;
        let mut songs = self.songs_from_library()?;
        songs.sort_by_cached_key(|song| n32(first_song.bliss_song.distance(&song.bliss_song)));
        songs.truncate(playlist_length);
        songs.dedup_by(|s1, s2| {
            n32(s1
                .bliss_song
                .custom_distance(&s2.bliss_song, &euclidean_distance))
                < 0.05
                || (s1.bliss_song.title.is_some()
                    && s2.bliss_song.title.is_some()
                    && s1.bliss_song.artist.is_some()
                    && s2.bliss_song.artist.is_some()
                    && s1.bliss_song.title == s2.bliss_song.title
                    && s1.bliss_song.artist == s2.bliss_song.artist)
        });
        Ok(songs)
    }

    /// Analyze and store all songs in `paths` that haven't been already analyzed.
    ///
    /// Use this function if you don't have any extra data to bundle with each song.
    pub fn update_library(&mut self, paths: Vec<String>, show_progress_bar: bool) -> Result<()> {
        let paths_extra_info = paths.into_iter().map(|path| (path, ())).collect::<Vec<_>>();
        self.update_library_convert_extra_info(paths_extra_info, show_progress_bar, |x, _, _| x)
    }

    /// Analyze and store all songs in `paths_extra_info` that haven't already
    /// been analyzed, along with some extra metadata serializable, and known
    /// before song analysis.
    pub fn update_library_extra_info<T: Serialize + DeserializeOwned>(
        &mut self,
        paths_extra_info: Vec<(String, T)>,
        show_progress_bar: bool,
    ) -> Result<()> {
        self.update_library_convert_extra_info(
            paths_extra_info,
            show_progress_bar,
            |extra_info, _, _| extra_info,
        )
    }

    /// Analyze and store all songs in `paths_extra_info` that haven't
    /// been already analyzed, as well as handling extra, user-specified metadata,
    /// that can't directly be serializable,
    /// or that need input from the analyzed Song to be processed. If you
    /// just want to analyze and store songs along with some directly
    /// serializable values, consider using [update_library_extra_info].
    ///
    /// `paths_extra_info` is a tuple made out of song paths, along
    /// with any extra info you want to store for each song.
    ///
    /// `convert_extra_info` is a function that you should specify
    /// to convert that extra info to something serializable.
    pub fn update_library_convert_extra_info<T: Serialize + DeserializeOwned, U>(
        &mut self,
        paths_extra_info: Vec<(String, U)>,
        show_progress_bar: bool,
        convert_extra_info: fn(U, &Song, &Self) -> T,
    ) -> Result<()> {
        let existing_paths = {
            let connection = self
                .sqlite_conn
                .lock()
                .map_err(|e| BlissError::ProviderError(e.to_string()))?;
            let mut path_statement = connection.prepare(
                "
                select
                    path
                    from song where analyzed = true and version = ? order by id
                ",
            )?;
            #[allow(clippy::let_and_return)]
            let return_value = path_statement
                .query_map([FEATURES_VERSION], |row| Ok(row.get_unwrap(0)))?
                .map(|x| x.unwrap())
                .collect::<Vec<String>>();
            return_value
        };

        let paths_to_analyze = paths_extra_info
            .into_iter()
            .filter(|(path, _)| !existing_paths.contains(path))
            .collect::<Vec<_>>();
        self.analyze_paths_convert_extra_info(
            paths_to_analyze,
            show_progress_bar,
            convert_extra_info,
        )
    }

    /// Analyze and store all songs in `paths`.
    ///
    /// Use this function if you don't have any extra data to bundle with each song.
    pub fn analyze_paths(&mut self, paths: Vec<String>, show_progress_bar: bool) -> Result<()> {
        let paths_extra_info = paths.into_iter().map(|path| (path, ())).collect::<Vec<_>>();
        self.analyze_paths_convert_extra_info(paths_extra_info, show_progress_bar, |x, _, _| x)
    }

    /// Analyze and store all songs in `paths_extra_info`, along with some
    /// extra metadata serializable, and known before song analysis.
    pub fn analyze_paths_extra_info<T: Serialize + DeserializeOwned + std::fmt::Debug>(
        &mut self,
        paths_extra_info: Vec<(String, T)>,
        show_progress_bar: bool,
    ) -> Result<()> {
        self.analyze_paths_convert_extra_info(
            paths_extra_info,
            show_progress_bar,
            |extra_info, _, _| extra_info,
        )
    }

    /// Analyze and store all songs in `paths_extra_info`, along with some
    /// extra, user-specified metadata, that can't directly be serializable,
    /// or that need input from the analyzed Song to be processed.
    /// If you just want to analyze and store songs, along with some
    /// directly serializable metadata values, consider using
    /// [analyze_paths_extra_info].
    ///
    /// `paths_extra_info` is a tuple made out of song paths, along
    /// with any extra info you want to store for each song.
    ///
    /// `convert_extra_info` is a function that you should specify
    /// to convert that extra info to something serializable.
    pub fn analyze_paths_convert_extra_info<T: Serialize + DeserializeOwned, U>(
        &mut self,
        paths_extra_info: Vec<(String, U)>,
        show_progress_bar: bool,
        convert_extra_info: fn(U, &Song, &Self) -> T,
    ) -> Result<()> {
        let number_songs = paths_extra_info.len();
        if number_songs == 0 {
            log::info!("No (new) songs found.");
            return Ok(());
        }
        log::info!(
            "Analyzing {} songs, this might take some time…",
            number_songs
        );
        let pb = if show_progress_bar {
            ProgressBar::new(number_songs.try_into().unwrap())
        } else {
            ProgressBar::hidden()
        };
        let style = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos:>7}/{len:7} {wide_msg}")?
            .progress_chars("##-");
        pb.set_style(style);

        let mut paths_extra_info: HashMap<String, U> = paths_extra_info.into_iter().collect();
        let results = analyze_paths(paths_extra_info.keys());
        let mut success_count = 0;
        let mut failure_count = 0;
        for (path, result) in results {
            if show_progress_bar {
                pb.set_message(format!("Analyzing {}", path));
            }
            match result {
                Ok(song) => {
                    let extra = paths_extra_info.remove(&path).unwrap();
                    let e = convert_extra_info(extra, &song, self);
                    let library_song = LibrarySong::<T> {
                        bliss_song: song,
                        extra_info: e,
                    };
                    self.store_song(&library_song)?;
                    success_count += 1;
                }
                Err(e) => {
                    log::error!(
                        "Analysis of song '{}' failed: {} The error has been stored.",
                        path,
                        e
                    );

                    self.store_failed_song(path, e)?;
                    failure_count += 1;
                }
            };
            pb.inc(1);
        }
        pb.finish_with_message(format!(
            "Analyzed {} song(s) successfully. {} Failure(s).",
            success_count, failure_count
        ));

        log::info!(
            "Analyzed {} song(s) successfully. {} Failure(s).",
            success_count,
            failure_count,
        );

        Ok(())
    }

    /// Retrieve all songs which have been analyzed with
    /// current bliss version.
    ///
    /// Returns an error if one or several songs have a different number of
    /// features than they should, indicating the offending song id.
    ///
    // TODO maybe allow to specify the version?
    // TODO maybe the error should make the song id / song path
    // accessible easily?
    pub fn songs_from_library<T: Serialize + DeserializeOwned>(
        &self,
    ) -> Result<Vec<LibrarySong<T>>> {
        let connection = self
            .sqlite_conn
            .lock()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        let mut songs_statement = connection.prepare(
            "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version, extra_info, id
                from song where analyzed = true and version = ? order by id
            ",
        )?;
        let mut features_statement = connection.prepare(
            "
            select
                feature, song.id from feature join song on song.id = feature.song_id
                where song.analyzed = true and song.version = ? order by song_id, feature_index
                ",
        )?;
        let song_rows = songs_statement.query_map([FEATURES_VERSION], |row| {
            Ok((row.get(10)?, Self::_song_from_row_closure(row)?))
        })?;
        let feature_rows = features_statement
            .query_map([FEATURES_VERSION], |row| Ok((row.get(1)?, row.get(0)?)))?;

        let mut feature_iterator = feature_rows.into_iter().peekable();
        let mut songs = Vec::new();
        // Poor man's way to double check that each feature correspond to the
        // right song, and group them.
        for row in song_rows {
            let song_id: u32 = row.as_ref().unwrap().0;
            let mut chunk: Vec<f32> = Vec::with_capacity(NUMBER_FEATURES);

            while let Some(first_value) = feature_iterator.peek() {
                let (song_feature_id, feature): (u32, f32) = *first_value.as_ref().unwrap();
                if song_feature_id == song_id {
                    chunk.push(feature);
                    feature_iterator.next();
                } else {
                    break;
                };
            }
            let mut song = row.unwrap().1;
            song.bliss_song.analysis = Analysis {
                internal_analysis: chunk.try_into().map_err(|_| {
                    BlissError::ProviderError(format!(
                        "Song with ID {} and path {} has a different feature \
                        number than expected. Please rescan or update \
                        the song library.",
                        song_id,
                        song.bliss_song.path.display(),
                    ))
                })?,
            };
            songs.push(song);
        }
        Ok(songs)
    }

    fn _song_from_row_closure<T: Serialize + DeserializeOwned>(
        row: &Row,
    ) -> Result<LibrarySong<T>, RusqliteError> {
        let path: String = row.get(0)?;
        let song = Song {
            path: PathBuf::from(path),
            artist: row.get(1).unwrap(),
            title: row.get(2).unwrap(),
            album: row.get(3).unwrap(),
            album_artist: row.get(4).unwrap(),
            track_number: row.get(5).unwrap(),
            genre: row.get(6).unwrap(),
            analysis: Analysis {
                internal_analysis: [0.; NUMBER_FEATURES],
            },
            duration: Duration::from_secs_f64(row.get(7).unwrap()),
            features_version: row.get(8).unwrap(),
            cue_info: None,
        };

        let serialized: String = row.get(9).unwrap();
        let extra_info = serde_json::from_str(&serialized).unwrap();
        Ok(LibrarySong {
            bliss_song: song,
            extra_info,
        })
    }

    /// Get a LibrarySong from a given file path.
    pub fn song_from_path<T: Serialize + DeserializeOwned>(
        &self,
        song_path: &str,
    ) -> Result<LibrarySong<T>> {
        let connection = self
            .sqlite_conn
            .lock()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        // Get the song's metadata. The analysis is populated yet.
        let mut song = connection.query_row(
            "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version, extra_info
                from song where path=? and analyzed = true
            ",
            params![song_path],
            Self::_song_from_row_closure,
        )?;

        // Get the song's analysis, and attach it to the existing song.
        let mut stmt = connection.prepare(
            "
            select
                feature from feature join song on song.id = feature.song_id
                where song.path = ? order by feature_index
            ",
        )?;
        let analysis_vector = Analysis {
            internal_analysis: stmt
                .query_map(params![song_path], |row| row.get(0))
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>()
                .try_into()
                .map_err(|_| {
                    BlissError::ProviderError(format!(
                        "song has more or less than {} features",
                        NUMBER_FEATURES
                    ))
                })?,
        };
        song.bliss_song.analysis = analysis_vector;
        Ok(song)
    }

    /// Store a [Song] in the database, overidding any existing
    /// song with the same path by that one.
    pub fn store_song<T: Serialize + DeserializeOwned>(
        &mut self,
        library_song: &LibrarySong<T>,
    ) -> Result<(), BlissError> {
        let mut sqlite_conn = self.sqlite_conn.lock().unwrap();
        let tx = sqlite_conn
            .transaction()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        let song = &library_song.bliss_song;
        tx.execute(
            "
            insert into song (
                path, artist, title, album, album_artist,
                duration, track_number, genre, analyzed, version, extra_info
            )
            values (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
            )
            on conflict(path)
            do update set
                artist=excluded.artist,
                title=excluded.title,
                album=excluded.album,
                track_number=excluded.track_number,
                album_artist=excluded.album_artist,
                duration=excluded.duration,
                genre=excluded.genre,
                analyzed=excluded.analyzed,
                version=excluded.version,
                extra_info=excluded.extra_info
            ",
            params![
                song.path.to_str(),
                song.artist,
                song.title,
                song.album,
                song.album_artist,
                song.duration.as_secs_f64(),
                song.track_number,
                song.genre,
                true,
                song.features_version,
                serde_json::to_string(&library_song.extra_info)
                    .map_err(|e| BlissError::ProviderError(e.to_string()))?,
            ],
        )
        .map_err(|e| BlissError::ProviderError(e.to_string()))?;

        // Override existing features.
        tx.execute(
            "delete from feature where song_id in (select id from song where path = ?1);",
            params![song.path.to_str()],
        )
        .map_err(|e| BlissError::ProviderError(e.to_string()))?;

        for (index, feature) in song.analysis.as_vec().iter().enumerate() {
            tx.execute(
                "
                insert into feature (song_id, feature, feature_index)
                values ((select id from song where path = ?1), ?2, ?3)
                on conflict(song_id, feature_index) do update set feature=excluded.feature;
                ",
                params![song.path.to_str(), feature, index],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        Ok(())
    }

    /// Store an errored [Song](Song) in the SQLite database.
    ///
    /// If there already is an existing song with that path, replace it by
    /// the latest failed result.
    pub fn store_failed_song(&mut self, song_path: String, e: BlissError) -> Result<()> {
        self.sqlite_conn
            .lock()
            .unwrap()
            .execute(
                "
            insert or replace into song (path, error) values (?1, ?2)
            ",
                [song_path, e.to_string()],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        Ok(())
    }

    /// Delete a song with path `song_path` from the database.
    ///
    /// Errors out if the song is not in the database.
    pub fn delete_song(&mut self, song_path: String) -> Result<()> {
        let count = self
            .sqlite_conn
            .lock()
            .unwrap()
            .execute(
                "
                delete from song where path = ?1;
            ",
                [song_path.to_owned()],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        if count == 0 {
            bail!(BlissError::ProviderError(format!(
                "tried to delete song {}, not existing in the database.",
                song_path,
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
fn data_local_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/local/directory"))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Analysis, NUMBER_FEATURES};
    use pretty_assertions::assert_eq;
    use serde::{de::DeserializeOwned, Deserialize};
    use std::{convert::TryInto, fmt::Debug, sync::MutexGuard, time::Duration};
    use tempdir::TempDir;

    #[derive(Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
    struct ExtraInfo {
        ignore: bool,
        metadata_bliss_does_not_have: String,
    }

    #[derive(Deserialize, Serialize, PartialEq, Eq, Debug, Clone)]
    struct CustomConfig {
        #[serde(flatten)]
        base_config: BaseConfig,
        second_path_to_music_library: String,
        ignore_wav_files: bool,
    }

    impl AppConfigTrait for CustomConfig {
        fn base_config(&self) -> &BaseConfig {
            &self.base_config
        }
    }

    // Returning the TempDir here, so it doesn't go out of scope, removing
    // the directory.
    //
    // Setup a test library made of 3 analyzed songs, with every field being different,
    // as well as an unanalyzed song and a song analyzed with a previous version.
    fn setup_test_library() -> (
        Library<BaseConfig>,
        TempDir,
        (
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
        ),
    ) {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");
        let library =
            Library::<BaseConfig>::new_from_base(Some(config_file), Some(database_file)).unwrap();

        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 / 10.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        let song = Song {
            path: "/path/to/song1001".into(),
            artist: Some("Artist1001".into()),
            title: Some("Title1001".into()),
            album: Some("An Album1001".into()),
            album_artist: Some("An Album Artist1001".into()),
            track_number: Some("01".into()),
            genre: Some("Electronica1001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(310),
            features_version: 1,
            cue_info: None,
        };
        let first_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: true,
                metadata_bliss_does_not_have: String::from("/path/to/charlie1001"),
            },
        };
        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 + 10.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let song = Song {
            path: "/path/to/song2001".into(),
            artist: Some("Artist2001".into()),
            title: Some("Title2001".into()),
            album: Some("An Album2001".into()),
            album_artist: Some("An Album Artist2001".into()),
            track_number: Some("02".into()),
            genre: Some("Electronica2001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(410),
            features_version: 1,
            cue_info: None,
        };
        let second_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie2001"),
            },
        };
        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 / 2.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let song = Song {
            path: "/path/to/song5001".into(),
            artist: Some("Artist5001".into()),
            title: Some("Title5001".into()),
            album: Some("An Album5001".into()),
            album_artist: Some("An Album Artist5001".into()),
            track_number: Some("04".into()),
            genre: Some("Electronica5001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(610),
            features_version: 1,
            cue_info: None,
        };
        let third_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie5001"),
            },
        };

        {
            let connection = library.sqlite_conn.lock().unwrap();
            connection
                .execute(
                    "
                    insert into song (
                        id, path, artist, title, album, album_artist, track_number,
                        genre, duration, analyzed, version, extra_info
                    ) values (
                        1001, '/path/to/song1001', 'Artist1001', 'Title1001', 'An Album1001',
                        'An Album Artist1001', '01', 'Electronica1001', 310, true,
                        1, '{\"ignore\": true, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie1001\"}'
                    ),
                    (
                        2001, '/path/to/song2001', 'Artist2001', 'Title2001', 'An Album2001',
                        'An Album Artist2001', '02', 'Electronica2001', 410, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie2001\"}'
                    ),
                    (
                        3001, '/path/to/song3001', null, null, null,
                        null, null, null, null, false,
                        1, '{}'
                    ),
                    (
                        4001, '/path/to/song4001', 'Artist4001', 'Title4001', 'An Album4001',
                        'An Album Artist4001', '03', 'Electronica4001', 510, true,
                        0, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie4001\"}'
                    ),
                    (
                        5001, '/path/to/song5001', 'Artist5001', 'Title5001', 'An Album5001',
                        'An Album Artist5001', '04', 'Electronica5001', 610, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie5001\"}'
                    );
                    ",
                    [],
                )
                .unwrap();
            for index in 0..NUMBER_FEATURES {
                connection
                    .execute(
                        "
                            insert into feature(song_id, feature, feature_index)
                            values (1001, ?1, ?2), (2001, ?3, ?2), (3001, ?4, ?2), (5001, ?5, ?2);
                            ",
                        params![
                            index as f32 / 10.,
                            index,
                            index as f32 + 10.,
                            index as f32 / 10. + 1.,
                            index as f32 / 2.
                        ],
                    )
                    .unwrap();
            }
        }
        (library, config_dir, (first_song, second_song, third_song))
    }

    fn _library_song_from_database<T: DeserializeOwned + Serialize + Clone + Debug>(
        connection: MutexGuard<Connection>,
        song_path: &str,
    ) -> LibrarySong<T> {
        let mut song = connection
            .query_row(
                "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version, extra_info
                from song where path=?
            ",
                params![song_path],
                |row| {
                    let path: String = row.get(0)?;
                    let song = Song {
                        path: PathBuf::from(path),
                        artist: row.get(1).unwrap(),
                        title: row.get(2).unwrap(),
                        album: row.get(3).unwrap(),
                        album_artist: row.get(4).unwrap(),
                        track_number: row.get(5).unwrap(),
                        genre: row.get(6).unwrap(),
                        analysis: Analysis {
                            internal_analysis: [0.; NUMBER_FEATURES],
                        },
                        duration: Duration::from_secs_f64(row.get(7).unwrap()),
                        features_version: row.get(8).unwrap(),
                        cue_info: None,
                    };

                    let serialized: String = row.get(9).unwrap();
                    let extra_info = serde_json::from_str(&serialized).unwrap();
                    Ok(LibrarySong {
                        bliss_song: song,
                        extra_info,
                    })
                },
            )
            .expect("Song probably does not exist in the db.");
        let mut stmt = connection
            .prepare(
                "
            select
                feature from feature join song on song.id = feature.song_id
                where song.path = ? order by feature_index
            ",
            )
            .unwrap();
        let analysis_vector = Analysis {
            internal_analysis: stmt
                .query_map(params![song_path], |row| row.get(0))
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        };
        song.bliss_song.analysis = analysis_vector;
        song
    }

    fn _basic_song_from_database(connection: MutexGuard<Connection>, song_path: &str) -> Song {
        let mut expected_song = connection
            .query_row(
                "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version
                from song where path=? and analyzed = true
            ",
                params![song_path],
                |row| {
                    let path: String = row.get(0)?;
                    Ok(Song {
                        path: PathBuf::from(path),
                        artist: row.get(1).unwrap(),
                        title: row.get(2).unwrap(),
                        album: row.get(3).unwrap(),
                        album_artist: row.get(4).unwrap(),
                        track_number: row.get(5).unwrap(),
                        genre: row.get(6).unwrap(),
                        analysis: Analysis {
                            internal_analysis: [0.; NUMBER_FEATURES],
                        },
                        duration: Duration::from_secs_f64(row.get(7).unwrap()),
                        features_version: row.get(8).unwrap(),
                        cue_info: None,
                    })
                },
            )
            .expect("Song is probably not in the db");
        let mut stmt = connection
            .prepare(
                "
            select
                feature from feature join song on song.id = feature.song_id
                where song.path = ? order by feature_index
            ",
            )
            .unwrap();
        let expected_analysis_vector = Analysis {
            internal_analysis: stmt
                .query_map(params![song_path], |row| row.get(0))
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        };
        expected_song.analysis = expected_analysis_vector;
        expected_song
    }

    fn _generate_basic_song(path: Option<String>) -> Song {
        let path = path.unwrap_or_else(|| "/path/to/song".into());
        // Add some "randomness" to the features
        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 + 0.1)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        Song {
            path: path.into(),
            artist: Some("An Artist".into()),
            title: Some("Title".into()),
            album: Some("An Album".into()),
            album_artist: Some("An Album Artist".into()),
            track_number: Some("03".into()),
            genre: Some("Electronica".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(80),
            features_version: 1,
            cue_info: None,
        }
    }

    fn _generate_library_song(path: Option<String>) -> LibrarySong<ExtraInfo> {
        let song = _generate_basic_song(path);
        let extra_info = ExtraInfo {
            ignore: true,
            metadata_bliss_does_not_have: "FoobarIze".into(),
        };
        LibrarySong {
            bliss_song: song,
            extra_info,
        }
    }

    #[test]
    fn test_library_playlist_song_not_existing() {
        let (library, _temp_dir, _) = setup_test_library();
        assert!(library
            .playlist_from::<ExtraInfo>("not-existing", 2)
            .is_err());
    }

    #[test]
    fn test_library_playlist_crop() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> =
            library.playlist_from("/path/to/song2001", 2).unwrap();
        assert_eq!(2, songs.len());
    }

    #[test]
    fn test_library_simple_playlist() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> =
            library.playlist_from("/path/to/song2001", 20).unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song5001",
                "/path/to/song1001"
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    #[test]
    fn test_library_delete_song_non_existing() {
        let (mut library, _temp_dir, _) = setup_test_library();
        {
            let connection = library.sqlite_conn.lock().unwrap();
            let count: u32 = connection
                    .query_row(
                        "select count(*) from feature join song on song.id = feature.song_id where song.path = ?",
                        ["not-existing"],
                        |row| row.get(0),
                    )
                    .unwrap();
            assert_eq!(count, 0);
            let count: u32 = connection
                .query_row(
                    "select count(*) from song where path = ?",
                    ["not-existing"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 0);
        }
        assert!(library.delete_song("not-existing".into()).is_err());
    }

    #[test]
    fn test_library_delete_song() {
        let (mut library, _temp_dir, _) = setup_test_library();
        {
            let connection = library.sqlite_conn.lock().unwrap();
            let count: u32 = connection
                    .query_row(
                        "select count(*) from feature join song on song.id = feature.song_id where song.path = ?",
                        ["/path/to/song1001"],
                        |row| row.get(0),
                    )
                    .unwrap();
            assert!(count >= 1);
            let count: u32 = connection
                .query_row(
                    "select count(*) from song where path = ?",
                    ["/path/to/song1001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(count >= 1);
        }

        library
            .delete_song(String::from("/path/to/song1001"))
            .unwrap();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            let count: u32 = connection
                .query_row(
                    "select count(*) from feature join song on song.id = feature.song_id where song.path = ?",
                    ["/path/to/song1001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(0, count);
            let count: u32 = connection
                .query_row(
                    "select count(*) from song where path = ?",
                    ["/path/to/song1001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(0, count);
        }
    }

    #[test]
    fn test_analyze_paths() {
        let (mut library, _temp_dir, _) = setup_test_library();

        let paths = vec![
            "./data/s16_mono_22_5kHz.flac".into(),
            "./data/s16_stereo_22_5kHz.flac".into(),
            "non-existing".into(),
        ];
        library.analyze_paths(paths.to_owned(), false).unwrap();
        let songs = paths[..2]
            .iter()
            .map(|path| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<()>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(vec![(), ()].into_iter())
            .map(|(path, expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<()>>>();
        assert_eq!(songs, expected_songs);
    }

    #[test]
    fn test_analyze_paths_convert_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac".into(), true),
            ("./data/s16_stereo_22_5kHz.flac".into(), false),
            ("non-existing".into(), false),
        ];
        library
            .analyze_paths_convert_extra_info::<ExtraInfo, bool>(
                paths.to_owned(),
                true,
                |b, _, _| ExtraInfo {
                    ignore: b,
                    metadata_bliss_does_not_have: String::from("coucou"),
                },
            )
            .unwrap();
        library
            .analyze_paths_convert_extra_info::<ExtraInfo, bool>(
                paths.to_owned(),
                false,
                |b, _, _| ExtraInfo {
                    ignore: b,
                    metadata_bliss_does_not_have: String::from("coucou"),
                },
            )
            .unwrap();
        let songs = paths[..2]
            .iter()
            .map(|(path, _)| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(
                vec![
                    ExtraInfo {
                        ignore: true,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                    ExtraInfo {
                        ignore: false,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                ]
                .into_iter(),
            )
            .map(|((path, _extra_info), expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
    }

    #[test]
    fn test_analyze_paths_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();

        let paths = vec![
            (
                "./data/s16_mono_22_5kHz.flac".into(),
                ExtraInfo {
                    ignore: true,
                    metadata_bliss_does_not_have: String::from("hey"),
                },
            ),
            (
                "./data/s16_stereo_22_5kHz.flac".into(),
                ExtraInfo {
                    ignore: false,
                    metadata_bliss_does_not_have: String::from("hello"),
                },
            ),
            (
                "non-existing".into(),
                ExtraInfo {
                    ignore: true,
                    metadata_bliss_does_not_have: String::from("coucou"),
                },
            ),
        ];
        library
            .analyze_paths_extra_info::<ExtraInfo>(paths.to_owned(), false)
            .unwrap();
        let songs = paths[..2]
            .iter()
            .map(|(path, _)| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(
                vec![
                    ExtraInfo {
                        ignore: true,
                        metadata_bliss_does_not_have: String::from("hey"),
                    },
                    ExtraInfo {
                        ignore: false,
                        metadata_bliss_does_not_have: String::from("hello"),
                    },
                ]
                .into_iter(),
            )
            .map(|((path, _extra_info), expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
    }

    #[test]
    // Check that a song already in the database is not
    // analyzed again on updates.
    fn test_update_skip_analyzed() {
        let (mut library, _temp_dir, _) = setup_test_library();

        for input in vec![
            ("./data/s16_mono_22_5kHz.flac".into(), true),
            ("./data/s16_mono_22_5khz.flac".into(), false),
        ]
        .into_iter()
        {
            let paths = vec![input.to_owned()];
            library
                .update_library_convert_extra_info::<ExtraInfo, bool>(
                    paths.to_owned(),
                    false,
                    |b, _, _| ExtraInfo {
                        ignore: b,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                )
                .unwrap();
            let song = {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database::<ExtraInfo>(connection, "./data/s16_mono_22_5kHz.flac")
            };
            let expected_song = {
                LibrarySong {
                    bliss_song: Song::from_path("./data/s16_mono_22_5kHz.flac").unwrap(),
                    extra_info: ExtraInfo {
                        ignore: true,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                }
            };
            assert_eq!(song, expected_song);
        }
    }

    fn _get_song_analyzed(connection: MutexGuard<Connection>, path: String) -> bool {
        let mut stmt = connection
            .prepare(
                "
                select
                    analyzed from song
                    where song.path = ?
                ",
            )
            .unwrap();
        stmt.query_row([path], |row| row.get(0)).unwrap()
    }

    #[test]
    fn test_update_library() {
        let (mut library, _temp_dir, _) = setup_test_library();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()));
        }

        let paths = vec![
            "./data/s16_mono_22_5kHz.flac".into(),
            "./data/s16_stereo_22_5kHz.flac".into(),
            "/path/to/song4001".into(),
            "non-existing".into(),
        ];
        library.update_library(paths.to_owned(), false).unwrap();
        library.update_library(paths.to_owned(), true).unwrap();

        let songs = paths[..2]
            .iter()
            .map(|path| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<()>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(vec![(), ()].into_iter())
            .map(|(path, expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<()>>>();

        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()));
        }
    }

    #[test]
    fn test_update_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()));
        }

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac".into(), true),
            ("./data/s16_stereo_22_5kHz.flac".into(), false),
            ("/path/to/song4001".into(), false),
            ("non-existing".into(), false),
        ];
        library
            .update_library_extra_info::<bool>(paths.to_owned(), false)
            .unwrap();
        let songs = paths[..2]
            .iter()
            .map(|(path, _)| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<bool>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(vec![true, false].into_iter())
            .map(|((path, _extra_info), expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<bool>>>();
        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()));
        }
    }

    #[test]
    fn test_update_convert_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()));
        }

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac".into(), true),
            ("./data/s16_stereo_22_5kHz.flac".into(), false),
            ("/path/to/song4001".into(), false),
            ("non-existing".into(), false),
        ];
        library
            .update_library_convert_extra_info::<ExtraInfo, bool>(
                paths.to_owned(),
                false,
                |b, _, _| ExtraInfo {
                    ignore: b,
                    metadata_bliss_does_not_have: String::from("coucou"),
                },
            )
            .unwrap();
        let songs = paths[..2]
            .iter()
            .map(|(path, _)| {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database(connection, path)
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        let expected_songs = paths[..2]
            .iter()
            .zip(
                vec![
                    ExtraInfo {
                        ignore: true,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                    ExtraInfo {
                        ignore: false,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                ]
                .into_iter(),
            )
            .map(|((path, _extra_info), expected_extra_info)| LibrarySong {
                bliss_song: Song::from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()));
        }
    }

    #[test]
    fn test_song_from_path() {
        let (library, _temp_dir, _) = setup_test_library();
        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 + 10.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let song = Song {
            path: "/path/to/song2001".into(),
            artist: Some("Artist2001".into()),
            title: Some("Title2001".into()),
            album: Some("An Album2001".into()),
            album_artist: Some("An Album Artist2001".into()),
            track_number: Some("02".into()),
            genre: Some("Electronica2001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(410),
            features_version: 1,
            cue_info: None,
        };
        let expected_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie2001"),
            },
        };

        let song = library
            .song_from_path::<ExtraInfo>("/path/to/song2001")
            .unwrap();
        assert_eq!(song, expected_song)
    }

    #[test]
    fn test_store_failed_song() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library
            .store_failed_song(
                "/some/failed/path".into(),
                BlissError::ProviderError("error with the analysis".into()),
            )
            .unwrap();
        let connection = library.sqlite_conn.lock().unwrap();
        let (error, analyzed): (String, bool) = connection
            .query_row(
                "
            select
                error, analyzed
                from song where path=?
            ",
                params!["/some/failed/path"],
                |row| Ok((row.get_unwrap(0), row.get_unwrap(1))),
            )
            .unwrap();
        assert_eq!(
            error,
            String::from(
                "error happened with the music library provider - error with the analysis"
            )
        );
        assert_eq!(analyzed, false);
        let count_features: u32 = connection
            .query_row(
                "
            select
                count(*) from feature join song
                on song.id = feature.song_id where path=?
            ",
                params!["/some/failed/path"],
                |row| Ok(row.get_unwrap(0)),
            )
            .unwrap();
        assert_eq!(count_features, 0);
    }

    #[test]
    fn test_songs_from_library() {
        let (library, _temp_dir, expected_library_songs) = setup_test_library();

        let library_songs = library.songs_from_library::<ExtraInfo>().unwrap();
        assert_eq!(library_songs.len(), 3);
        assert_eq!(
            expected_library_songs,
            (
                library_songs[0].to_owned(),
                library_songs[1].to_owned(),
                library_songs[2].to_owned()
            )
        );
    }

    #[test]
    fn test_songs_from_library_screwed_db() {
        let (library, _temp_dir, _) = setup_test_library();
        {
            let connection = library.sqlite_conn.lock().unwrap();
            connection
                .execute(
                    "insert into feature (song_id, feature, feature_index)
                values (2001, 1.5, 21)
                ",
                    [],
                )
                .unwrap();
        }

        let error = library.songs_from_library::<ExtraInfo>().unwrap_err();
        assert_eq!(
            error.to_string(),
            String::from(
                "error happened with the music library provider - \
                Song with ID 2001 and path /path/to/song2001 has a \
                different feature number than expected. Please rescan or \
                update the song library.",
            ),
        );
    }

    #[test]
    fn test_song_from_path_not_analyzed() {
        let (library, _temp_dir, _) = setup_test_library();
        let error = library.song_from_path::<ExtraInfo>("/path/to/song4001");
        assert!(error.is_err());
    }

    #[test]
    fn test_song_from_path_not_found() {
        let (library, _temp_dir, _) = setup_test_library();
        let error = library.song_from_path::<ExtraInfo>("/path/to/song4001");
        assert!(error.is_err());
    }

    #[test]
    fn test_get_default_data_folder_no_default_path() {
        env::set_var("XDG_DATA_HOME", "/home/foo/.local/share/");
        assert_eq!(
            PathBuf::from("/home/foo/.local/share/bliss-rs"),
            BaseConfig::get_default_data_folder().unwrap()
        );
        env::remove_var("XDG_DATA_HOME");

        assert_eq!(
            PathBuf::from("/local/directory/bliss-rs"),
            BaseConfig::get_default_data_folder().unwrap()
        );
    }

    #[test]
    fn test_library_new_default_write() {
        let (library, _temp_dir, _) = setup_test_library();
        let config_content = fs::read_to_string(&library.config.base_config().config_path).unwrap();
        assert_eq!(
            config_content,
            format!(
                "{{\"config_path\":\"{}\",\"database_path\":\"{}\"}}",
                library.config.base_config().config_path.display(),
                library.config.base_config().database_path.display(),
            )
        );
    }

    #[test]
    fn test_library_new_create_database() {
        let (library, _temp_dir, _) = setup_test_library();
        let sqlite_conn = Connection::open(&library.config.base_config().database_path).unwrap();
        sqlite_conn
            .execute(
                "
            insert into song (
                id, path, artist, title, album, album_artist,
                track_number, genre, stamp, version, duration, analyzed,
                extra_info
            )
            values (
                1, '/random/path', 'Some Artist', 'A Title', 'Some Album',
                'Some Album Artist', '01', 'Electronica', '2022-01-01',
                1, 250, true, '{\"key\": \"value\"}'
            );
            ",
                [],
            )
            .unwrap();
        sqlite_conn
            .execute(
                "
            insert into feature(id, song_id, feature, feature_index)
            values (2000, 1, 1.1, 1)
            on conflict(song_id, feature_index) do update set feature=excluded.feature;
            ",
                [],
            )
            .unwrap();
    }

    #[test]
    fn test_library_store_song() {
        let (mut library, _temp_dir, _) = setup_test_library();
        let song = _generate_basic_song(None);
        let library_song = LibrarySong {
            bliss_song: song.to_owned(),
            extra_info: (),
        };
        library.store_song(&library_song).unwrap();
        let connection = library.sqlite_conn.lock().unwrap();
        let expected_song = _basic_song_from_database(connection, &song.path.to_string_lossy());
        assert_eq!(expected_song, song);
    }

    #[test]
    fn test_library_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();
        let song = _generate_library_song(None);
        library.store_song(&song).unwrap();
        let connection = library.sqlite_conn.lock().unwrap();
        let returned_song =
            _library_song_from_database(connection, &song.bliss_song.path.to_string_lossy());
        assert_eq!(returned_song, song);
    }

    #[test]
    fn test_from_config_path_non_existing() {
        assert!(
            Library::<CustomConfig>::from_config_path(Some(PathBuf::from("non-existing"))).is_err()
        );
    }

    #[test]
    fn test_from_config_path() {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");

        // In reality, someone would just do that with `(None, None)` to get the default
        // paths.
        let base_config =
            BaseConfig::new(Some(config_file.to_owned()), Some(database_file)).unwrap();

        let config = CustomConfig {
            base_config,
            second_path_to_music_library: "/path/to/somewhere".into(),
            ignore_wav_files: true,
        };
        // Test that it is possible to store a song in a library instance,
        // make that instance go out of scope, load the library again, and
        // get the stored song.
        let song = _generate_library_song(None);
        {
            let mut library = Library::new(config.to_owned()).unwrap();
            library.store_song(&song).unwrap();
        }

        let library: Library<CustomConfig> = Library::from_config_path(Some(config_file)).unwrap();
        let connection = library.sqlite_conn.lock().unwrap();
        let returned_song =
            _library_song_from_database(connection, &song.bliss_song.path.to_string_lossy());

        assert_eq!(library.config, config);
        assert_eq!(song, returned_song);
    }

    #[test]
    fn test_config_serialize_deserialize() {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");

        // In reality, someone would just do that with `(None, None)` to get the default
        // paths.
        let base_config =
            BaseConfig::new(Some(config_file.to_owned()), Some(database_file)).unwrap();

        let config = CustomConfig {
            base_config,
            second_path_to_music_library: "/path/to/somewhere".into(),
            ignore_wav_files: true,
        };
        config.write().unwrap();

        assert_eq!(
            config,
            CustomConfig::from_path(&config_file.to_string_lossy()).unwrap(),
        );
    }
}
