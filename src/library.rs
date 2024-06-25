//! Module containing utilities to properly manage a library of [Song]s,
//! for people wanting to e.g. implement a bliss plugin for an existing
//! audio player. A good resource to look at for inspiration is
//! [blissify](https://github.com/Polochon-street/blissify-rs)'s source code.
//!
//! Useful to have direct and easy access to functions that analyze
//! and store analysis of songs in a SQLite database, as well as retrieve it,
//! and make playlists directly from analyzed songs. All functions are as
//! thoroughly tested as possible, so you don't have to do it yourself,
//! including for instance bliss features version handling, etc.
//!
//! It works in three parts:
//! * The first part is the configuration part, which allows you to
//!   specify extra information that your plugin might need that will
//!   be automatically stored / retrieved when you instanciate a
//!   [Library] (the core of your plugin).
//!
//!   To do so implies specifying a configuration struct, that will implement
//!   [AppConfigTrait], i.e. implement `Serialize`, `Deserialize`, and a
//!   function to retrieve the [BaseConfig] (which is just a structure
//!   holding the path to the configuration file and the path to the database).
//!
//!   The most straightforward way to do so is to have something like this (
//!   in this example, we assume that `path_to_extra_information` is something
//!   you would want stored in your configuration file, path to a second music
//!   folder for instance:
//!   ```
//!     use anyhow::Result;
//!     use serde::{Deserialize, Serialize};
//!     use std::path::PathBuf;
//!     use std::num::NonZeroUsize;
//!     use bliss_audio::BlissError;
//!     use bliss_audio::library::{AppConfigTrait, BaseConfig};
//!
//!     #[derive(Serialize, Deserialize, Clone, Debug)]
//!     pub struct Config {
//!         #[serde(flatten)]
//!         pub base_config: BaseConfig,
//!         pub music_library_path: PathBuf,
//!     }
//!
//!     impl AppConfigTrait for Config {
//!         fn base_config(&self) -> &BaseConfig {
//!             &self.base_config
//!         }
//!
//!         fn base_config_mut(&mut self) -> &mut BaseConfig {
//!             &mut self.base_config
//!         }
//!     }
//!     impl Config {
//!         pub fn new(
//!             music_library_path: PathBuf,
//!             config_path: Option<PathBuf>,
//!             database_path: Option<PathBuf>,
//!             number_cores: Option<NonZeroUsize>,
//!         ) -> Result<Self> {
//!             // Note that by passing `(None, None)` here, the paths will
//!             // be inferred automatically using user data dirs.
//!             let base_config = BaseConfig::new(config_path, database_path, number_cores)?;
//!             Ok(Self {
//!                 base_config,
//!                 music_library_path,
//!             })
//!         }
//!     }
//!   ```
//!
//! * The second part is the actual [Library] structure, that makes the
//!   bulk of the plug-in. To initialize a library once with a given config,
//!   you can do (here with a base configuration, requiring ffmpeg):
#![cfg_attr(
    feature = "ffmpeg",
    doc = r##"
```no_run
  use anyhow::{Error, Result};
  use bliss_audio::library::{BaseConfig, Library};
  use bliss_audio::decoder::ffmpeg::FFmpeg;
  use std::path::PathBuf;

  let config_path = Some(PathBuf::from("path/to/config/config.json"));
  let database_path = Some(PathBuf::from("path/to/config/bliss.db"));
  let config = BaseConfig::new(config_path, database_path, None)?;
  let library: Library<BaseConfig, FFmpeg> = Library::new(config)?;
  # Ok::<(), Error>(())
```"##
)]
//!   Once this is done, you can simply load the library by doing
//!   `Library::from_config_path(config_path);`
//! * The third part is using the [Library] itself: it provides you with
//!   utilies such as [Library::analyze_paths], which analyzes all songs
//!   in given paths and stores it in the databases, as well as
//!   [Library::playlist_from], which allows you to generate a playlist
//!   from any given analyzed song.
//!
//!   The [Library] structure also comes with a [LibrarySong] song struct,
//!   which represents a song stored in the database.
//!
//!   It is made of a `bliss_song` field, containing the analyzed bliss
//!   song (with the normal metatada such as the artist, etc), and an
//!   `extra_info` field, which can be any user-defined serialized struct.
//!   For most use cases, it would just be the unit type `()` (which is no
//!   extra info), that would be used like
//!   `library.playlist_from<()>(song, path, playlist_length)`,
//!   but functions such as [Library::analyze_paths_extra_info] and
//!   [Library::analyze_paths_convert_extra_info] let you customize what
//!   information you store for each song.
//!
//! The files in
//! [examples/library.rs](https://github.com/Polochon-street/bliss-rs/blob/master/examples/library.rs)
//! and
//! [examples/libray_extra_info.rs](https://github.com/Polochon-street/bliss-rs/blob/master/examples/library_extra_info.rs)
//! should provide the user with enough information to start with. For a more
//! "real-life" example, the
//! [blissify](https://github.com/Polochon-street/blissify-rs)'s code is using
//! [Library] to implement bliss for a MPD player.
use crate::cue::CueInfo;
use crate::playlist::closest_album_to_group;
use crate::playlist::closest_to_songs;
use crate::playlist::dedup_playlist_custom_distance;
use crate::playlist::euclidean_distance;
use crate::playlist::DistanceMetricBuilder;
use anyhow::{bail, Context, Result};
#[cfg(not(test))]
use dirs::data_local_dir;
use indicatif::{ProgressBar, ProgressStyle};
use log::warn;
use rusqlite::params;
use rusqlite::params_from_iter;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::Params;
use rusqlite::Row;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::fs::create_dir_all;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use crate::decoder::Decoder as DecoderTrait;
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

    // Implementers have to provide these.
    /// This trait should return the [BaseConfig] from the parent,
    /// user-created `Config`.
    fn base_config_mut(&mut self) -> &mut BaseConfig;

    // Default implementation to output the config as a JSON file.
    /// Convert the current config to a [String], to be written to
    /// a file.
    ///
    /// The default writes a JSON file, but any format can be used,
    /// using for example the various Serde libraries (`serde_yaml`, etc) -
    /// just overwrite this method.
    fn serialize_config(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self)?)
    }

    /// Set the number of desired cores for analysis, and write it to the
    /// configuration file.
    fn set_number_cores(&mut self, number_cores: NonZeroUsize) -> Result<()> {
        self.base_config_mut().number_cores = number_cores;
        self.write()
    }

    /// Get the number of desired cores for analysis, and write it to the
    /// configuration file.
    fn get_number_cores(&self) -> NonZeroUsize {
        self.base_config().number_cores
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

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
/// The minimum configuration an application needs to work with
/// a [Library].
pub struct BaseConfig {
    /// The path to where the configuration file should be stored,
    /// e.g. `/home/foo/.local/share/bliss-rs/config.json`
    config_path: PathBuf,
    /// The path to where the database file should be stored,
    /// e.g. `/home/foo/.local/share/bliss-rs/bliss.db`
    database_path: PathBuf,
    /// The latest features version a song has been analyzed
    /// with.
    features_version: u16,
    /// The number of CPU cores an analysis will be performed with.
    /// Defaults to the number of CPUs in the user's computer.
    number_cores: NonZeroUsize,
}

impl BaseConfig {
    pub(crate) fn get_default_data_folder() -> Result<PathBuf> {
        let path = match env::var("XDG_DATA_HOME") {
            Ok(path) => Path::new(&path).join("bliss-rs"),
            Err(_) => {
                    data_local_dir()
                    .with_context(|| "No suitable path found to store bliss' song database. Consider specifying such a path.")?
                    .join("bliss-rs")
            },
        };
        Ok(path)
    }

    /// Create a new, basic config. Upon calls of `Config.write()`, it will be
    /// written to `config_path`.
    //
    /// Any path omitted will instead default to a "clever" path using
    /// data directory inference. The number of cores is the number of cores
    /// that should be used for any analysis. If not provided, it will default
    /// to the computer's number of cores.
    pub fn new(
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
        number_cores: Option<NonZeroUsize>,
    ) -> Result<Self> {
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

        let number_cores = number_cores.unwrap_or_else(|| {
            thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap())
        });

        Ok(Self {
            config_path,
            database_path,
            features_version: FEATURES_VERSION,
            number_cores,
        })
    }
}

impl AppConfigTrait for BaseConfig {
    fn base_config(&self) -> &BaseConfig {
        self
    }

    fn base_config_mut(&mut self) -> &mut BaseConfig {
        self
    }
}

/// A struct used to hold a collection of [Song]s, with convenience
/// methods to add, remove and update songs.
///
/// Provide it either the `BaseConfig`, or a `Config` extending
/// `BaseConfig`.
/// TODO code example
pub struct Library<Config, D: ?Sized> {
    /// The configuration struct, containing both information
    /// from `BaseConfig` as well as user-defined values.
    pub config: Config,
    /// SQL connection to the database.
    pub sqlite_conn: Arc<Mutex<Connection>>,
    decoder: PhantomData<D>,
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

impl<T: Serialize + DeserializeOwned> AsRef<Song> for LibrarySong<T> {
    fn as_ref(&self) -> &Song {
        &self.bliss_song
    }
}

// TODO add logging statement
// TODO concrete examples
// TODO example LibrarySong without any extra_info
// TODO maybe return number of elements updated / deleted / whatev in analysis
//      functions?
// TODO add full rescan
// TODO a song_from_path with custom filters
// TODO "smart" playlist
// TODO should it really use anyhow errors?
// TODO make sure that the path to string is consistent
// TODO make a function that returns a list of all analyzed songs in the db
impl<Config: AppConfigTrait, D: ?Sized + DecoderTrait> Library<Config, D> {
    /// Create a new [Library] object from the given Config struct that
    /// implements the [AppConfigTrait].
    /// writing the configuration to the file given in
    /// `config.config_path`.
    ///
    /// This function should only be called once, when a user wishes to
    /// create a completely new "library".
    /// Otherwise, load an existing library file using
    /// [Library::from_config_path].
    pub fn new(config: Config) -> Result<Self> {
        if !config
            .base_config()
            .config_path
            .parent()
            .ok_or_else(|| {
                BlissError::ProviderError(format!(
                    "specified path {} is not a valid file path.",
                    config.base_config().config_path.display()
                ))
            })?
            .is_dir()
        {
            create_dir_all(config.base_config().config_path.parent().unwrap())?;
        }
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
                cue_path text,
                audio_file_path text,
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
        Ok(Self {
            config,
            sqlite_conn: Arc::new(Mutex::new(sqlite_conn)),
            decoder: PhantomData,
        })
    }

    /// Load a library from a configuration path.
    ///
    /// If no configuration path is provided, the path is
    /// set using default data folder path.
    pub fn from_config_path(config_path: Option<PathBuf>) -> Result<Self> {
        let config_path: Result<PathBuf> =
            config_path.map_or_else(|| Ok(BaseConfig::new(None, None, None)?.config_path), Ok);
        let config_path = config_path?;
        let data = fs::read_to_string(config_path)?;
        let config = Config::deserialize_config(&data)?;
        let sqlite_conn = Connection::open(&config.base_config().database_path)?;
        let mut library = Self {
            config,
            sqlite_conn: Arc::new(Mutex::new(sqlite_conn)),
            decoder: PhantomData,
        };
        if !library.version_sanity_check()? {
            warn!(
                "Songs have been analyzed with different versions of bliss; \
                older versions will be ignored from playlists. Update your \
                bliss library to correct the issue."
            );
        }
        Ok(library)
    }

    /// Check whether the library contains songs analyzed with different,
    /// incompatible versions of bliss.
    ///
    /// Returns true if the database is clean (only one version of the
    /// features), and false otherwise.
    pub fn version_sanity_check(&mut self) -> Result<bool> {
        let connection = self
            .sqlite_conn
            .lock()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        let count: u32 = connection
            .query_row("select count(distinct version) from song", [], |row| {
                row.get(0)
            })
            .optional()?
            .unwrap_or(0);
        Ok(count <= 1)
    }

    /// Create a new [Library] object from a minimal configuration setup,
    /// writing it to `config_path`.
    pub fn new_from_base(
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
        number_cores: Option<NonZeroUsize>,
    ) -> Result<Self>
    where
        BaseConfig: Into<Config>,
    {
        let base = BaseConfig::new(config_path, database_path, number_cores)?;
        let config = base.into();
        Self::new(config)
    }

    /// Build a playlist of `playlist_length` items from a set of already analyzed
    /// songs in the library at `song_path`.
    ///
    /// It uses the ExentedIsolationForest score as a distance between songs, and deduplicates
    /// songs that are too close.
    ///
    /// Generating a playlist from a single song is also possible, and is just the special case
    /// where song_paths is a slice of length 1.
    pub fn playlist_from<T: Serialize + DeserializeOwned>(
        &self,
        song_paths: &[&str],
        playlist_length: usize,
    ) -> Result<Vec<LibrarySong<T>>> {
        self.playlist_from_custom(
            song_paths,
            playlist_length,
            &euclidean_distance,
            &mut closest_to_songs,
            true,
        )
    }

    /// Build a playlist of `playlist_length` items from a set of already analyzed
    /// song in the library at `initial_songs`, using distance metric `distance`,
    /// the sorting function `sort_by` and deduplicating if `dedup` is set to
    /// `true`.
    /// Note: The playlist includes the songs specified in `initial_songs`, so `playlist_length`
    /// has to be higher or equal than the length of `initial_songs`.
    ///
    /// You can use ready-to-use distance metrics such as
    /// [ExtendedIsolationForest](extended_isolation_forest::Forest), and ready-to-use
    /// sorting functions like [closest_to_songs].
    ///
    /// Generating a playlist from a single song is also possible, and is just the special case
    /// where song_paths is a slice of length 1.
    // TODO: making a playlist with the entire list of songs and then truncating is not
    // really the best approach - maybe sort_by should take a number of songs an Option?
    // Or maybe make `sort_by` return an iterator over songs? Something something a struct
    // with an internal state etc
    pub fn playlist_from_custom<
        T: Serialize + DeserializeOwned,
        F: FnMut(&[LibrarySong<T>], &mut [LibrarySong<T>], &dyn DistanceMetricBuilder),
    >(
        &self,
        initial_songs: &[&str],
        playlist_length: usize,
        distance: &dyn DistanceMetricBuilder,
        sort_by: &mut F,
        dedup: bool,
    ) -> Result<Vec<LibrarySong<T>>> {
        if playlist_length <= initial_songs.len() {
            bail!("A playlist length less than the initial playlist size was provided.")
        }
        let mut playlist: Vec<LibrarySong<T>> = initial_songs
            .iter()
            .map(|s| {
                self.song_from_path(s).map_err(|_| {
                    BlissError::ProviderError(format!("song '{s}' has not been analyzed"))
                })
            })
            .collect::<Result<Vec<_>, BlissError>>()?;
        // Remove the songs that are already in the playlist, so they don't get
        // sorted in the mess.
        let mut songs = self
            .songs_from_library()?
            .into_iter()
            .filter(|s| !initial_songs.contains(&&*s.bliss_song.path.to_string_lossy().to_string()))
            .collect::<Vec<_>>();
        sort_by(&playlist, &mut songs, distance);
        if dedup {
            dedup_playlist_custom_distance(&mut songs, None, distance);
        }
        songs.truncate(playlist_length - playlist.len());
        // We're reallocating a whole vector here, there must better ways to do what we want to
        // do.
        playlist.append(&mut songs);
        Ok(playlist)
    }

    /// Make a playlist of `number_albums` albums closest to the album
    /// with title `album_title`.
    /// The playlist starts with the album with `album_title`, and contains
    /// `number_albums` on top of that one.
    ///
    /// Returns the songs of each album ordered by bliss' `track_number`.
    pub fn album_playlist_from<T: Serialize + DeserializeOwned + Clone + PartialEq>(
        &self,
        album_title: String,
        number_albums: usize,
    ) -> Result<Vec<LibrarySong<T>>> {
        let album = self.songs_from_album(&album_title)?;
        // Every song should be from the same album. Hopefully...
        let songs = self.songs_from_library()?;
        let playlist = closest_album_to_group(album, songs)?;

        let mut album_count = 0;
        let mut index = 0;
        let mut current_album = Some(album_title);
        for song in playlist.iter() {
            if song.bliss_song.album != current_album {
                album_count += 1;
                if album_count > number_albums {
                    break;
                }
                song.bliss_song.album.clone_into(&mut current_album);
            }
            index += 1;
        }
        let playlist = &playlist[..index];
        Ok(playlist.to_vec())
    }

    /// Analyze and store all songs in `paths` that haven't been already analyzed.
    ///
    /// Use this function if you don't have any extra data to bundle with each song.
    ///
    /// Setting `delete_everything_else` to true will delete the paths that are
    /// not mentionned in `paths_extra_info` from the database. If you do not
    /// use it, because you only pass the new paths that need to be analyzed to
    /// this function, make sure to delete yourself from the database the songs
    /// that have been deleted from storage.
    ///
    /// If your library
    /// contains CUE files, pass the CUE file path only, and not individual
    /// CUE track names: passing `vec![file.cue]` will add
    /// individual tracks with the `cue_info` field set in the database.
    pub fn update_library<P: Into<PathBuf>>(
        &mut self,
        paths: Vec<P>,
        delete_everything_else: bool,
        show_progress_bar: bool,
    ) -> Result<()> {
        let paths_extra_info = paths.into_iter().map(|path| (path, ())).collect::<Vec<_>>();
        self.update_library_convert_extra_info(
            paths_extra_info,
            delete_everything_else,
            show_progress_bar,
            |x, _, _| x,
        )
    }

    /// Analyze and store all songs in `paths_extra_info` that haven't already
    /// been analyzed, along with some extra metadata serializable, and known
    /// before song analysis.
    ///
    /// Setting `delete_everything_else` to true will delete the paths that are
    /// not mentionned in `paths_extra_info` from the database. If you do not
    /// use it, because you only pass the new paths that need to be analyzed to
    /// this function, make sure to delete yourself from the database the songs
    /// that have been deleted from storage.
    pub fn update_library_extra_info<T: Serialize + DeserializeOwned, P: Into<PathBuf>>(
        &mut self,
        paths_extra_info: Vec<(P, T)>,
        delete_everything_else: bool,
        show_progress_bar: bool,
    ) -> Result<()> {
        self.update_library_convert_extra_info(
            paths_extra_info,
            delete_everything_else,
            show_progress_bar,
            |extra_info, _, _| extra_info,
        )
    }

    /// Analyze and store all songs in `paths_extra_info` that haven't
    /// been already analyzed, as well as handling extra, user-specified metadata,
    /// that can't directly be serializable,
    /// or that need input from the analyzed Song to be processed. If you
    /// just want to analyze and store songs along with some directly
    /// serializable values, consider using [Library::update_library_extra_info],
    /// or [Library::update_library] if you just want the analyzed songs
    /// stored as is.
    ///
    /// `paths_extra_info` is a tuple made out of song paths, along
    /// with any extra info you want to store for each song.
    /// If your library
    /// contains CUE files, pass the CUE file path only, and not individual
    /// CUE track names: passing `vec![file.cue]` will add
    /// individual tracks with the `cue_info` field set in the database.
    ///
    /// Setting `delete_everything_else` to true will delete the paths that are
    /// not mentionned in `paths_extra_info` from the database. If you do not
    /// use it, because you only pass the new paths that need to be analyzed to
    /// this function, make sure to delete yourself from the database the songs
    /// that have been deleted from storage.
    ///
    /// `convert_extra_info` is a function that you should specify how
    /// to convert that extra info to something serializable.
    pub fn update_library_convert_extra_info<
        T: Serialize + DeserializeOwned,
        U,
        P: Into<PathBuf>,
    >(
        &mut self,
        paths_extra_info: Vec<(P, U)>,
        delete_everything_else: bool,
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
                .query_map([FEATURES_VERSION], |row| {
                    Ok(row.get_unwrap::<usize, String>(0))
                })?
                .map(|x| PathBuf::from(x.unwrap()))
                .collect::<HashSet<PathBuf>>();
            return_value
        };

        let paths_extra_info: Vec<_> = paths_extra_info
            .into_iter()
            .map(|(x, y)| (x.into(), y))
            .collect();
        let paths: HashSet<_> = paths_extra_info.iter().map(|(p, _)| p.to_owned()).collect();

        if delete_everything_else {
            let paths_to_delete = existing_paths.difference(&paths);

            self.delete_paths(paths_to_delete)?;
        }

        // Can't use hashsets because we need the extra info here too,
        // and U might not be hashable.
        let paths_to_analyze = paths_extra_info
            .into_iter()
            .filter(|(path, _)| !existing_paths.contains(path))
            .collect::<Vec<(PathBuf, U)>>();

        self.analyze_paths_convert_extra_info(
            paths_to_analyze,
            show_progress_bar,
            convert_extra_info,
        )
    }

    /// Analyze and store all songs in `paths`.
    ///
    /// Updates the value of `features_version` in the config, using bliss'
    /// latest version.
    ///
    /// Use this function if you don't have any extra data to bundle with each song.
    ///
    /// If your library
    /// contains CUE files, pass the CUE file path only, and not individual
    /// CUE track names: passing `vec![file.cue]` will add
    /// individual tracks with the `cue_info` field set in the database.
    pub fn analyze_paths<P: Into<PathBuf>>(
        &mut self,
        paths: Vec<P>,
        show_progress_bar: bool,
    ) -> Result<()> {
        let paths_extra_info = paths.into_iter().map(|path| (path, ())).collect::<Vec<_>>();
        self.analyze_paths_convert_extra_info(paths_extra_info, show_progress_bar, |x, _, _| x)
    }

    /// Analyze and store all songs in `paths_extra_info`, along with some
    /// extra metadata serializable, and known before song analysis.
    ///
    /// Updates the value of `features_version` in the config, using bliss'
    /// latest version.
    /// If your library
    /// contains CUE files, pass the CUE file path only, and not individual
    /// CUE track names: passing `vec![file.cue]` will add
    /// individual tracks with the `cue_info` field set in the database.
    pub fn analyze_paths_extra_info<
        T: Serialize + DeserializeOwned + std::fmt::Debug,
        P: Into<PathBuf>,
    >(
        &mut self,
        paths_extra_info: Vec<(P, T)>,
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
    /// [Library::analyze_paths_extra_info], or [Library::analyze_paths] for
    /// the simpler use cases.
    ///
    /// Updates the value of `features_version` in the config, using bliss'
    /// latest version.
    ///
    /// `paths_extra_info` is a tuple made out of song paths, along
    /// with any extra info you want to store for each song. If your library
    /// contains CUE files, pass the CUE file path only, and not individual
    /// CUE track names: passing `vec![file.cue]` will add
    /// individual tracks with the `cue_info` field set in the database.
    ///
    /// `convert_extra_info` is a function that you should specify
    /// to convert that extra info to something serializable.
    pub fn analyze_paths_convert_extra_info<
        T: Serialize + DeserializeOwned,
        U,
        P: Into<PathBuf>,
    >(
        &mut self,
        paths_extra_info: Vec<(P, U)>,
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

        let mut paths_extra_info: HashMap<PathBuf, U> = paths_extra_info
            .into_iter()
            .map(|(x, y)| (x.into(), y))
            .collect();
        let mut cue_extra_info: HashMap<PathBuf, String> = HashMap::new();

        let results = D::analyze_paths_with_cores(
            paths_extra_info.keys(),
            self.config.base_config().number_cores,
        );
        let mut success_count = 0;
        let mut failure_count = 0;
        for (path, result) in results {
            if show_progress_bar {
                pb.set_message(format!("Analyzing {}", path.display()));
            }
            match result {
                Ok(song) => {
                    let is_cue = song.cue_info.is_some();
                    // If it's a song that's part of a CUE, its path will be
                    // something like `testcue.flac/CUE_TRACK001`, so we need
                    // to get the path of the main CUE file.
                    let path = {
                        if let Some(cue_info) = song.cue_info.to_owned() {
                            cue_info.cue_path
                        } else {
                            path
                        }
                    };
                    // Some magic to avoid having to depend on T: Clone, because
                    // all CUE tracks on a CUE file have the same extra_info.
                    // This serializes the data, store the serialized version
                    // in a hashmap, and then deserializes that when needed.
                    let extra = {
                        if is_cue && paths_extra_info.contains_key(&path) {
                            let extra = paths_extra_info.remove(&path).unwrap();
                            let e = convert_extra_info(extra, &song, self);
                            cue_extra_info.insert(
                                path,
                                serde_json::to_string(&e)
                                    .map_err(|e| BlissError::ProviderError(e.to_string()))?,
                            );
                            e
                        } else if is_cue {
                            let serialized_extra_info =
                                cue_extra_info.get(&path).unwrap().to_owned();
                            serde_json::from_str(&serialized_extra_info).unwrap()
                        } else {
                            let extra = paths_extra_info.remove(&path).unwrap();
                            convert_extra_info(extra, &song, self)
                        }
                    };
                    let library_song = LibrarySong::<T> {
                        bliss_song: song,
                        extra_info: extra,
                    };
                    self.store_song(&library_song)?;
                    success_count += 1;
                }
                Err(e) => {
                    log::error!(
                        "Analysis of song '{}' failed: {} The error has been stored.",
                        path.display(),
                        e
                    );

                    self.store_failed_song(path, e)?;
                    failure_count += 1;
                }
            };
            pb.inc(1);
        }
        pb.finish_with_message(format!(
            "Analyzed {success_count} song(s) successfully. {failure_count} Failure(s).",
        ));

        log::info!(
            "Analyzed {} song(s) successfully. {} Failure(s).",
            success_count,
            failure_count,
        );

        self.config.base_config_mut().features_version = FEATURES_VERSION;
        self.config.write()?;

        Ok(())
    }

    // Get songs from a songs / features statement.
    // BEWARE that the two songs and features query MUST be the same
    fn _songs_from_statement<T: Serialize + DeserializeOwned, P: Params + Clone>(
        &self,
        songs_statement: &str,
        features_statement: &str,
        params: P,
    ) -> Result<Vec<LibrarySong<T>>> {
        let connection = self
            .sqlite_conn
            .lock()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        let mut songs_statement = connection.prepare(songs_statement)?;
        let mut features_statement = connection.prepare(features_statement)?;
        let song_rows = songs_statement.query_map(params.to_owned(), |row| {
            Ok((row.get(12)?, Self::_song_from_row_closure(row)?))
        })?;
        let feature_rows =
            features_statement.query_map(params, |row| Ok((row.get(1)?, row.get(0)?)))?;

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

    /// Retrieve all songs which have been analyzed with
    /// current bliss version.
    ///
    /// Returns an error if one or several songs have a different number of
    /// features than they should, indicating the offending song id.
    ///
    // TODO maybe the error should make the song id / song path
    // accessible easily?
    pub fn songs_from_library<T: Serialize + DeserializeOwned>(
        &self,
    ) -> Result<Vec<LibrarySong<T>>> {
        let songs_statement = "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version, extra_info, cue_path,
                audio_file_path, id
                from song where analyzed = true and version = ? order by id
            ";
        let features_statement = "
            select
                feature, song.id from feature join song on song.id = feature.song_id
                where song.analyzed = true and song.version = ? order by song_id, feature_index
                ";
        let params = params![self.config.base_config().features_version];
        self._songs_from_statement(songs_statement, features_statement, params)
    }

    /// Get a LibrarySong from a given album title.
    ///
    /// This will return all songs with corresponding bliss "album" tag,
    /// and will order them by track number.
    pub fn songs_from_album<T: Serialize + DeserializeOwned>(
        &self,
        album_title: &str,
    ) -> Result<Vec<LibrarySong<T>>> {
        let params = params![album_title, self.config.base_config().features_version];
        let songs_statement = "
            select
                path, artist, title, album, album_artist,
                track_number, genre, duration, version, extra_info, cue_path,
                audio_file_path, id
                from song where album = ? and analyzed = true and version = ?
                order
                by cast(track_number as integer);
            ";

        // Get the song's analysis, and attach it to the existing song.
        let features_statement = "
            select
                feature, song.id from feature join song on song.id = feature.song_id
                where album=? and analyzed = true and version = ?
                order by cast(track_number as integer);
            ";
        let songs = self._songs_from_statement(songs_statement, features_statement, params)?;
        if songs.is_empty() {
            bail!(BlissError::ProviderError(String::from(
                "target album was not found in the database.",
            )));
        };
        Ok(songs)
    }

    /// Get a LibrarySong from a given file path.
    /// TODO pathbuf here too
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
                track_number, genre, duration, version, extra_info,
                cue_path, audio_file_path
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
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>()
                .try_into()
                .map_err(|_| {
                    BlissError::ProviderError(format!(
                        "song has more or less than {NUMBER_FEATURES} features",
                    ))
                })?,
        };
        song.bliss_song.analysis = analysis_vector;
        Ok(song)
    }

    fn _song_from_row_closure<T: Serialize + DeserializeOwned>(
        row: &Row,
    ) -> Result<LibrarySong<T>, RusqliteError> {
        let path: String = row.get(0)?;

        let cue_path: Option<String> = row.get(10)?;
        let audio_file_path: Option<String> = row.get(11)?;
        let mut cue_info = None;
        if let Some(cue_path) = cue_path {
            cue_info = Some(CueInfo {
                cue_path: PathBuf::from(cue_path),
                audio_file_path: PathBuf::from(audio_file_path.unwrap()),
            })
        };

        let song = Song {
            path: PathBuf::from(path),
            artist: row
                .get_ref(1)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            title: row
                .get_ref(2)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            album: row
                .get_ref(3)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            album_artist: row
                .get_ref(4)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            track_number: row
                .get_ref(5)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            genre: row
                .get_ref(6)
                .unwrap()
                .as_bytes_or_null()
                .unwrap()
                .map(|v| String::from_utf8_lossy(v).to_string()),
            analysis: Analysis {
                internal_analysis: [0.; NUMBER_FEATURES],
            },
            duration: Duration::from_secs_f64(row.get(7).unwrap()),
            features_version: row.get(8).unwrap(),
            cue_info,
        };

        let serialized: Option<String> = row.get(9).unwrap();
        let serialized = serialized.unwrap_or_else(|| "null".into());
        let extra_info = serde_json::from_str(&serialized).unwrap();
        Ok(LibrarySong {
            bliss_song: song,
            extra_info,
        })
    }

    /// Store a [Song] in the database, overidding any existing
    /// song with the same path by that one.
    // TODO to_str() returns an option; return early and avoid panicking
    pub fn store_song<T: Serialize + DeserializeOwned>(
        &mut self,
        library_song: &LibrarySong<T>,
    ) -> Result<(), BlissError> {
        let mut sqlite_conn = self.sqlite_conn.lock().unwrap();
        let tx = sqlite_conn
            .transaction()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        let song = &library_song.bliss_song;
        let (cue_path, audio_file_path) = match &song.cue_info {
            Some(c) => (
                Some(c.cue_path.to_string_lossy()),
                Some(c.audio_file_path.to_string_lossy()),
            ),
            None => (None, None),
        };
        tx.execute(
            "
            insert into song (
                path, artist, title, album, album_artist,
                duration, track_number, genre, analyzed, version, extra_info,
                cue_path, audio_file_path
            )
            values (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13
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
                extra_info=excluded.extra_info,
                cue_path=excluded.cue_path,
                audio_file_path=excluded.audio_file_path
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
                cue_path,
                audio_file_path,
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

    /// Store an errored [Song] in the SQLite database.
    ///
    /// If there already is an existing song with that path, replace it by
    /// the latest failed result.
    pub fn store_failed_song<P: Into<PathBuf>>(
        &mut self,
        song_path: P,
        e: BlissError,
    ) -> Result<()> {
        self.sqlite_conn
            .lock()
            .unwrap()
            .execute(
                "
            insert or replace into song (path, error) values (?1, ?2)
            ",
                [
                    song_path.into().to_string_lossy().to_string(),
                    e.to_string(),
                ],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        Ok(())
    }

    /// Delete a song with path `song_path` from the database.
    ///
    /// Errors out if the song is not in the database.
    pub fn delete_path<P: Into<PathBuf>>(&mut self, song_path: P) -> Result<()> {
        let song_path = song_path.into();
        let count = self
            .sqlite_conn
            .lock()
            .unwrap()
            .execute(
                "
                delete from song where path = ?1;
            ",
                [song_path.to_str()],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        if count == 0 {
            bail!(BlissError::ProviderError(format!(
                "tried to delete song {}, not existing in the database.",
                song_path.display(),
            )));
        }
        Ok(())
    }

    /// Delete a set of songs with paths `song_paths` from the database.
    ///
    /// Will return Ok(count) even if less songs than expected were deleted from the database.
    pub fn delete_paths<P: Into<PathBuf>, I: IntoIterator<Item = P>>(
        &mut self,
        paths: I,
    ) -> Result<usize> {
        let song_paths: Vec<String> = paths
            .into_iter()
            .map(|x| x.into().to_string_lossy().to_string())
            .collect();
        if song_paths.is_empty() {
            return Ok(0);
        };
        let count = self
            .sqlite_conn
            .lock()
            .unwrap()
            .execute(
                &format!(
                    "delete from song where path in ({})",
                    repeat_vars(song_paths.len()),
                ),
                params_from_iter(song_paths),
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        Ok(count)
    }
}

// Copied from
// https://docs.rs/rusqlite/latest/rusqlite/struct.ParamsFromIter.html#realistic-use-case
fn repeat_vars(count: usize) -> String {
    assert_ne!(count, 0);
    let mut s = "?,".repeat(count);
    // Remove trailing comma
    s.pop();
    s
}

#[cfg(test)]
fn data_local_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/local/directory"))
}

#[cfg(test)]
// TODO refactor (especially the helper functions)
// TODO the tests should really open a songs.db
// TODO test with invalid UTF-8
mod test {
    use super::*;
    use crate::{decoder::PreAnalyzedSong, Analysis, NUMBER_FEATURES};
    use ndarray::Array1;
    use pretty_assertions::assert_eq;
    use serde::{de::DeserializeOwned, Deserialize};
    use std::{convert::TryInto, fmt::Debug, sync::MutexGuard, time::Duration};
    use tempdir::TempDir;

    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::ffmpeg::FFmpeg as Decoder;
    use crate::song::decoder::Decoder as DecoderTrait;

    struct DummyDecoder;

    // Here to test an ffmpeg-agnostic library
    impl DecoderTrait for DummyDecoder {
        fn decode(_: &Path) -> crate::BlissResult<crate::decoder::PreAnalyzedSong> {
            Ok(PreAnalyzedSong::default())
        }
    }

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

        fn base_config_mut(&mut self) -> &mut BaseConfig {
            &mut self.base_config
        }
    }

    fn nzus(i: usize) -> NonZeroUsize {
        NonZeroUsize::new(i).unwrap()
    }

    // Returning the TempDir here, so it doesn't go out of scope, removing
    // the directory.
    //
    // Setup a test library made of 3 analyzed songs, with every field being different,
    // as well as an unanalyzed song and a song analyzed with a previous version.
    #[cfg(feature = "ffmpeg")]
    fn setup_test_library() -> (
        Library<BaseConfig, Decoder>,
        TempDir,
        (
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
            LibrarySong<ExtraInfo>,
        ),
    ) {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");
        let library = Library::<BaseConfig, Decoder>::new_from_base(
            Some(config_file),
            Some(database_file),
            None,
        )
        .unwrap();

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
            track_number: Some("03".into()),
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
            album: Some("An Album1001".into()),
            album_artist: Some("An Album Artist5001".into()),
            track_number: Some("01".into()),
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

        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 * 0.9)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        let song = Song {
            path: "/path/to/song6001".into(),
            artist: Some("Artist6001".into()),
            title: Some("Title6001".into()),
            album: Some("An Album2001".into()),
            album_artist: Some("An Album Artist6001".into()),
            track_number: Some("01".into()),
            genre: Some("Electronica6001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(710),
            features_version: 1,
            cue_info: None,
        };
        let fourth_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie6001"),
            },
        };

        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 * 50.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        let song = Song {
            path: "/path/to/song7001".into(),
            artist: Some("Artist7001".into()),
            title: Some("Title7001".into()),
            album: Some("An Album7001".into()),
            album_artist: Some("An Album Artist7001".into()),
            track_number: Some("01".into()),
            genre: Some("Electronica7001".into()),
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(810),
            features_version: 1,
            cue_info: None,
        };
        let fifth_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie7001"),
            },
        };

        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 * 100.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let song = Song {
            path: "/path/to/cuetrack.cue/CUE_TRACK001".into(),
            artist: Some("CUE Artist".into()),
            title: Some("CUE Title 01".into()),
            album: Some("CUE Album".into()),
            album_artist: Some("CUE Album Artist".into()),
            track_number: Some("01".into()),
            genre: None,
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(810),
            features_version: 1,
            cue_info: Some(CueInfo {
                cue_path: PathBuf::from("/path/to/cuetrack.cue"),
                audio_file_path: PathBuf::from("/path/to/cuetrack.flac"),
            }),
        };
        let sixth_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie7001"),
            },
        };

        let analysis_vector: [f32; NUMBER_FEATURES] = (0..NUMBER_FEATURES)
            .map(|x| x as f32 * 101.)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let song = Song {
            path: "/path/to/cuetrack.cue/CUE_TRACK002".into(),
            artist: Some("CUE Artist".into()),
            title: Some("CUE Title 02".into()),
            album: Some("CUE Album".into()),
            album_artist: Some("CUE Album Artist".into()),
            track_number: Some("02".into()),
            genre: None,
            analysis: Analysis {
                internal_analysis: analysis_vector,
            },
            duration: Duration::from_secs(910),
            features_version: 1,
            cue_info: Some(CueInfo {
                cue_path: PathBuf::from("/path/to/cuetrack.cue"),
                audio_file_path: PathBuf::from("/path/to/cuetrack.flac"),
            }),
        };
        let seventh_song = LibrarySong {
            bliss_song: song,
            extra_info: ExtraInfo {
                ignore: false,
                metadata_bliss_does_not_have: String::from("/path/to/charlie7001"),
            },
        };

        {
            let connection = library.sqlite_conn.lock().unwrap();
            connection
                .execute(
                    "
                    insert into song (
                        id, path, artist, title, album, album_artist, track_number,
                        genre, duration, analyzed, version, extra_info,
                        cue_path, audio_file_path
                    ) values (
                        1001, '/path/to/song1001', 'Artist1001', 'Title1001', 'An Album1001',
                        'An Album Artist1001', '03', 'Electronica1001', 310, true,
                        1, '{\"ignore\": true, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie1001\"}', null, null
                    ),
                    (
                        2001, '/path/to/song2001', 'Artist2001', 'Title2001', 'An Album2001',
                        'An Album Artist2001', '02', 'Electronica2001', 410, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie2001\"}', null, null
                    ),
                    (
                        3001, '/path/to/song3001', null, null, null,
                        null, null, null, null, false, 1, '{}', null, null
                    ),
                    (
                        4001, '/path/to/song4001', 'Artist4001', 'Title4001', 'An Album4001',
                        'An Album Artist4001', '01', 'Electronica4001', 510, true,
                        0, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie4001\"}', null, null
                    ),
                    (
                        5001, '/path/to/song5001', 'Artist5001', 'Title5001', 'An Album1001',
                        'An Album Artist5001', '01', 'Electronica5001', 610, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie5001\"}', null, null
                    ),
                    (
                        6001, '/path/to/song6001', 'Artist6001', 'Title6001', 'An Album2001',
                        'An Album Artist6001', '01', 'Electronica6001', 710, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie6001\"}', null, null
                    ),
                    (
                        7001, '/path/to/song7001', 'Artist7001', 'Title7001', 'An Album7001',
                        'An Album Artist7001', '01', 'Electronica7001', 810, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie7001\"}', null, null
                    ),
                    (
                        7002, '/path/to/cuetrack.cue/CUE_TRACK001', 'CUE Artist',
                        'CUE Title 01', 'CUE Album',
                        'CUE Album Artist', '01', null, 810, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie7001\"}', '/path/to/cuetrack.cue',
                        '/path/to/cuetrack.flac'
                    ),
                    (
                        7003, '/path/to/cuetrack.cue/CUE_TRACK002', 'CUE Artist',
                        'CUE Title 02', 'CUE Album',
                        'CUE Album Artist', '02', null, 910, true,
                        1, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie7001\"}', '/path/to/cuetrack.cue',
                        '/path/to/cuetrack.flac'
                    ),
                    (
                        8001, '/path/to/song8001', 'Artist8001', 'Title8001', 'An Album1001',
                        'An Album Artist8001', '03', 'Electronica8001', 910, true,
                        0, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie8001\"}', null, null
                    ),
                    (
                        9001, './data/s16_stereo_22_5kHz.flac', 'Artist9001', 'Title9001',
                        'An Album9001', 'An Album Artist8001', '03', 'Electronica8001',
                        1010, true, 0, '{\"ignore\": false, \"metadata_bliss_does_not_have\":
                        \"/path/to/charlie7001\"}', null, null
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
                            values
                                (1001, ?2, ?1),
                                (2001, ?3, ?1),
                                (3001, ?4, ?1),
                                (5001, ?5, ?1),
                                (6001, ?6, ?1),
                                (7001, ?7, ?1),
                                (7002, ?8, ?1),
                                (7003, ?9, ?1);
                            ",
                        params![
                            index,
                            index as f32 / 10.,
                            index as f32 + 10.,
                            index as f32 / 10. + 1.,
                            index as f32 / 2.,
                            index as f32 * 0.9,
                            index as f32 * 50.,
                            index as f32 * 100.,
                            index as f32 * 101.,
                        ],
                    )
                    .unwrap();
            }
            // Imaginary version 0 of bliss with less features.
            for index in 0..NUMBER_FEATURES - 5 {
                connection
                    .execute(
                        "
                            insert into feature(song_id, feature, feature_index)
                            values
                                (8001, ?2, ?1),
                                (9001, ?3, ?1);
                            ",
                        params![index, index as f32 / 20., index + 1],
                    )
                    .unwrap();
            }
        }
        (
            library,
            config_dir,
            (
                first_song,
                second_song,
                third_song,
                fourth_song,
                fifth_song,
                sixth_song,
                seventh_song,
            ),
        )
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
                track_number, genre, duration, version, extra_info,
                cue_path, audio_file_path
                from song where path=?
            ",
                params![song_path],
                |row| {
                    let path: String = row.get(0)?;
                    let cue_path: Option<String> = row.get(10)?;
                    let audio_file_path: Option<String> = row.get(11)?;
                    let mut cue_info = None;
                    if let Some(cue_path) = cue_path {
                        cue_info = Some(CueInfo {
                            cue_path: PathBuf::from(cue_path),
                            audio_file_path: PathBuf::from(audio_file_path.unwrap()),
                        })
                    };
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
                        cue_info,
                    };

                    let serialized: String = row.get(9).unwrap();
                    let extra_info = serde_json::from_str(&serialized).unwrap();
                    Ok(LibrarySong {
                        bliss_song: song,
                        extra_info,
                    })
                },
            )
            .expect("Song does not exist in the database");
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

    fn first_factor_divided_by_30_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        ((a[1] - b[1]).abs() / 30.).floor()
    }

    fn first_factor_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        (a[1] - b[1]).abs()
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_playlist_song_not_existing() {
        let (library, _temp_dir, _) = setup_test_library();
        assert!(library
            .playlist_from::<ExtraInfo>(&["not-existing"], 2)
            .is_err());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_playlist_crop() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> =
            library.playlist_from(&["/path/to/song2001"], 2).unwrap();
        assert_eq!(2, songs.len());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_simple_playlist() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> =
            library.playlist_from(&["/path/to/song2001"], 20).unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song6001",
                "/path/to/song5001",
                "/path/to/song1001",
                "/path/to/song7001",
                "/path/to/cuetrack.cue/CUE_TRACK001",
                "/path/to/cuetrack.cue/CUE_TRACK002",
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_playlist_length() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> =
            library.playlist_from(&["/path/to/song2001"], 3).unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song6001",
                "/path/to/song5001",
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_custom_playlist_distance() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> = library
            .playlist_from_custom(
                &["/path/to/song2001"],
                20,
                &first_factor_distance,
                &mut closest_to_songs,
                true,
            )
            .unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song6001",
                "/path/to/song5001",
                "/path/to/song1001",
                "/path/to/song7001",
                "/path/to/cuetrack.cue/CUE_TRACK001",
                "/path/to/cuetrack.cue/CUE_TRACK002",
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    fn custom_sort(
        _: &[LibrarySong<ExtraInfo>],
        songs: &mut [LibrarySong<ExtraInfo>],
        _distance: &dyn DistanceMetricBuilder,
    ) {
        songs.sort_by(|s1, s2| s1.bliss_song.path.cmp(&s2.bliss_song.path));
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_custom_playlist_sort() {
        let (library, _temp_dir, _) = setup_test_library();
        let songs: Vec<LibrarySong<ExtraInfo>> = library
            .playlist_from_custom(
                &["/path/to/song2001"],
                20,
                &euclidean_distance,
                &mut custom_sort,
                true,
            )
            .unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/cuetrack.cue/CUE_TRACK001",
                "/path/to/cuetrack.cue/CUE_TRACK002",
                "/path/to/song1001",
                "/path/to/song5001",
                "/path/to/song6001",
                "/path/to/song7001",
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_custom_playlist_dedup() {
        let (library, _temp_dir, _) = setup_test_library();

        let songs: Vec<LibrarySong<ExtraInfo>> = library
            .playlist_from_custom(
                &["/path/to/song2001"],
                20,
                &first_factor_divided_by_30_distance,
                &mut closest_to_songs,
                true,
            )
            .unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song1001",
                "/path/to/song7001",
                "/path/to/cuetrack.cue/CUE_TRACK001"
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        );

        let songs: Vec<LibrarySong<ExtraInfo>> = library
            .playlist_from_custom(
                &["/path/to/song2001"],
                20,
                &first_factor_distance,
                &mut closest_to_songs,
                false,
            )
            .unwrap();
        assert_eq!(
            vec![
                "/path/to/song2001",
                "/path/to/song6001",
                "/path/to/song5001",
                "/path/to/song1001",
                "/path/to/song7001",
                "/path/to/cuetrack.cue/CUE_TRACK001",
                "/path/to/cuetrack.cue/CUE_TRACK002",
            ],
            songs
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_album_playlist() {
        let (library, _temp_dir, _) = setup_test_library();
        let album: Vec<LibrarySong<ExtraInfo>> = library
            .album_playlist_from("An Album1001".to_string(), 20)
            .unwrap();
        assert_eq!(
            vec![
                // First album.
                "/path/to/song5001".to_string(),
                "/path/to/song1001".to_string(),
                // Second album, well ordered.
                "/path/to/song6001".to_string(),
                "/path/to/song2001".to_string(),
                // Third album.
                "/path/to/song7001".to_string(),
                // Fourth album.
                "/path/to/cuetrack.cue/CUE_TRACK001".to_string(),
                "/path/to/cuetrack.cue/CUE_TRACK002".to_string(),
            ],
            album
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_album_playlist_crop() {
        let (library, _temp_dir, _) = setup_test_library();
        let album: Vec<LibrarySong<ExtraInfo>> = library
            .album_playlist_from("An Album1001".to_string(), 1)
            .unwrap();
        assert_eq!(
            vec![
                // First album.
                "/path/to/song5001".to_string(),
                "/path/to/song1001".to_string(),
                // Second album, well ordered.
                "/path/to/song6001".to_string(),
                "/path/to/song2001".to_string(),
            ],
            album
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_songs_from_album() {
        let (library, _temp_dir, _) = setup_test_library();
        let album: Vec<LibrarySong<ExtraInfo>> = library.songs_from_album("An Album1001").unwrap();
        assert_eq!(
            vec![
                "/path/to/song5001".to_string(),
                "/path/to/song1001".to_string()
            ],
            album
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_songs_from_album_proper_features_version() {
        let (library, _temp_dir, _) = setup_test_library();
        let album: Vec<LibrarySong<ExtraInfo>> = library.songs_from_album("An Album1001").unwrap();
        assert_eq!(
            vec![
                "/path/to/song5001".to_string(),
                "/path/to/song1001".to_string()
            ],
            album
                .into_iter()
                .map(|s| s.bliss_song.path.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_songs_from_album_not_existing() {
        let (library, _temp_dir, _) = setup_test_library();
        assert!(library
            .songs_from_album::<ExtraInfo>("not-existing")
            .is_err());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_delete_path_non_existing() {
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
        assert!(library.delete_path("not-existing").is_err());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_delete_path() {
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

        library.delete_path("/path/to/song1001").unwrap();

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
    #[cfg(feature = "ffmpeg")]
    fn test_library_delete_paths() {
        let (mut library, _temp_dir, _) = setup_test_library();
        {
            let connection = library.sqlite_conn.lock().unwrap();
            let count: u32 = connection
                    .query_row(
                        "select count(*) from feature join song on song.id = feature.song_id where song.path in (?1, ?2)",
                        ["/path/to/song1001", "/path/to/song2001"],
                        |row| row.get(0),
                    )
                    .unwrap();
            assert!(count >= 1);
            let count: u32 = connection
                .query_row(
                    "select count(*) from song where path in (?1, ?2)",
                    ["/path/to/song1001", "/path/to/song2001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(count >= 1);
        }

        library
            .delete_paths(vec!["/path/to/song1001", "/path/to/song2001"])
            .unwrap();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            let count: u32 = connection
                .query_row(
                    "select count(*) from feature join song on song.id = feature.song_id where song.path in (?1, ?2)",
                    ["/path/to/song1001", "/path/to/song2001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(0, count);
            let count: u32 = connection
                .query_row(
                    "select count(*) from song where path in (?1, ?2)",
                    ["/path/to/song1001", "/path/to/song2001"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(0, count);
            // Make sure we did not delete everything else
            let count: u32 = connection
                .query_row("select count(*) from feature", [], |row| row.get(0))
                .unwrap();
            assert!(count >= 1);
            let count: u32 = connection
                .query_row("select count(*) from song", [], |row| row.get(0))
                .unwrap();
            assert!(count >= 1);
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_delete_paths_empty() {
        let (mut library, _temp_dir, _) = setup_test_library();
        assert_eq!(library.delete_paths::<String, _>([]).unwrap(), 0);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_delete_paths_non_existing() {
        let (mut library, _temp_dir, _) = setup_test_library();
        assert_eq!(library.delete_paths(["not-existing"]).unwrap(), 0);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze_paths_cue() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;
        {
            let sqlite_conn =
                Connection::open(&library.config.base_config().database_path).unwrap();
            sqlite_conn.execute("delete from song", []).unwrap();
        }

        let paths = vec![
            "./data/s16_mono_22_5kHz.flac",
            "./data/testcue.cue",
            "non-existing",
        ];
        library.analyze_paths(paths.to_owned(), false).unwrap();
        let expected_analyzed_paths = vec![
            "./data/s16_mono_22_5kHz.flac",
            "./data/testcue.cue/CUE_TRACK001",
            "./data/testcue.cue/CUE_TRACK002",
            "./data/testcue.cue/CUE_TRACK003",
        ];
        {
            let connection = library.sqlite_conn.lock().unwrap();
            let mut stmt = connection
                .prepare(
                    "
                select
                    path from song where analyzed = true and path not like '%song%'
                    order by path
                ",
                )
                .unwrap();
            let paths = stmt
                .query_map(params![], |row| row.get(0))
                .unwrap()
                .map(|x| x.unwrap())
                .collect::<Vec<String>>();

            assert_eq!(paths, expected_analyzed_paths);
        }
        {
            let connection = library.sqlite_conn.lock().unwrap();
            let song: LibrarySong<()> =
                _library_song_from_database(connection, "./data/testcue.cue/CUE_TRACK001");
            assert!(song.bliss_song.cue_info.is_some());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze_paths() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        let paths = vec![
            "./data/s16_mono_22_5kHz.flac",
            "./data/s16_stereo_22_5kHz.flac",
            "non-existing",
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<()>>>();
        assert_eq!(songs, expected_songs);
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze_paths_convert_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;
        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac", true),
            ("./data/s16_stereo_22_5kHz.flac", false),
            ("non-existing", false),
        ];
        library
            .analyze_paths_convert_extra_info(paths.to_owned(), true, |b, _, _| ExtraInfo {
                ignore: b,
                metadata_bliss_does_not_have: String::from("coucou"),
            })
            .unwrap();
        library
            .analyze_paths_convert_extra_info(paths.to_owned(), false, |b, _, _| ExtraInfo {
                ignore: b,
                metadata_bliss_does_not_have: String::from("coucou"),
            })
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_analyze_paths_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();

        let paths = vec![
            (
                "./data/s16_mono_22_5kHz.flac",
                ExtraInfo {
                    ignore: true,
                    metadata_bliss_does_not_have: String::from("hey"),
                },
            ),
            (
                "./data/s16_stereo_22_5kHz.flac",
                ExtraInfo {
                    ignore: false,
                    metadata_bliss_does_not_have: String::from("hello"),
                },
            ),
            (
                "non-existing",
                ExtraInfo {
                    ignore: true,
                    metadata_bliss_does_not_have: String::from("coucou"),
                },
            ),
        ];
        library
            .analyze_paths_extra_info(paths.to_owned(), false)
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    // Check that a song already in the database is not
    // analyzed again on updates.
    fn test_update_skip_analyzed() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        for input in vec![
            ("./data/s16_mono_22_5kHz.flac", true),
            ("./data/s16_mono_22_5kHz.flac", false),
        ]
        .into_iter()
        {
            let paths = vec![input.to_owned()];
            library
                .update_library_convert_extra_info(paths.to_owned(), true, false, |b, _, _| {
                    ExtraInfo {
                        ignore: b,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    }
                })
                .unwrap();
            let song = {
                let connection = library.sqlite_conn.lock().unwrap();
                _library_song_from_database::<ExtraInfo>(connection, "./data/s16_mono_22_5kHz.flac")
            };
            let expected_song = {
                LibrarySong {
                    bliss_song: Decoder::song_from_path("./data/s16_mono_22_5kHz.flac").unwrap(),
                    extra_info: ExtraInfo {
                        ignore: true,
                        metadata_bliss_does_not_have: String::from("coucou"),
                    },
                }
            };
            assert_eq!(song, expected_song);
            assert_eq!(
                library.config.base_config_mut().features_version,
                FEATURES_VERSION
            );
        }
    }

    fn _get_song_analyzed(
        connection: MutexGuard<Connection>,
        path: String,
    ) -> Result<bool, RusqliteError> {
        let mut stmt = connection.prepare(
            "
                select
                    analyzed from song
                    where song.path = ?
                ",
        )?;
        stmt.query_row([path], |row| (row.get(0)))
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_update_library_override_old_features() {
        let (mut library, _temp_dir, _) = setup_test_library();
        let path: String = "./data/s16_stereo_22_5kHz.flac".into();

        {
            let connection = library.sqlite_conn.lock().unwrap();
            let mut stmt = connection
                .prepare(
                    "
                select
                    feature from feature join song on song.id = feature.song_id
                    where song.path = ? order by feature_index
                ",
                )
                .unwrap();
            let analysis_vector = stmt
                .query_map(params![path], |row| row.get(0))
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>();
            assert_eq!(
                analysis_vector,
                vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
            )
        }

        library
            .update_library(vec![path.to_owned()], true, false)
            .unwrap();

        let connection = library.sqlite_conn.lock().unwrap();
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
                .query_map(params![path], |row| row.get(0))
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        };
        let expected_analysis_vector = Decoder::song_from_path(path).unwrap().analysis;
        assert_eq!(analysis_vector, expected_analysis_vector);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_update_library() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }

        let paths = vec![
            "./data/s16_mono_22_5kHz.flac",
            "./data/s16_stereo_22_5kHz.flac",
            "/path/to/song4001",
            "non-existing",
        ];
        library
            .update_library(paths.to_owned(), true, false)
            .unwrap();
        library
            .update_library(paths.to_owned(), true, true)
            .unwrap();

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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<()>>>();

        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_update_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac", true),
            ("./data/s16_stereo_22_5kHz.flac", false),
            ("/path/to/song4001", false),
            ("non-existing", false),
        ];
        library
            .update_library_extra_info(paths.to_owned(), true, false)
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<bool>>>();
        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_update_convert_extra_info() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that all the starting songs are there
            assert!(_get_song_analyzed(connection, "/path/to/song2001".into()).unwrap());
        }

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac", true),
            ("./data/s16_stereo_22_5kHz.flac", false),
            ("/path/to/song4001", false),
            ("non-existing", false),
        ];
        library
            .update_library_convert_extra_info(paths.to_owned(), true, false, |b, _, _| ExtraInfo {
                ignore: b,
                metadata_bliss_does_not_have: String::from("coucou"),
            })
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we deleted older songs
            assert_eq!(
                rusqlite::Error::QueryReturnedNoRows,
                _get_song_analyzed(connection, "/path/to/song2001".into()).unwrap_err(),
            );
        }
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    // TODO maybe we can merge / DRY this and the function ⬆
    fn test_update_convert_extra_info_do_not_delete() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library.config.base_config_mut().features_version = 0;

        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that all the starting songs are there
            assert!(_get_song_analyzed(connection, "/path/to/song2001".into()).unwrap());
        }

        let paths = vec![
            ("./data/s16_mono_22_5kHz.flac", true),
            ("./data/s16_stereo_22_5kHz.flac", false),
            ("/path/to/song4001", false),
            ("non-existing", false),
        ];
        library
            .update_library_convert_extra_info(paths.to_owned(), false, false, |b, _, _| {
                ExtraInfo {
                    ignore: b,
                    metadata_bliss_does_not_have: String::from("coucou"),
                }
            })
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
                bliss_song: Decoder::song_from_path(path).unwrap(),
                extra_info: expected_extra_info,
            })
            .collect::<Vec<LibrarySong<ExtraInfo>>>();
        assert_eq!(songs, expected_songs);
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we tried to "update" song4001 with the new features.
            assert!(!_get_song_analyzed(connection, "/path/to/song4001".into()).unwrap());
        }
        {
            let connection = library.sqlite_conn.lock().unwrap();
            // Make sure that we did not delete older songs
            assert!(_get_song_analyzed(connection, "/path/to/song2001".into()).unwrap());
        }
        assert_eq!(
            library.config.base_config_mut().features_version,
            FEATURES_VERSION
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
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
    #[cfg(feature = "ffmpeg")]
    fn test_store_failed_song() {
        let (mut library, _temp_dir, _) = setup_test_library();
        library
            .store_failed_song(
                "/some/failed/path",
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
    #[cfg(feature = "ffmpeg")]
    fn test_songs_from_library() {
        let (library, _temp_dir, expected_library_songs) = setup_test_library();

        let library_songs = library.songs_from_library::<ExtraInfo>().unwrap();
        assert_eq!(library_songs.len(), 7);
        assert_eq!(
            expected_library_songs,
            (
                library_songs[0].to_owned(),
                library_songs[1].to_owned(),
                library_songs[2].to_owned(),
                library_songs[3].to_owned(),
                library_songs[4].to_owned(),
                library_songs[5].to_owned(),
                library_songs[6].to_owned(),
            )
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
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
    #[cfg(feature = "ffmpeg")]
    fn test_song_from_path_not_analyzed() {
        let (library, _temp_dir, _) = setup_test_library();
        let error = library.song_from_path::<ExtraInfo>("/path/to/song4001");
        assert!(error.is_err());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
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
    #[cfg(feature = "ffmpeg")]
    fn test_library_new_default_write() {
        let (library, _temp_dir, _) = setup_test_library();
        let config_content = fs::read_to_string(&library.config.base_config().config_path)
            .unwrap()
            .replace(' ', "")
            .replace('\n', "");
        assert_eq!(
            config_content,
            format!(
                "{{\"config_path\":\"{}\",\"database_path\":\"{}\",\"features_version\":{},\"number_cores\":{}}}",
                library.config.base_config().config_path.display(),
                library.config.base_config().database_path.display(),
                FEATURES_VERSION,
                thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap()),
            )
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
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
    #[cfg(feature = "ffmpeg")]
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
    #[cfg(feature = "ffmpeg")]
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
            Library::<CustomConfig, DummyDecoder>::from_config_path(Some(PathBuf::from(
                "non-existing"
            )))
            .is_err()
        );
    }

    #[test]
    fn test_from_config_path() {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");

        // In reality, someone would just do that with `(None, None)` to get the default
        // paths.
        let base_config = BaseConfig::new(
            Some(config_file.to_owned()),
            Some(database_file),
            Some(nzus(1)),
        )
        .unwrap();

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
            let mut library = Library::<_, DummyDecoder>::new(config.to_owned()).unwrap();
            library.store_song(&song).unwrap();
        }

        let library: Library<CustomConfig, DummyDecoder> =
            Library::from_config_path(Some(config_file)).unwrap();
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
        let base_config = BaseConfig::new(
            Some(config_file.to_owned()),
            Some(database_file),
            Some(nzus(1)),
        )
        .unwrap();

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

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_sanity_check_fail() {
        let (mut library, _temp_dir, _) = setup_test_library();
        assert!(!library.version_sanity_check().unwrap());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_library_sanity_check_ok() {
        let (mut library, _temp_dir, _) = setup_test_library();
        {
            let sqlite_conn =
                Connection::open(&library.config.base_config().database_path).unwrap();
            sqlite_conn
                .execute("delete from song where version != 1", [])
                .unwrap();
        }
        assert!(library.version_sanity_check().unwrap());
    }

    #[test]
    fn test_config_number_cpus() {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");

        let base_config = BaseConfig::new(
            Some(config_file.to_owned()),
            Some(database_file.to_owned()),
            None,
        )
        .unwrap();
        let config = CustomConfig {
            base_config,
            second_path_to_music_library: "/path/to/somewhere".into(),
            ignore_wav_files: true,
        };

        assert_eq!(
            config.get_number_cores().get(),
            usize::from(thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap())),
        );

        let base_config =
            BaseConfig::new(Some(config_file), Some(database_file), Some(nzus(1))).unwrap();
        let mut config = CustomConfig {
            base_config,
            second_path_to_music_library: "/path/to/somewhere".into(),
            ignore_wav_files: true,
        };

        assert_eq!(config.get_number_cores().get(), 1);
        config.set_number_cores(nzus(2)).unwrap();
        assert_eq!(config.get_number_cores().get(), 2);
    }

    #[test]
    fn test_library_create_all_dirs() {
        let config_dir = TempDir::new("coucou")
            .unwrap()
            .path()
            .join("path")
            .join("to");
        assert!(!config_dir.is_dir());
        let config_file = config_dir.join("config.json");
        let database_file = config_dir.join("bliss.db");
        Library::<BaseConfig, DummyDecoder>::new_from_base(
            Some(config_file),
            Some(database_file),
            Some(nzus(1)),
        )
        .unwrap();
        assert!(config_dir.is_dir());
    }
}
