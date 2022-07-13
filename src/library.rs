//! Module containing utilities to manage a SQLite library of [Song]s.
use anyhow::{Context, Result};
#[cfg(not(test))]
use dirs::data_local_dir;
use rusqlite::params;
use rusqlite::Connection;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use crate::BlissError;
use crate::Song;

/// Configuration trait, used for instance to customize
/// the format in which the configuration file should be written.
pub trait AppConfigTrait: Serialize + Sized {
    // Implementers have to provide these.
    /// This trait should return the [BaseConfig] from the parent,
    /// user-created `Config`.
    fn base_config(&self) -> &BaseConfig;

    // Default implementation to output the config as a JSON file.
    /// Convert the current config to a [String], to be written to
    /// a file.
    ///
    /// The default writes a JSON file, but any format can be used,
    /// using for example the various Serde libraries (`serde_yaml`, etc).
    fn serialize_config(&self) -> Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    // This default impl is what requires the `Serialize` supertrait
    /// Write the configuration to a file using
    /// [AppConfigTrait::serialize_config]. This method can be overriden
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

#[derive(Serialize)]
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
pub struct LibrarySong<T: Serialize> {
    bliss_song: Song,
    extra_info: T,
}

impl<Config: ConfigTrait> Library<Config> {
    /// Create a new [Library] object from the given [Config] struct,
    /// writing the configuration to the file given in
    /// `config.config_path`.
    ///
    /// This function should only be called once, when a user wishes to
    /// create a completely new "library".
    /// Otherwise, load an existing library file using [Library::from_config].
    // TODO add serializable extra json info
    pub fn new(config: Config) -> Result<Self> {
        let sqlite_conn = Connection::open(&config.base_config().database_path)?;
        sqlite_conn.execute(
            "
            create table if not exists song (
                id integer primary key,
                path text not null unique,
                duration integer not null,
                album_artist text,
                artist text,
                title text,
                album text,
                track_number text,
                genre text,
                stamp timestamp default current_timestamp,
                version integer not null default 1,
                analyzed boolean default false,
                extra_info json
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
                unique(id, feature_index),
                foreign key(song_id) references song(id)
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

    /// Store a [Song] in the database, overidding any existing
    /// song with the same path by that one.
    pub fn store_song<T: Serialize>(
        &mut self,
        library_song: &LibrarySong<T>,
    ) -> Result<(), BlissError> {
        let mut sqlite_conn = self.sqlite_conn.lock().unwrap();
        let song = &library_song.bliss_song;
        sqlite_conn
            .execute(
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
                version=excluded.version
            ",
                params![
                    song.path.to_str(),
                    song.artist,
                    song.title,
                    song.album,
                    song.album_artist,
                    song.duration.as_secs(),
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
        sqlite_conn
            .execute(
                "delete from feature where song_id in (select id from song where path = ?1);",
                params![song.path.to_str()],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;

        let tx = sqlite_conn
            .transaction()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        for (index, feature) in song.analysis.as_vec().iter().enumerate() {
            tx.execute(
                "
                insert into feature (song_id, feature, feature_index)
                values ((select id from song where path = ?1), ?2, ?3)
                ",
                params![song.path.to_str(), feature, index],
            )
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| BlissError::ProviderError(e.to_string()))?;
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
    use std::fs;
    use tempdir::TempDir;

    // Returning the TempDir here, so it doesn't go out of scope, removing
    // the directory.
    fn setup_test_library() -> (Library<BaseConfig>, TempDir) {
        let config_dir = TempDir::new("coucou").unwrap();
        let config_file = config_dir.path().join("config.json");
        let database_file = config_dir.path().join("bliss.db");
        (
            Library::<BaseConfig>::new_from_base(Some(config_file), Some(database_file)).unwrap(),
            config_dir,
        )
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
        let (library, _temp_dir) = setup_test_library();
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
        let (library, _temp_dir) = setup_test_library();
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
            values (1, 1, 1.1, 1);
            ",
                [],
            )
            .unwrap();
    }

    #[test]
    fn test_library_extra_info() {
        #[derive(Serialize)]
        struct ExtraInfo {
            ignore: bool,
        }
        let (mut library, _temp_dir) = setup_test_library();
        let song = Song {
            path: "/path/to/song".into(),
            ..Default::default()
        };
        let extra_info = ExtraInfo { ignore: true };
        let song = LibrarySong {
            bliss_song: song,
            extra_info,
        };
        library.store_song(&song).unwrap();
    }
}
