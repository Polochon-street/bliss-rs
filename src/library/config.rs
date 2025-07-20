//! Submodule containing everything pertaining to the configuration of
//! Libraries. The configuration typically holds the path to the database,
//! as well as other information related to the library providers
//! want to store.
//!
//! This extra information will be automatically stored / retrieved when you instanciate a
//! [Library] (the core of your plugin).
//!
//! In practice, this entails specifying a configuration struct, that will
//! implement the [AppConfigTrait], i.e. implement `Serialize`, `Deserialize`, and a
//! function to retrieve the [BaseConfig] (which is just a structure
//! holding the path to the configuration file and the path to the database).
//!
//! The most straightforward way to do so is to have something like this
//! (in this example, we assume that `path_to_extra_information` is something
//! you would want stored in your configuration file, path to a second music
//! folder for instance):
//! ```
//!   use anyhow::Result;
//!   use serde::{Deserialize, Serialize};
//!   use std::path::PathBuf;
//!   use std::num::NonZeroUsize;
//!   use bliss_audio::BlissError;
//!   use bliss_audio::library::config::{AppConfigTrait, BaseConfig};
//!
//!   // The actual Config struct holding all the information
//!   // that you want stored in your configuration file.
//!   //
//!   // The `base_config` field is a struct containing basic configuration
//!   // information needed for the Library to function, i.e. the path to the
//!   // configuration file itself, the database path, etc.
//!   #[derive(Serialize, Deserialize, Clone, Debug)]
//!   pub struct Config {
//!       #[serde(flatten)]
//!       pub base_config: BaseConfig,
//!       pub music_library_path: PathBuf,
//!   }
//!
//!   impl AppConfigTrait for Config {
//!       fn base_config(&self) -> &BaseConfig {
//!           &self.base_config
//!       }
//!
//!       fn base_config_mut(&mut self) -> &mut BaseConfig {
//!           &mut self.base_config
//!       }
//!   }
//!
//!   impl Config {
//!       pub fn new(
//!           music_library_path: PathBuf,
//!           config_path: Option<PathBuf>,
//!           database_path: Option<PathBuf>,
//!           number_cores: Option<NonZeroUsize>,
//!       ) -> Result<Self> {
//!           // Note that by passing `(None, None)` here, the paths will
//!           // be inferred automatically using user data dirs.
//!           let base_config = BaseConfig::new(config_path, database_path, number_cores)?;
//!           Ok(Self {
//!               base_config,
//!               music_library_path,
//!           })
//!       }
//!   }
//! ```

use anyhow::{Context, Result};
#[cfg(all(not(test), not(feature = "integration-tests")))]
use dirs::config_local_dir;
#[cfg(all(not(test), not(feature = "integration-tests")))]
use dirs::data_local_dir;
use ndarray::Array2;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;

use crate::FEATURES_VERSION;
use crate::{BlissError, NUMBER_FEATURES};

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

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
/// The minimum configuration an application needs to work with
/// a [Library].
pub struct BaseConfig {
    /// The path to where the configuration file should be stored,
    /// e.g. `/home/foo/.local/share/bliss-rs/config.json`
    pub config_path: PathBuf,
    /// The path to where the database file should be stored,
    /// e.g. `/home/foo/.local/share/bliss-rs/bliss.db`
    pub database_path: PathBuf,
    /// The latest features' version a song has been analyzed
    /// with.
    pub features_version: u16,
    /// The number of CPU cores an analysis will be performed with.
    /// Defaults to the number of CPUs in the user's computer.
    pub number_cores: NonZeroUsize,
    /// The mahalanobis matrix used for mahalanobis distance.
    /// Used to customize the distance metric beyond simple euclidean distance.
    /// Uses ndarray's `serde` feature for serialization / deserialization.
    /// Field would look like this:
    /// "m": {"v": 1, "dim": [20, 20], "data": [1.0, 0.0, ..., 1.0]}
    #[serde(default = "default_m")]
    pub m: Array2<f32>,
}

fn default_m() -> Array2<f32> {
    Array2::eye(NUMBER_FEATURES)
}

impl BaseConfig {
    /// Because we spent some time using XDG_DATA_HOME instead of XDG_CONFIG_HOME
    /// as the default folder, we have to jump through some hoops:
    ///
    /// - Legacy path exists, new path doesn't exist => legacy path should be returned
    /// - Legacy path exists, new path exists => new path should be returned
    /// - Legacy path doesn't exist => new path should be returned
    pub(crate) fn get_default_data_folder() -> Result<PathBuf> {
        let error_message = "No suitable path found to store bliss' song database. Consider specifying such a path.";
        let default_folder = env::var("XDG_CONFIG_HOME")
            .map(|path| Path::new(&path).join("bliss-rs"))
            .or_else(|_| {
                config_local_dir()
                    .map(|p| p.join("bliss-rs"))
                    .with_context(|| error_message)
            });

        if let Ok(folder) = &default_folder {
            if folder.exists() {
                return Ok(folder.clone());
            }
        }

        if let Ok(legacy_folder) = BaseConfig::get_legacy_data_folder() {
            if legacy_folder.exists() {
                return Ok(legacy_folder);
            }
        }

        // If neither default_folder nor legacy_folder exist, return the default folder
        default_folder
    }

    fn get_legacy_data_folder() -> Result<PathBuf> {
        let path = match env::var("XDG_DATA_HOME") {
            Ok(path) => Path::new(&path).join("bliss-rs"),
            Err(_) => data_local_dir().with_context(|| "No suitable path found to store bliss' song database. Consider specifying such a path.")?.join("bliss-rs"),
        };
        Ok(path)
    }

    /// Create a new, basic config. Upon calls of `Config.write()`, it will be
    /// written to `config_path`.
    //
    /// Any path omitted will instead default to a "clever" path using
    /// data directory inference. The "clever" thinking is as follows:
    /// - If the user specified only one of the paths, it will put the other
    ///   file in the same folder as the given path.
    /// - If the user specified both paths, it will go with what the user
    ///   chose.
    /// - If the user didn't select any path, it will try to put everything in
    ///   the user's configuration directory, i.e. XDG_CONFIG_HOME.
    ///
    /// The number of cores is the number of cores that should be used for
    /// any analysis. If not provided, it will default to the computer's
    /// number of cores.
    pub fn new(
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
        number_cores: Option<NonZeroUsize>,
    ) -> Result<Self> {
        let provided_database_path = database_path.is_some();
        let provided_config_path = config_path.is_some();
        let mut final_config_path = {
            // User provided a path; let the future file creation determine
            // whether it points to something valid or not
            if let Some(path) = config_path {
                path
            } else {
                Self::get_default_data_folder()?.join(Path::new("config.json"))
            }
        };

        let mut final_database_path = {
            if let Some(path) = database_path {
                path
            } else {
                Self::get_default_data_folder()?.join(Path::new("songs.db"))
            }
        };

        if provided_database_path && !provided_config_path {
            final_config_path = final_database_path
                .parent()
                .ok_or(BlissError::ProviderError(String::from(
                    "provided database path was invalid.",
                )))?
                .join(Path::new("config.json"))
        } else if !provided_database_path && provided_config_path {
            final_database_path = final_config_path
                .parent()
                .ok_or(BlissError::ProviderError(String::from(
                    "provided config path was invalid.",
                )))?
                .join(Path::new("songs.db"))
        }

        let number_cores = number_cores.unwrap_or_else(|| {
            thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap())
        });

        Ok(Self {
            config_path: final_config_path,
            database_path: final_database_path,
            features_version: FEATURES_VERSION,
            number_cores,
            m: Array2::eye(NUMBER_FEATURES),
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

#[cfg(any(test, feature = "integration-tests"))]
fn data_local_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/tmp/data"))
}

#[cfg(any(test, feature = "integration-tests"))]
fn config_local_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/tmp/"))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::NUMBER_FEATURES;
    use pretty_assertions::assert_eq;
    use serde::Deserialize;
    use std::fs::create_dir_all;
    use std::{fmt::Debug, str::FromStr};
    use tempdir::TempDir;

    #[derive(Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
    struct ExtraInfo {
        ignore: bool,
        metadata_bliss_does_not_have: String,
    }

    #[derive(Deserialize, Serialize, PartialEq, Debug, Clone)]
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

    #[test]
    fn test_get_default_data_folder_no_default_path() {
        // Cases to test:
        // - Legacy path exists, new path doesn't exist => legacy path should be returned
        // - Legacy path exists, new path exists => new path should be returned
        // - Legacy path doesn't exist => new path should be returned

        // Nothing exists: XDG_CONFIG_HOME takes precedence
        env::set_var("XDG_CONFIG_HOME", "/home/foo/.config");
        env::set_var("XDG_DATA_HOME", "/home/foo/.local/share");
        assert_eq!(
            PathBuf::from("/home/foo/.config/bliss-rs"),
            BaseConfig::get_default_data_folder().unwrap()
        );
        env::remove_var("XDG_CONFIG_HOME");
        env::remove_var("XDG_DATA_HOME");

        // Legacy folder exists, new folder does not exist, it takes precedence
        let existing_legacy_folder = TempDir::new("tmp").unwrap();
        create_dir_all(existing_legacy_folder.path().join("bliss-rs")).unwrap();
        env::set_var("XDG_CONFIG_HOME", "/home/foo/.config");
        env::set_var("XDG_DATA_HOME", existing_legacy_folder.path().as_os_str());
        assert_eq!(
            existing_legacy_folder.path().join("bliss-rs"),
            BaseConfig::get_default_data_folder().unwrap()
        );

        // Both exists, new folder takes precedence
        let existing_folder = TempDir::new("tmp").unwrap();
        create_dir_all(existing_folder.path().join("bliss-rs")).unwrap();
        env::set_var("XDG_CONFIG_HOME", existing_folder.path().as_os_str());
        assert_eq!(
            existing_folder.path().join("bliss-rs"),
            BaseConfig::get_default_data_folder().unwrap()
        );

        env::remove_var("XDG_DATA_HOME");
        env::remove_var("XDG_CONFIG_HOME");

        assert_eq!(
            PathBuf::from("/tmp/bliss-rs/"),
            BaseConfig::get_default_data_folder().unwrap()
        );
    }

    #[test]
    fn test_base_config_new() {
        {
            let xdg_config_home = TempDir::new("test-bliss").unwrap();
            env::set_var("XDG_CONFIG_HOME", xdg_config_home.path());

            // First test case: default options go to the XDG_CONFIG_HOME path.
            let base_config = BaseConfig::new(None, None, None).unwrap();

            assert_eq!(
                base_config.config_path,
                xdg_config_home.path().join("bliss-rs/config.json"),
            );
            assert_eq!(
                base_config.database_path,
                xdg_config_home.path().join("bliss-rs/songs.db"),
            );
        }

        // Second test case: config path, no db path.
        {
            let random_config_home = TempDir::new("config").unwrap();
            let base_config = BaseConfig::new(
                Some(random_config_home.path().join("test.json")),
                None,
                None,
            )
            .unwrap();

            assert_eq!(
                base_config.config_path,
                random_config_home.path().join("test.json"),
            );
            assert_eq!(
                base_config.database_path,
                random_config_home.path().join("songs.db")
            );
        }

        // Third test case: no config path, but db path.
        {
            let random_config_home = TempDir::new("database").unwrap();
            let base_config =
                BaseConfig::new(None, Some(random_config_home.path().join("test.db")), None)
                    .unwrap();

            assert_eq!(
                base_config.config_path,
                random_config_home.path().join("config.json"),
            );
            assert_eq!(
                base_config.database_path,
                random_config_home.path().join("test.db"),
            );
        }
        // Last test case: both paths specified.
        {
            let random_config_home = TempDir::new("config").unwrap();
            let random_database_home = TempDir::new("database").unwrap();
            let base_config = BaseConfig::new(
                Some(random_config_home.path().join("config_test.json")),
                Some(random_database_home.path().join("test-database.db")),
                None,
            )
            .unwrap();

            assert_eq!(
                base_config.config_path,
                random_config_home.path().join("config_test.json"),
            );
            assert_eq!(
                base_config.database_path,
                random_database_home.path().join("test-database.db"),
            );
        }
    }

    #[test]
    fn test_config_from_file() {
        let config = BaseConfig::from_path("./data/sample-config.json").unwrap();
        let mut m: Array2<f32> = Array2::eye(NUMBER_FEATURES);
        m[[0, 1]] = 1.;
        assert_eq!(
            config,
            BaseConfig {
                config_path: PathBuf::from_str("/tmp/bliss-rs/config.json").unwrap(),
                database_path: PathBuf::from_str("/tmp/bliss-rs/songs.db").unwrap(),
                features_version: 1,
                number_cores: NonZeroUsize::new(8).unwrap(),
                m,
            }
        );
    }

    #[test]
    fn test_config_old_existing() {
        let config = BaseConfig::from_path("./data/old_config.json").unwrap();
        assert_eq!(
            config,
            BaseConfig {
                config_path: PathBuf::from_str("/tmp/bliss-rs/config.json").unwrap(),
                database_path: PathBuf::from_str("/tmp/bliss-rs/songs.db").unwrap(),
                features_version: 1,
                number_cores: NonZeroUsize::new(8).unwrap(),
                m: Array2::eye(NUMBER_FEATURES),
            }
        );
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
}
