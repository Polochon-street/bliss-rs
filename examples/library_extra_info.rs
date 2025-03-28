/// Basic example of how one would combine bliss with an "audio player",
/// through [Library], showing how to put extra info in the database for
/// each song.
///
/// For simplicity's sake, this example recursively gets songs from a folder
/// to emulate an audio player library, without handling CUE files.
use anyhow::Result;
use bliss_audio::decoder::ffmpeg::FFmpegDecoder as Decoder;
use bliss_audio::library::{AppConfigTrait, BaseConfig, Library};
use clap::{App, Arg, SubCommand};
use glob::glob;
use serde::{Deserialize, Serialize};
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
/// A config structure, that will be serialized as a
/// JSON file upon Library creation.
pub struct Config {
    #[serde(flatten)]
    /// The base configuration, containing both the config file
    /// path, as well as the database path.
    pub base_config: BaseConfig,
    /// An extra field, to store the music library path. Any number
    /// of arbitrary fields (even Serializable structures) can
    /// of course be added.
    pub music_library_path: PathBuf,
}

impl Config {
    pub fn new(
        music_library_path: PathBuf,
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
        number_cores: Option<NonZeroUsize>,
    ) -> Result<Self> {
        let base_config = BaseConfig::new(config_path, database_path, number_cores)?;
        Ok(Self {
            base_config,
            music_library_path,
        })
    }
}

// The AppConfigTrait must know how to access the base config.
impl AppConfigTrait for Config {
    fn base_config(&self) -> &BaseConfig {
        &self.base_config
    }

    fn base_config_mut(&mut self) -> &mut BaseConfig {
        &mut self.base_config
    }
}

// A trait allowing to implement methods for the Library,
// useful if you don't need to store extra information in fields.
// Otherwise, doing
// ```
// struct CustomLibrary {
//    library: Library<Config>,
//    extra_field: ...,
// }
// ```
// and implementing functions for that struct would be the way to go.
// That's what the [reference](https://github.com/Polochon-street/blissify-rs)
// implementation does.
trait CustomLibrary {
    fn song_paths_info(&self) -> Result<Vec<(String, ExtraInfo)>>;
}

impl CustomLibrary for Library<Config, Decoder> {
    /// Get all songs in the player library, along with the extra info
    /// one would want to store along with each song.
    fn song_paths_info(&self) -> Result<Vec<(String, ExtraInfo)>> {
        let music_path = &self.config.music_library_path;
        let pattern = Path::new(&music_path).join("**").join("*");

        Ok(glob(&pattern.to_string_lossy())?
            .map(|e| fs::canonicalize(e.unwrap()).unwrap())
            .filter_map(|e| {
                mime_guess::from_path(&e).first().map(|m| {
                    (
                        e.to_string_lossy().to_string(),
                        ExtraInfo {
                            extension: e.extension().map(|e| e.to_string_lossy().to_string()),
                            file_name: e.file_name().map(|e| e.to_string_lossy().to_string()),
                            mime_type: format!("{}/{}", m.type_(), m.subtype()),
                        },
                    )
                })
            })
            .collect::<Vec<(String, ExtraInfo)>>())
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
// An (somewhat simple) example of what extra metadata one would put, along
// with song analysis data.
struct ExtraInfo {
    extension: Option<String>,
    file_name: Option<String>,
    mime_type: String,
}

// A simple example of what a CLI-app would look.
//
// Note that `Library::new` is used only on init, and subsequent
// commands use `Library::from_path`.
fn main() -> Result<()> {
    let matches = App::new("library-example")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Polochon_street")
        .about("Example binary implementing bliss for an audio player.")
        .subcommand(
            SubCommand::with_name("init")
                .about(
                    "Initialize a Library, both storing the config and analyzing folders
                containing songs.",
                )
                .arg(
                    Arg::with_name("FOLDER")
                        .help("A folder containing the music library to analyze.")
                        .required(true),
                )
                .arg(
                    Arg::with_name("database-path")
                        .short("d")
                        .long("database-path")
                        .help(
                            "Optional path where to store the database file containing
                 the songs' analysis. Defaults to XDG_DATA_HOME/bliss-rs/bliss.db.",
                        )
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("config-path")
                        .short("c")
                        .long("config-path")
                        .help(
                            "Optional path where to store the config file containing
                 the library setup. Defaults to XDG_DATA_HOME/bliss-rs/config.json.",
                        )
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("update")
                .about(
                    "Update a Library's songs, trying to analyze failed songs,
                    as well as songs not in the library.",
                )
                .arg(
                    Arg::with_name("config-path")
                        .short("c")
                        .long("config-path")
                        .help(
                            "Optional path where to load the config file containing
                 the library setup. Defaults to XDG_DATA_HOME/bliss-rs/config.json.",
                        )
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("playlist")
                .about(
                    "Make a playlist, starting with the song at SONG_PATH, returning
                       the songs' paths.",
                )
                .arg(Arg::with_name("SONG_PATH").takes_value(true))
                .arg(
                    Arg::with_name("config-path")
                        .short("c")
                        .long("config-path")
                        .help(
                            "Optional path where to load the config file containing
                 the library setup. Defaults to XDG_DATA_HOME/bliss-rs/config.json.",
                        )
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("playlist-length")
                        .short("l")
                        .long("playlist-length")
                        .help("Optional playlist length. Defaults to 20.")
                        .takes_value(true),
                ),
        )
        .get_matches();
    if let Some(sub_m) = matches.subcommand_matches("init") {
        let folder = PathBuf::from(sub_m.value_of("FOLDER").unwrap());
        let config_path = sub_m.value_of("config-path").map(PathBuf::from);
        let database_path = sub_m.value_of("database-path").map(PathBuf::from);

        let config = Config::new(folder, config_path, database_path, None)?;
        let mut library = Library::new(config)?;

        library.analyze_paths_extra_info(library.song_paths_info()?, true)?;
    } else if let Some(sub_m) = matches.subcommand_matches("update") {
        let config_path = sub_m.value_of("config-path").map(PathBuf::from);
        let mut library: Library<Config, Decoder> = Library::from_config_path(config_path)?;
        library.update_library_extra_info(library.song_paths_info()?, true, true)?;
    } else if let Some(sub_m) = matches.subcommand_matches("playlist") {
        let song_path = sub_m.value_of("SONG_PATH").unwrap();
        let config_path = sub_m.value_of("config-path").map(PathBuf::from);
        let playlist_length = sub_m
            .value_of("playlist-length")
            .unwrap_or("20")
            .parse::<usize>()?;
        let library: Library<Config, Decoder> = Library::from_config_path(config_path)?;
        let playlist = library
            .playlist_from::<ExtraInfo>(&[song_path])?
            .take(playlist_length)
            .map(|s| {
                (
                    s.bliss_song.path.to_string_lossy().to_string(),
                    s.extra_info.mime_type,
                )
            })
            .collect::<Vec<(String, String)>>();
        for (path, mime_type) in playlist {
            println!("{path} <{mime_type}>");
        }
    }

    Ok(())
}
