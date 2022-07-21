use anyhow::Result;
use bliss_audio::library::{AppConfigTrait, BaseConfig, Library};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
pub struct MPDConfig {
    #[serde(flatten)]
    pub base_config: BaseConfig,
    pub mpd_config_path: PathBuf,
    pub music_library_path: PathBuf,
}

impl MPDConfig {
    fn get_mpd_path(mpd_path: Option<PathBuf>) -> Result<PathBuf> {
        Ok(mpd_path.unwrap_or_else(|| PathBuf::from("/path/to/MPD")))
    }

    fn get_library_path(music_library_path: Option<PathBuf>) -> Result<PathBuf> {
        Ok(music_library_path.unwrap_or_else(|| PathBuf::from("path/fom")))
    }

    pub fn new(
        config_path: Option<PathBuf>,
        database_path: Option<PathBuf>,
        mpd_path: Option<PathBuf>,
        music_library_path: Option<PathBuf>,
    ) -> Result<Self> {
        let base_config = BaseConfig::new(config_path, database_path)?;
        Ok(Self {
            base_config,
            mpd_config_path: Self::get_mpd_path(mpd_path)?,
            music_library_path: Self::get_library_path(music_library_path)?,
        })
    }
}

impl AppConfigTrait for MPDConfig {
    fn base_config(&self) -> &BaseConfig {
        &self.base_config
    }

    fn serialize_config(&self) -> Result<String> {
        Ok(serde_ini::to_string(&self).unwrap())
    }
}

fn main() -> Result<()> {
    let mpd_config = MPDConfig::new(
        Some(PathBuf::from("/home/polochon/config.ini")),
        None,
        None,
        None,
    )?;
    Library::new(mpd_config)?;
    Ok(())
}
