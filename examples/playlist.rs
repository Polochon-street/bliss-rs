use anyhow::Result;
use bliss_audio::distance::{closest_to_first_song, dedup_playlist, euclidean_distance};
use bliss_audio::{library::analyze_paths_streaming, Song};
use glob::glob;
use mime_guess;
use serde_json;
use std::env;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/* Analyzes a folder recursively, and make a playlist out of the file
 * provided by the user. */
// TODO still:
// * Mention it in the README
// * Make the output file configurable
// * Allow to choose between outputing to stdout and a file
#[cfg(feature = "serde")]
fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 3 || args.len() < 2 {
        println!("Usage: ./playlist <folder> <file>");
        println!(
            "Creates a playlist of all audio files in a folder (recursively), \
            starting with <file>, and outputs the result both to stdout and \
            a `playlist.m3u` file in the current folder."
        );
        return Ok(());
    }
    let folder = &args[0];
    let file = fs::canonicalize(&args[1])?;
    let pattern = Path::new(folder).join("**").join("*");

    let mut songs: Vec<Song> = Vec::new();
    let analysis_file = fs::File::open("./songs.json");
    if let Ok(f) = analysis_file {
        let reader = BufReader::new(f);
        songs = serde_json::from_reader(reader)?;
    }

    let analyzed_paths = songs
        .iter()
        .map(|s| s.path.to_owned())
        .collect::<Vec<PathBuf>>();

    let paths = glob(&pattern.to_string_lossy())?
        .map(|e| fs::canonicalize(e.unwrap()).unwrap())
        .filter(|e| match mime_guess::from_path(e).first() {
            Some(m) => m.type_() == "audio",
            None => false,
        })
        .map(|x| x.to_string_lossy().to_string())
        .collect::<Vec<String>>();

    let rx = analyze_paths_streaming(
        paths
            .iter()
            .filter(|p| !analyzed_paths.contains(&PathBuf::from(p)))
            .map(|p| p.to_owned())
            .collect(),
    )?;
    let first_song = Song::new(file)?;
    let mut analyzed_songs = vec![first_song.to_owned()];
    for (path, result) in rx.iter() {
        match result {
            Ok(song) => analyzed_songs.push(song),
            Err(e) => println!("error analyzing {}: {}", path, e),
        };
    }
    analyzed_songs.extend_from_slice(&songs);
    let serialized = serde_json::to_string(&analyzed_songs).unwrap();
    let mut songs_to_chose_from = analyzed_songs
        .into_iter()
        .filter(|x| x == &first_song || paths.contains(&x.path.to_string_lossy().to_string()))
        .collect();
    closest_to_first_song(&first_song, &mut songs_to_chose_from, euclidean_distance);
    dedup_playlist(&mut songs_to_chose_from, None);
    fs::write("./songs.json", serialized)?;
    let playlist = songs_to_chose_from
        .iter()
        .map(|s| s.path.to_string_lossy().to_string())
        .collect::<Vec<String>>()
        .join("\n");
    println!("{}", playlist);
    fs::write("./playlist.m3u", playlist)?;
    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("You need the serde feature enabled to run this file.");
}
