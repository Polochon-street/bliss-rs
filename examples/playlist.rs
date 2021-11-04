use anyhow::Result;
use bliss_audio::distance::{closest_to_first_song, dedup_playlist, euclidean_distance};
use bliss_audio::{library::analyze_paths_streaming, Song};
use glob::glob;
use mime_guess;
use std::env;
use std::fs;
use std::path::Path;

/* Analyzes a folder recursively, and make a playlist out of the file
 * provided by the user. */
// TODO still:
// * Save the results somewhere to avoid analyzing stuff over and over
// * Make the output file configurable
// * Allow to choose between outputing to stdout and a file
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
    let songs = glob(&pattern.to_string_lossy())?
        .map(|e| fs::canonicalize(e.unwrap()).unwrap())
        .filter(|e| match mime_guess::from_path(e).first() {
            Some(m) => m.type_() == "audio",
            None => false,
        })
        .map(|x| x.to_string_lossy().to_string())
        .collect::<Vec<String>>();
    let rx = analyze_paths_streaming(songs)?;
    let first_song = Song::new(file)?;
    let mut analyzed_songs = vec![first_song.to_owned()];
    for (path, result) in rx.iter() {
        match result {
            Ok(song) => analyzed_songs.push(song),
            Err(e) => println!("error analyzing {}: {}", path, e),
        };
    }
    closest_to_first_song(&first_song, &mut analyzed_songs, euclidean_distance);
    dedup_playlist(&mut analyzed_songs, None);
    let playlist = analyzed_songs
        .iter()
        .map(|s| s.path.to_string_lossy().to_string())
        .collect::<Vec<String>>()
        .join("\n");
    println!("{}", playlist);
    fs::write("./playlist.m3u", playlist)?;
    Ok(())
}
