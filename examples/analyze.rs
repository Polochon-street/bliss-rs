use bliss_audio::Song;
use std::env;

/**
 * Simple utility to print the result of an Analysis.
 *
 * Takes a list of files to analyze an the result of the corresponding Analysis.
 */
fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    for path in &args {
        match Song::from_path(&path) {
            Ok(song) => println!("{}: {:?}", path, song.analysis),
            Err(e) => println!("{}: {}", path, e),
        }
    }
}
