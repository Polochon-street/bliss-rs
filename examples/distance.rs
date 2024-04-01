use bliss_audio::{playlist::euclidean_distance, Song};
use std::env;

/**
 * Simple utility to print distance between two songs according to bliss.
 *
 * Takes two file paths, and analyze the corresponding songs, printing
 * the distance between the two files according to bliss.
 */
fn main() -> Result<(), String> {
    let mut paths = env::args().skip(1).take(2);

    let first_path = paths.next().ok_or("Help: ./distance <song1> <song2>")?;
    let second_path = paths.next().ok_or("Help: ./distance <song1> <song2>")?;

    let song1 = Song::from_path(first_path).map_err(|x| x.to_string())?;
    let song2 = Song::from_path(second_path).map_err(|x| x.to_string())?;

    println!(
        "d({:?}, {:?}) = {}",
        song1.path,
        song2.path,
        euclidean_distance(&song1.analysis.as_arr1(), &song2.analysis.as_arr1())
    );
    Ok(())
}
