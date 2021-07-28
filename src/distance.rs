//! Module containing various distance metric functions.
//!
//! All of these functions are intended to be used with the
//! [custom_distance](Song::custom_distance) method, or with
//! [playlist_from_songs_custom_distance](Library::playlist_from_song_custom_distance).
//!
//! They will yield different styles of playlists, so don't hesitate to
//! experiment with them if the default (euclidean distance for now) doesn't
//! suit you.
#[cfg(doc)]
use crate::Library;
use crate::Song;
use crate::NUMBER_FEATURES;
use ndarray::{Array, Array1};
use noisy_float::prelude::*;

/// Convenience trait for user-defined distance metrics.
pub trait DistanceMetric: Fn(&Array1<f32>, &Array1<f32>) -> f32 {}
impl<F> DistanceMetric for F where F: Fn(&Array1<f32>, &Array1<f32>) -> f32 {}

/// Return the [euclidean
/// distance](https://en.wikipedia.org/wiki/Euclidean_distance#Higher_dimensions)
/// between two vectors.
pub fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    // Could be any square symmetric positive semi-definite matrix;
    // just no metric learning has been done yet.
    // See https://lelele.io/thesis.pdf chapter 4.
    let m = Array::eye(NUMBER_FEATURES);

    (a - b).dot(&m).dot(&(a - b)).sqrt()
}

/// Return the [cosine
/// distance](https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity)
/// between two vectors.
pub fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let similarity = a.dot(b) / (a.dot(a).sqrt() * b.dot(b).sqrt());
    1. - similarity
}

/// Sort `songs` in place by putting songs close to `first_song` first
/// using the `distance` metric. Deduplicate identical songs.
pub fn closest_to_first_song(
    first_song: &Song,
    songs: &mut Vec<Song>,
    distance: impl DistanceMetric,
) {
    songs.sort_by_cached_key(|song| n32(first_song.custom_distance(song, &distance)));
    songs.dedup_by_key(|song| n32(first_song.custom_distance(song, &distance)));
}

/// Sort `songs` in place using the `distance` metric and ordering by
/// the smallest distance between each song. Deduplicate identical songs.
///
/// If the generated playlist is `[song1, song2, song3, song4]`, it means
/// song2 is closest to song1, song3 is closest to song2, and song4 is closest
/// to song3.
pub fn song_to_song(first_song: &Song, songs: &mut Vec<Song>, distance: impl DistanceMetric) {
    let mut new_songs = vec![first_song.to_owned()];
    let mut song = first_song.to_owned();
    loop {
        if songs.is_empty() {
            break;
        }
        songs 
            .retain(|s| n32(song.custom_distance(s, &distance)) != 0.);
        songs.sort_by_key(|s| n32(song.custom_distance(s, &distance)));
        song = songs.remove(0);
        new_songs.push(song.to_owned());
    }
    *songs = new_songs;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Analysis;
    use ndarray::arr1;
    use std::path::Path;

    #[test]
    fn test_song_to_song() {
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let first_song_dupe = Song {
            path: Path::new("path-to-dupe").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.9, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.5, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let mut songs = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
        ];
        song_to_song(&first_song, &mut songs, euclidean_distance);
        assert_eq!(
            songs,
            vec![first_song, second_song, third_song, fourth_song],
        );
    }

    #[test]
    fn test_sort_closest_to_first_song() {
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let first_song_dupe = Song {
            path: Path::new("path-to-dupe").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.9, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.5, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fifth_song = Song {
            path: Path::new("path-to-fifth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let mut songs = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        closest_to_first_song(&first_song, &mut songs, euclidean_distance);
        assert_eq!(
            songs,
            vec![first_song, second_song, fourth_song, third_song],
        );
    }

    #[test]
    fn test_euclidean_distance() {
        let a = arr1(&[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
        ]);
        let b = arr1(&[
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        ]);
        assert_eq!(euclidean_distance(&a, &b), 4.242640687119285);

        let a = arr1(&[0.5; 20]);
        let b = arr1(&[0.5; 20]);
        assert_eq!(euclidean_distance(&a, &b), 0.);
        assert_eq!(euclidean_distance(&a, &b), 0.);
    }

    #[test]
    fn test_cosine_distance() {
        let a = arr1(&[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
        ]);
        let b = arr1(&[
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        ]);
        assert_eq!(cosine_distance(&a, &b), 0.7705842661294382);

        let a = arr1(&[0.5; 20]);
        let b = arr1(&[0.5; 20]);
        assert_eq!(cosine_distance(&a, &b), 0.);
        assert_eq!(cosine_distance(&a, &b), 0.);
    }
}
