//! Module containing various functions to build playlists, as well as various
//! distance metrics.
//!
//! All of the distance functions are intended to be used with the
//! [custom_distance](Song::custom_distance) method, or with
//!
//! They will yield different styles of playlists, so don't hesitate to
//! experiment with them if the default (euclidean distance for now) doesn't
//! suit you.
use crate::{BlissError, BlissResult, Song, NUMBER_FEATURES};
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use noisy_float::prelude::*;
use std::collections::HashMap;

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
/// using the `distance` metric.
pub fn closest_to_first_song(
    first_song: &Song,
    #[allow(clippy::ptr_arg)] songs: &mut Vec<Song>,
    distance: impl DistanceMetric,
) {
    songs.sort_by_cached_key(|song| n32(first_song.custom_distance(song, &distance)));
}

/// Sort `songs` in place using the `distance` metric and ordering by
/// the smallest distance between each song.
///
/// If the generated playlist is `[song1, song2, song3, song4]`, it means
/// song2 is closest to song1, song3 is closest to song2, and song4 is closest
/// to song3.
///
/// Note that this has a tendency to go from one style to the other very fast,
/// and it can be slow on big libraries.
pub fn song_to_song(first_song: &Song, songs: &mut Vec<Song>, distance: impl DistanceMetric) {
    let mut new_songs = Vec::with_capacity(songs.len());
    let mut song = first_song.to_owned();

    while !songs.is_empty() {
        let distances: Array1<f32> =
            Array::from_shape_fn(songs.len(), |i| song.custom_distance(&songs[i], &distance));
        let idx = distances.argmin().unwrap();
        song = songs[idx].to_owned();
        new_songs.push(song.to_owned());
        songs.retain(|s| s != &song);
    }
    *songs = new_songs;
}

/// Remove duplicate songs from a playlist, in place.
///
/// Two songs are considered duplicates if they either have the same,
/// non-empty title and artist name, or if they are close enough in terms
/// of distance.
///
/// # Arguments
///
/// * `songs`: The playlist to remove duplicates from.
/// * `distance_threshold`: The distance threshold under which two songs are
///   considered identical. If `None`, a default value of 0.05 will be used.
pub fn dedup_playlist(songs: &mut Vec<Song>, distance_threshold: Option<f32>) {
    dedup_playlist_custom_distance(songs, distance_threshold, euclidean_distance);
}

/// Remove duplicate songs from a playlist, in place, using a custom distance
/// metric.
///
/// Two songs are considered duplicates if they either have the same,
/// non-empty title and artist name, or if they are close enough in terms
/// of distance.
///
/// # Arguments
///
/// * `songs`: The playlist to remove duplicates from.
/// * `distance_threshold`: The distance threshold under which two songs are
///   considered identical. If `None`, a default value of 0.05 will be used.
/// * `distance`: A custom distance metric.
pub fn dedup_playlist_custom_distance(
    songs: &mut Vec<Song>,
    distance_threshold: Option<f32>,
    distance: impl DistanceMetric,
) {
    songs.dedup_by(|s1, s2| {
        n32(s1.custom_distance(s2, &distance)) < distance_threshold.unwrap_or(0.05)
            || (s1.title.is_some()
                && s2.title.is_some()
                && s1.artist.is_some()
                && s2.artist.is_some()
                && s1.title == s2.title
                && s1.artist == s2.artist)
    });
}

/// Return a list of albums in a `pool` of songs that are similar to
/// songs in `group`, discarding songs that don't belong to an album.
/// It basically makes an "album" playlist from the `pool` of songs.
///
/// `group` should be ordered by track number.
///
/// Songs from `group` would usually just be songs from an album, but not
/// necessarily - they are discarded from `pool` no matter what.
///
/// # Arguments
///
/// * `group` - A small group of songs, e.g. an album.
/// * `pool` - A pool of songs to find similar songs in, e.g. a user's song
/// library.
///
/// # Returns
///
/// A vector of songs, including `group` at the beginning, that you
/// most likely want to plug in your audio player by using something like
/// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
pub fn closest_album_to_group(group: Vec<Song>, pool: Vec<Song>) -> BlissResult<Vec<Song>> {
    let mut albums_analysis: HashMap<&str, Array2<f32>> = HashMap::new();
    let mut albums = Vec::new();

    // Remove songs from the group from the pool.
    let pool = pool
        .into_iter()
        .filter(|s| !group.contains(s))
        .collect::<Vec<_>>();
    for song in &pool {
        if let Some(album) = &song.album {
            if let Some(analysis) = albums_analysis.get_mut(album as &str) {
                analysis
                    .push_row(song.analysis.as_arr1().view())
                    .map_err(|e| {
                        BlissError::ProviderError(format!("while computing distances: {}", e))
                    })?;
            } else {
                let mut array = Array::zeros((1, song.analysis.as_arr1().len()));
                array.assign(&song.analysis.as_arr1());
                albums_analysis.insert(album, array);
            }
        }
    }
    let mut group_analysis = Array::zeros((group.len(), NUMBER_FEATURES));
    for (song, mut column) in group.iter().zip(group_analysis.axis_iter_mut(Axis(0))) {
        column.assign(&song.analysis.as_arr1());
    }
    let first_analysis = group_analysis
        .mean_axis(Axis(0))
        .ok_or_else(|| BlissError::ProviderError(String::from("Mean of empty slice")))?;
    for (album, analysis) in albums_analysis.iter() {
        let mean_analysis = analysis
            .mean_axis(Axis(0))
            .ok_or_else(|| BlissError::ProviderError(String::from("Mean of empty slice")))?;
        let album = album.to_owned();
        albums.push((album, mean_analysis.to_owned()));
    }

    albums.sort_by_key(|(_, analysis)| n32(euclidean_distance(&first_analysis, analysis)));
    let mut playlist = group;
    for (album, _) in albums {
        let mut al = pool
            .iter()
            .filter(|s| s.album.is_some() && s.album.as_ref().unwrap() == &album.to_string())
            .map(|s| s.to_owned())
            .collect::<Vec<Song>>();
        al.sort_by(|s1, s2| {
            let track_number1 = s1
                .track_number
                .to_owned()
                .unwrap_or_else(|| String::from(""));
            let track_number2 = s2
                .track_number
                .to_owned()
                .unwrap_or_else(|| String::from(""));
            if let Ok(x) = track_number1.parse::<i32>() {
                if let Ok(y) = track_number2.parse::<i32>() {
                    return x.cmp(&y);
                }
            }
            s1.track_number.cmp(&s2.track_number)
        });
        playlist.extend_from_slice(&al);
    }
    Ok(playlist)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Analysis;
    use ndarray::arr1;
    use std::path::Path;

    #[test]
    fn test_dedup_playlist_custom_distance() {
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
            title: Some(String::from("dupe-title")),
            artist: Some(String::from("dupe-artist")),
            ..Default::default()
        };
        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            title: Some(String::from("dupe-title")),
            artist: Some(String::from("dupe-artist")),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.5, 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            artist: Some(String::from("no-dupe-artist")),
            title: Some(String::from("dupe-title")),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let fifth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([
                2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0.001, 1., 1., 1.,
            ]),
            ..Default::default()
        };

        let mut playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        dedup_playlist_custom_distance(&mut playlist, None, euclidean_distance);
        assert_eq!(
            playlist,
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
            ],
        );
        let mut playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        dedup_playlist_custom_distance(&mut playlist, Some(20.), cosine_distance);
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let mut playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        dedup_playlist(&mut playlist, Some(20.));
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let mut playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        dedup_playlist(&mut playlist, None);
        assert_eq!(
            playlist,
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
            ]
        );
    }

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
            vec![
                first_song,
                first_song_dupe.to_owned(),
                second_song,
                third_song,
                fourth_song
            ],
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
            vec![
                first_song,
                first_song_dupe,
                second_song,
                fourth_song,
                fifth_song,
                third_song
            ],
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

    #[test]
    fn test_closest_to_group() {
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([0.; 20]),
            album: Some(String::from("Album")),
            artist: Some(String::from("Artist")),
            track_number: Some(String::from("01")),
            ..Default::default()
        };

        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Artist")),
            track_number: Some(String::from("10")),
            ..Default::default()
        };

        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            album: Some(String::from("Album")),
            artist: Some(String::from("Another Artist")),
            track_number: Some(String::from("02")),
            ..Default::default()
        };

        let fourth_song = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Another Artist")),
            track_number: Some(String::from("01")),
            ..Default::default()
        };
        let fifth_song = Song {
            path: Path::new("path-to-fifth").to_path_buf(),
            analysis: Analysis::new([40.; 20]),
            artist: Some(String::from("Third Artist")),
            album: None,
            ..Default::default()
        };

        let pool = vec![
            first_song.to_owned(),
            fourth_song.to_owned(),
            third_song.to_owned(),
            second_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let group = vec![first_song.to_owned(), third_song.to_owned()];
        assert_eq!(
            vec![
                first_song.to_owned(),
                third_song.to_owned(),
                fourth_song.to_owned(),
                second_song.to_owned()
            ],
            closest_album_to_group(group, pool.to_owned()).unwrap(),
        );
    }
}
