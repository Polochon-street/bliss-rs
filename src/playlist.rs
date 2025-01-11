//! Module containing various functions to build playlists, as well as various
//! distance metrics.
//!
//! All of the distance functions are intended to be used in isolation
//! or with e.g. [dedup_playlist_custom_distance].
//!
//! They will yield different styles of playlists, so don't hesitate to
//! experiment with them if the default (euclidean distance for now) doesn't
//! suit you.
use crate::{BlissError, BlissResult, Song, NUMBER_FEATURES};
use extended_isolation_forest::{Forest, ForestOptions};
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use noisy_float::prelude::*;
use std::collections::HashMap;

/// Trait for creating a distance metric, measuring the distance to a set of vectors. If this
/// metric requires any kind of training, this should be done in the build function so that the
/// returned DistanceMetric instance is already trained and ready to use.
///
/// Currently, the best metric for measuring the distance to a set of songs is the extended
/// isolation forest (implemented on [ForestOptions]). For measuring the distance to a single song,
/// extended isolation forest doesn't work and [euclidean_distance] or [cosine_distance] are good
/// options.
pub trait DistanceMetricBuilder {
    /// Build a distance metric that measures the distance to vectors.
    fn build<'a>(&'a self, vectors: &[Array1<f32>]) -> Box<dyn DistanceMetric + 'a>;
}

/// Measure the distance to a vector, from the vector(s) in the internal state of this metric.
pub trait DistanceMetric {
    /// Return the distance from the set of vectors that this metric was built from.
    fn distance(&self, vector: &Array1<f32>) -> f32;
}

/// Convenience struct used for implementing DistanceMetric for plain functions.
pub struct FunctionDistanceMetric<'a, F: Fn(&Array1<f32>, &Array1<f32>) -> f32> {
    func: &'a F,
    state: Vec<Array1<f32>>,
}

impl<F> DistanceMetricBuilder for F
where
    F: Fn(&Array1<f32>, &Array1<f32>) -> f32 + 'static,
{
    fn build<'a>(&'a self, vectors: &[Array1<f32>]) -> Box<dyn DistanceMetric + 'a> {
        Box::new(FunctionDistanceMetric {
            func: self,
            state: vectors.iter().map(|s| s.to_owned()).collect(),
        })
    }
}

impl<F: Fn(&Array1<f32>, &Array1<f32>) -> f32 + 'static> DistanceMetric
    for FunctionDistanceMetric<'_, F>
{
    fn distance(&self, vector: &Array1<f32>) -> f32 {
        self.state.iter().map(|v| (self.func)(v, vector)).sum()
    }
}

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

/// Return a Mahalanobis distance function, usable with the "standard"
/// playlist-making functions provided here such as [closest_to_songs] and
/// [song_to_song].
///
/// # Arguments
///
/// * `m`: a matrix representing the weights of the different features.
///
/// # Usage
///
/// ```
/// use bliss_audio::{Song, Analysis, NUMBER_FEATURES};
/// use bliss_audio::playlist::{closest_to_songs, mahalanobis_distance_builder};
/// use ndarray::Array2;
///
/// // Songs here for the example; in reality, they would be analyzed or
/// // pulled from a database.
/// let first_song = Song {
///     path: "path-to-first".into(),
///         analysis: Analysis::new([
///             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
///         ]),
///         ..Default::default()
///     };
/// let second_song = Song {
///     path: "path-to-second".into(),
///     analysis: Analysis::new([
///         1.5, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
///     ]),
///     ..Default::default()
/// };
///
/// let third_song = Song {
///     path: "path-to-third".into(),
///     analysis: Analysis::new([
///         2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.9, 1., 1., 1.,
///     ]),
///     ..Default::default()
/// };
/// // The weights of the features, here, equal to the identity matrix, i.e.,
/// // it represents the euclidean distance.
/// let m = Array2::eye(NUMBER_FEATURES);
/// let distance = mahalanobis_distance_builder(m);
/// let playlist = closest_to_songs(&[first_song], &[second_song, third_song], &distance).collect::<Vec<_>>();
/// ```
pub fn mahalanobis_distance_builder(m: Array2<f32>) -> impl Fn(&Array1<f32>, &Array1<f32>) -> f32 {
    move |a: &Array1<f32>, b: &Array1<f32>| mahalanobis_distance(a, b, &m)
}

/// Returns the Mahalanobis distance between two vectors, also the weighted
/// distance between those two vectors. The weight is made according to the
/// distance matrix m.
/// In most cases, building a Mahalanobis distance function using
/// [mahalanobis_distance_builder] and using it makes more sense, since it
/// makes it usable with the other provided functions such as [closest_to_songs]2
/// and [song_to_song].
pub fn mahalanobis_distance(a: &Array1<f32>, b: &Array1<f32>, m: &Array2<f32>) -> f32 {
    (a - b).dot(m).dot(&(a - b)).sqrt()
}

fn feature_array1_to_array(f: &Array1<f32>) -> [f32; NUMBER_FEATURES] {
    f.as_slice()
        .expect("Couldn't convert feature vector to slice")
        .try_into()
        .expect("Couldn't convert slice to array")
}

impl DistanceMetricBuilder for ForestOptions {
    fn build(&self, vectors: &[Array1<f32>]) -> Box<dyn DistanceMetric> {
        let a = &*vectors
            .iter()
            .map(feature_array1_to_array)
            .collect::<Vec<_>>();

        if self.sample_size > vectors.len() {
            let mut opts = self.clone();
            opts.sample_size = self.sample_size.min(vectors.len());
            Box::new(Forest::from_slice(a, &opts).unwrap())
        } else {
            Box::new(Forest::from_slice(a, self).unwrap())
        }
    }
}

impl DistanceMetric for Forest<f32, NUMBER_FEATURES> {
    fn distance(&self, vector: &Array1<f32>) -> f32 {
        self.score(&feature_array1_to_array(vector)) as f32
    }
}

/// Return a playlist made of songs as close as possible to `selected_songs` from
/// the pool of songs in `candidate_songs`, using the `distance` metric to quantify
/// the distance between songs.
pub fn closest_to_songs<'a, T: AsRef<Song> + Clone + 'a>(
    initial_songs: &[T],
    candidate_songs: &[T],
    metric_builder: &'a dyn DistanceMetricBuilder,
) -> impl Iterator<Item = T> + 'a {
    let initial_songs = initial_songs
        .iter()
        .map(|c| c.as_ref().analysis.as_arr1())
        .collect::<Vec<_>>();
    let metric = metric_builder.build(&initial_songs);
    let mut candidate_songs = candidate_songs.to_vec();
    candidate_songs
        .sort_by_cached_key(|song| n32(metric.distance(&song.as_ref().analysis.as_arr1())));
    candidate_songs.into_iter()
}

struct SongToSongIterator<'a, T: AsRef<Song> + Clone> {
    pool: Vec<T>,
    vectors: Vec<Array1<f32>>,
    metric_builder: &'a dyn DistanceMetricBuilder,
}

impl<T: AsRef<Song> + Clone> Iterator for SongToSongIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.pool.is_empty() {
            return None;
        }
        let metric = self.metric_builder.build(&self.vectors);
        let distances: Array1<f32> = Array::from_shape_fn(self.pool.len(), |j| {
            metric.distance(&self.pool[j].as_ref().analysis.as_arr1())
        });
        let idx = distances.argmin().unwrap();
        // TODO instead of having a vector that's of size n and then
        // size 1 all the time, find a clever solution on iterator init
        // or something
        self.vectors.clear();
        let song = self.pool.remove(idx);
        self.vectors.push(song.as_ref().analysis.as_arr1());
        Some(song)
    }
}

/// Return an iterator of sorted songs from `candidate_songs` using
/// the `distance` metric and ordering by the smallest distance between each song.
///
/// If the generated playlist is `[song1, song2, song3, song4]`, it means
/// song2 is closest to song1, song3 is closest to song2, and song4 is closest
/// to song3.
///
/// Note that this has a tendency to go from one style to the other very fast,
/// and it can be slow on big libraries.
pub fn song_to_song<'a, T: AsRef<Song> + Clone + 'a>(
    initial_songs: &[T],
    candidate_songs: &[T],
    metric_builder: &'a dyn DistanceMetricBuilder,
) -> impl Iterator<Item = T> + 'a {
    let vectors = initial_songs
        .iter()
        .map(|s| s.as_ref().analysis.as_arr1())
        .collect::<Vec<_>>();
    let pool = candidate_songs.to_vec();
    let iterator = SongToSongIterator {
        vectors,
        metric_builder,
        pool,
    };
    iterator.into_iter()
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
pub fn dedup_playlist<'a, T: AsRef<Song>>(
    playlist: impl Iterator<Item = T> + 'a,
    distance_threshold: Option<f32>,
) -> impl Iterator<Item = T> + 'a {
    dedup_playlist_custom_distance(playlist, distance_threshold, &euclidean_distance)
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
pub fn dedup_playlist_custom_distance<'a, T: AsRef<Song>>(
    playlist: impl Iterator<Item = T> + 'a,
    distance_threshold: Option<f32>,
    metric_builder: &'a dyn DistanceMetricBuilder,
) -> impl Iterator<Item = T> + 'a {
    let mut peekable = playlist.peekable();
    let final_iterator = std::iter::from_fn(move || {
        if let Some(s1) = peekable.next() {
            loop {
                if let Some(s2) = peekable.peek() {
                    let s1_ref = s1.as_ref();
                    let s2_ref = s2.as_ref();
                    let vector = [s1_ref.analysis.as_arr1()];
                    let metric = metric_builder.build(&vector);
                    let is_same = n32(metric.distance(&s2_ref.analysis.as_arr1()))
                        < distance_threshold.unwrap_or(0.05)
                        || (s1_ref.title.is_some()
                            && s2_ref.title.is_some()
                            && s1_ref.artist.is_some()
                            && s2_ref.artist.is_some()
                            && s1_ref.title == s2_ref.title
                            && s1_ref.artist == s2_ref.artist);
                    if is_same {
                        peekable.next();
                        continue;
                    } else {
                        return Some(s1);
                    }
                }
                return Some(s1);
            }
        }
        None
    });
    final_iterator
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
///    library.
///
/// # Returns
///
/// A vector of songs, including `group` at the beginning, that you
/// most likely want to plug in your audio player by using something like
/// `ret.map(|song| song.path.to_owned()).collect::<Vec<String>>()`.
pub fn closest_album_to_group<T: AsRef<Song> + Clone>(
    group: Vec<T>,
    pool: Vec<T>,
) -> BlissResult<Vec<T>> {
    let mut albums_analysis: HashMap<&str, Array2<f32>> = HashMap::new();
    let mut albums = Vec::new();

    // Remove songs from the group from the pool.
    let pool = pool
        .into_iter()
        .filter(|s| !group.iter().any(|gs| gs.as_ref() == s.as_ref()))
        .collect::<Vec<_>>();
    for song in &pool {
        if let Some(album) = &song.as_ref().album {
            if let Some(analysis) = albums_analysis.get_mut(album as &str) {
                analysis
                    .push_row(song.as_ref().analysis.as_arr1().view())
                    .map_err(|e| {
                        BlissError::ProviderError(format!("while computing distances: {e}"))
                    })?;
            } else {
                let mut array = Array::zeros((1, song.as_ref().analysis.as_arr1().len()));
                array.assign(&song.as_ref().analysis.as_arr1());
                albums_analysis.insert(album, array);
            }
        }
    }
    let mut group_analysis = Array::zeros((group.len(), NUMBER_FEATURES));
    for (song, mut column) in group.iter().zip(group_analysis.axis_iter_mut(Axis(0))) {
        column.assign(&song.as_ref().analysis.as_arr1());
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
            .filter(|s| s.as_ref().album.as_deref() == Some(album))
            .cloned()
            .collect::<Vec<T>>();
        al.sort_by(|s1, s2| {
            let track_number1 = s1.as_ref().track_number.to_owned();
            let track_number2 = s2.as_ref().track_number.to_owned();
            let disc_number1 = s1.as_ref().disc_number.to_owned();
            let disc_number2 = s2.as_ref().disc_number.to_owned();
            (disc_number1, track_number1).cmp(&(disc_number2, track_number2))
        });
        playlist.extend(al);
    }
    Ok(playlist)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Analysis;
    use ndarray::arr1;
    use std::path::Path;

    #[derive(Debug, Clone, PartialEq)]
    struct CustomSong {
        something: bool,
        bliss_song: Song,
    }

    impl AsRef<Song> for CustomSong {
        fn as_ref(&self) -> &Song {
            &self.bliss_song
        }
    }

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

        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist =
            dedup_playlist_custom_distance(playlist.into_iter(), None, &euclidean_distance)
                .collect::<Vec<_>>();
        assert_eq!(
            playlist,
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
            ],
        );
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist =
            dedup_playlist_custom_distance(playlist.into_iter(), Some(20.), &euclidean_distance)
                .collect::<Vec<_>>();
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist = dedup_playlist(playlist.into_iter(), Some(20.)).collect::<Vec<_>>();
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist = dedup_playlist(playlist.into_iter(), None).collect::<Vec<_>>();
        assert_eq!(
            playlist,
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
            ]
        );

        let first_song = CustomSong {
            bliss_song: first_song,
            something: true,
        };
        let second_song = CustomSong {
            bliss_song: second_song,
            something: true,
        };
        let first_song_dupe = CustomSong {
            bliss_song: first_song_dupe,
            something: true,
        };
        let third_song = CustomSong {
            bliss_song: third_song,
            something: true,
        };
        let fourth_song = CustomSong {
            bliss_song: fourth_song,
            something: true,
        };

        let fifth_song = CustomSong {
            bliss_song: fifth_song,
            something: true,
        };

        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist =
            dedup_playlist_custom_distance(playlist.into_iter(), None, &euclidean_distance)
                .collect::<Vec<_>>();
        assert_eq!(
            playlist,
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                fourth_song.to_owned(),
            ],
        );
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist =
            dedup_playlist_custom_distance(playlist.into_iter(), Some(20.), &cosine_distance)
                .collect::<Vec<_>>();
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist = dedup_playlist(playlist.into_iter(), Some(20.)).collect::<Vec<_>>();
        assert_eq!(playlist, vec![first_song.to_owned()]);
        let playlist = vec![
            first_song.to_owned(),
            first_song_dupe.to_owned(),
            second_song.to_owned(),
            third_song.to_owned(),
            fourth_song.to_owned(),
            fifth_song.to_owned(),
        ];
        let playlist = dedup_playlist(playlist.into_iter(), None).collect::<Vec<_>>();
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
            &first_song,
            &third_song,
            &first_song_dupe,
            &second_song,
            &fourth_song,
        ];
        let songs =
            song_to_song(&[&first_song], &mut songs, &euclidean_distance).collect::<Vec<_>>();
        assert_eq!(
            songs,
            vec![
                &first_song,
                &first_song_dupe,
                &second_song,
                &third_song,
                &fourth_song,
            ],
        );

        let first_song = CustomSong {
            bliss_song: first_song,
            something: true,
        };
        let second_song = CustomSong {
            bliss_song: second_song,
            something: true,
        };
        let first_song_dupe = CustomSong {
            bliss_song: first_song_dupe,
            something: true,
        };
        let third_song = CustomSong {
            bliss_song: third_song,
            something: true,
        };
        let fourth_song = CustomSong {
            bliss_song: fourth_song,
            something: true,
        };

        let mut songs: Vec<&CustomSong> = vec![
            &first_song,
            &first_song_dupe,
            &third_song,
            &fourth_song,
            &second_song,
        ];

        let songs =
            song_to_song(&[&first_song], &mut songs, &euclidean_distance).collect::<Vec<_>>();

        assert_eq!(
            songs,
            vec![
                &first_song,
                &first_song_dupe,
                &second_song,
                &third_song,
                &fourth_song,
            ],
        );
    }

    #[test]
    fn test_sort_closest_to_songs() {
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

        let songs = [
            &fifth_song,
            &fourth_song,
            &first_song,
            &first_song_dupe,
            &second_song,
            &third_song,
        ];
        let playlist: Vec<_> =
            closest_to_songs(&[&first_song], &songs, &euclidean_distance).collect();
        assert_eq!(
            playlist,
            [
                &first_song,
                &first_song_dupe,
                &second_song,
                &fifth_song,
                &fourth_song,
                &third_song
            ],
        );

        let first_song = CustomSong {
            bliss_song: first_song,
            something: true,
        };
        let second_song = CustomSong {
            bliss_song: second_song,
            something: true,
        };
        let first_song_dupe = CustomSong {
            bliss_song: first_song_dupe,
            something: true,
        };
        let third_song = CustomSong {
            bliss_song: third_song,
            something: true,
        };
        let fourth_song = CustomSong {
            bliss_song: fourth_song,
            something: true,
        };

        let fifth_song = CustomSong {
            bliss_song: fifth_song,
            something: true,
        };

        let mut songs = [
            &second_song,
            &first_song,
            &fourth_song,
            &first_song_dupe,
            &third_song,
            &fifth_song,
        ];

        let playlist: Vec<_> =
            closest_to_songs(&[&first_song], &mut songs, &euclidean_distance).collect();

        assert_eq!(
            playlist,
            [
                &first_song,
                &first_song_dupe,
                &second_song,
                &fourth_song,
                &fifth_song,
                &third_song
            ],
        );
    }

    #[test]
    fn test_mahalanobis_distance() {
        let a = arr1(&[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
        ]);
        let b = arr1(&[
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        ]);
        let m = Array2::eye(NUMBER_FEATURES)
            * arr1(&[
                1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ]);

        let distance = mahalanobis_distance_builder(m);
        assert_eq!(distance(&a, &b), 1.);
    }

    #[test]
    fn test_mahalanobis_distance_with_songs() {
        let first_song = Song {
            path: Path::new("path-to-first").to_path_buf(),
            analysis: Analysis::new([
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let second_song = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([
                1.5, 5., 6., 5., 6., 6., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let third_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([
                5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ]),
            ..Default::default()
        };
        let m = Array2::eye(NUMBER_FEATURES)
            * arr1(&[
                1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ]);
        let distance = mahalanobis_distance_builder(m);

        let playlist = closest_to_songs(
            &[first_song.clone()],
            &[third_song.clone(), second_song.clone()],
            &distance,
        )
        .collect::<Vec<_>>();
        assert_eq!(playlist, vec![second_song, third_song,]);
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
            track_number: Some(1),
            disc_number: Some(1),
            ..Default::default()
        };
        let second_song = Song {
            path: Path::new("path-to-third").to_path_buf(),
            analysis: Analysis::new([10.; 20]),
            album: Some(String::from("Album")),
            artist: Some(String::from("Another Artist")),
            track_number: Some(2),
            disc_number: Some(1),
            ..Default::default()
        };

        let first_song_other_album_disc_1 = Song {
            path: Path::new("path-to-second-2").to_path_buf(),
            analysis: Analysis::new([0.15; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Artist")),
            track_number: Some(1),
            disc_number: Some(1),
            ..Default::default()
        };
        let second_song_other_album_disc_1 = Song {
            path: Path::new("path-to-second").to_path_buf(),
            analysis: Analysis::new([0.1; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Artist")),
            track_number: Some(2),
            disc_number: Some(1),
            ..Default::default()
        };
        let first_song_other_album_disc_2 = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Another Artist")),
            track_number: Some(1),
            disc_number: Some(2),
            ..Default::default()
        };
        let second_song_other_album_disc_2 = Song {
            path: Path::new("path-to-fourth").to_path_buf(),
            analysis: Analysis::new([20.; 20]),
            album: Some(String::from("Another Album")),
            artist: Some(String::from("Another Artist")),
            track_number: Some(4),
            disc_number: Some(2),
            ..Default::default()
        };

        let song_no_album = Song {
            path: Path::new("path-to-fifth").to_path_buf(),
            analysis: Analysis::new([40.; 20]),
            artist: Some(String::from("Third Artist")),
            album: None,
            ..Default::default()
        };

        let pool = vec![
            first_song.to_owned(),
            second_song_other_album_disc_1.to_owned(),
            second_song_other_album_disc_2.to_owned(),
            second_song.to_owned(),
            first_song_other_album_disc_2.to_owned(),
            first_song_other_album_disc_1.to_owned(),
            song_no_album.to_owned(),
        ];
        let group = vec![first_song.to_owned(), second_song.to_owned()];
        assert_eq!(
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                first_song_other_album_disc_1.to_owned(),
                second_song_other_album_disc_1.to_owned(),
                first_song_other_album_disc_2.to_owned(),
                second_song_other_album_disc_2.to_owned(),
            ],
            closest_album_to_group(group, pool.to_owned()).unwrap(),
        );

        let first_song = CustomSong {
            bliss_song: first_song,
            something: true,
        };
        let second_song = CustomSong {
            bliss_song: second_song,
            something: true,
        };

        let first_song_other_album_disc_1 = CustomSong {
            bliss_song: first_song_other_album_disc_1,
            something: true,
        };
        let second_song_other_album_disc_1 = CustomSong {
            bliss_song: second_song_other_album_disc_1,
            something: true,
        };
        let first_song_other_album_disc_2 = CustomSong {
            bliss_song: first_song_other_album_disc_2,
            something: true,
        };
        let second_song_other_album_disc_2 = CustomSong {
            bliss_song: second_song_other_album_disc_2,
            something: true,
        };
        let song_no_album = CustomSong {
            bliss_song: song_no_album,
            something: true,
        };

        let pool = vec![
            first_song.to_owned(),
            second_song_other_album_disc_2.to_owned(),
            second_song_other_album_disc_1.to_owned(),
            second_song.to_owned(),
            first_song_other_album_disc_2.to_owned(),
            first_song_other_album_disc_1.to_owned(),
            song_no_album.to_owned(),
        ];
        let group = vec![first_song.to_owned(), second_song.to_owned()];
        assert_eq!(
            vec![
                first_song.to_owned(),
                second_song.to_owned(),
                first_song_other_album_disc_1.to_owned(),
                second_song_other_album_disc_1.to_owned(),
                first_song_other_album_disc_2.to_owned(),
                second_song_other_album_disc_2.to_owned(),
            ],
            closest_album_to_group(group, pool.to_owned()).unwrap(),
        );
    }

    // This test case is non-deterministic and could fail in rare cases.
    #[test]
    fn test_forest_options() {
        // These songs contains analysis of actual music. Recordings of Mozart's piano concerto no.
        // 19, Mozart's piano concerto no. 23, and tracks Miles Davis' "Kind Of Blue".
        let mozart_piano_19 = [
            Song {
                path: Path::new("path-to-first").to_path_buf(),
                analysis: Analysis::new([
                    0.5522649,
                    -0.8664422,
                    -0.81236243,
                    -0.9475107,
                    -0.76129013,
                    -0.90520144,
                    -0.8474938,
                    -0.8924977,
                    0.4956385,
                    0.5076021,
                    -0.5037869,
                    -0.61038315,
                    -0.47157913,
                    -0.48194122,
                    -0.36397678,
                    -0.6443357,
                    -0.9713509,
                    -0.9781786,
                    -0.98285836,
                    -0.983834,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-second").to_path_buf(),
                analysis: Analysis::new([
                    0.28091776,
                    -0.86352056,
                    -0.8175835,
                    -0.9497457,
                    -0.77833027,
                    -0.91656536,
                    -0.8477104,
                    -0.889485,
                    0.41879785,
                    0.45311546,
                    -0.6252063,
                    -0.6838323,
                    -0.5326821,
                    -0.63320035,
                    -0.5573063,
                    -0.7433087,
                    -0.9815542,
                    -0.98570454,
                    -0.98824924,
                    -0.9903612,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-third").to_path_buf(),
                analysis: Analysis::new([
                    0.5978223,
                    -0.84076107,
                    -0.7841455,
                    -0.886415,
                    -0.72486377,
                    -0.8015111,
                    -0.79157853,
                    -0.7739525,
                    0.517207,
                    0.535398,
                    -0.30007458,
                    -0.3972137,
                    -0.41319674,
                    -0.40709,
                    -0.32283908,
                    -0.5261506,
                    -0.9656949,
                    -0.9715169,
                    -0.97524375,
                    -0.9756616,
                ]),
                ..Default::default()
            },
        ];

        let kind_of_blue = [
            Song {
                path: Path::new("path-to-fourth").to_path_buf(),
                analysis: Analysis::new([
                    0.35871255,
                    -0.8679545,
                    -0.6833263,
                    -0.87800264,
                    -0.7235142,
                    -0.73546195,
                    -0.48577756,
                    -0.7732977,
                    0.51237035,
                    0.5379869,
                    -0.00649637,
                    -0.534671,
                    -0.5743973,
                    -0.5706258,
                    -0.43162197,
                    -0.6356183,
                    -0.97918683,
                    -0.98091763,
                    -0.9845511,
                    -0.98359185,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-fifth").to_path_buf(),
                analysis: Analysis::new([
                    0.2806753,
                    -0.85013694,
                    -0.66921043,
                    -0.8938313,
                    -0.6848732,
                    -0.75377,
                    -0.48747814,
                    -0.793482,
                    0.44880342,
                    0.461563,
                    -0.115760505,
                    -0.535959,
                    -0.5749081,
                    -0.55055845,
                    -0.37976396,
                    -0.538705,
                    -0.97972554,
                    -0.97890633,
                    -0.98290455,
                    -0.98231846,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-sixth").to_path_buf(),
                analysis: Analysis::new([
                    0.1545173,
                    -0.8991263,
                    -0.79770947,
                    -0.87425447,
                    -0.77811325,
                    -0.71051484,
                    -0.7369138,
                    -0.8515074,
                    0.387398,
                    0.42035806,
                    -0.30229717,
                    -0.624056,
                    -0.6458885,
                    -0.66208386,
                    -0.5866134,
                    -0.7613628,
                    -0.98656195,
                    -0.98821944,
                    -0.99072844,
                    -0.98729765,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-seventh").to_path_buf(),
                analysis: Analysis::new([
                    0.3853314,
                    -0.8475499,
                    -0.64330614,
                    -0.85917395,
                    -0.6624141,
                    -0.6356613,
                    -0.40988427,
                    -0.7480691,
                    0.45981812,
                    0.47096932,
                    -0.19245929,
                    -0.5228787,
                    -0.42246288,
                    -0.52656835,
                    -0.45702273,
                    -0.569838,
                    -0.97620565,
                    -0.97741324,
                    -0.9776932,
                    -0.98088175,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-eight").to_path_buf(),
                analysis: Analysis::new([
                    0.18926656,
                    -0.86667925,
                    -0.7294189,
                    -0.856192,
                    -0.7180501,
                    -0.66697484,
                    -0.6093149,
                    -0.82118326,
                    0.3888924,
                    0.42430043,
                    -0.4414854,
                    -0.6957753,
                    -0.7092425,
                    -0.68237424,
                    -0.55543846,
                    -0.77678657,
                    -0.98610276,
                    -0.98707336,
                    -0.99165493,
                    -0.99011236,
                ]),
                ..Default::default()
            },
        ];

        let mozart_piano_23 = [
            Song {
                path: Path::new("path-to-ninth").to_path_buf(),
                analysis: Analysis::new([
                    0.38328362,
                    -0.8752751,
                    -0.8165319,
                    -0.948534,
                    -0.77668643,
                    -0.9051969,
                    -0.8473458,
                    -0.88643366,
                    0.49641085,
                    0.5132351,
                    -0.41367024,
                    -0.5279201,
                    -0.46787983,
                    -0.49218357,
                    -0.42164963,
                    -0.6597451,
                    -0.97317076,
                    -0.9800342,
                    -0.9832096,
                    -0.98385316,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-tenth").to_path_buf(),
                analysis: Analysis::new([
                    0.4301988,
                    -0.89864063,
                    -0.84993315,
                    -0.9518692,
                    -0.8329567,
                    -0.9293889,
                    -0.8605237,
                    -0.8901016,
                    0.35011983,
                    0.3822446,
                    -0.6384951,
                    -0.7537949,
                    -0.5867439,
                    -0.57371,
                    -0.5662942,
                    -0.76130676,
                    -0.9845436,
                    -0.9833387,
                    -0.9902381,
                    -0.9905396,
                ]),
                ..Default::default()
            },
            Song {
                path: Path::new("path-to-eleventh").to_path_buf(),
                analysis: Analysis::new([
                    0.42334664,
                    -0.8632808,
                    -0.80268145,
                    -0.91918564,
                    -0.7522441,
                    -0.8721291,
                    -0.81877685,
                    -0.8166921,
                    0.53626525,
                    0.540933,
                    -0.34771818,
                    -0.45362264,
                    -0.35523874,
                    -0.4072432,
                    -0.25506926,
                    -0.553644,
                    -0.9624399,
                    -0.9706371,
                    -0.9753268,
                    -0.9764576,
                ]),
                ..Default::default()
            },
        ];

        let mut songs: Vec<&Song> = mozart_piano_19
            .iter()
            .chain(kind_of_blue.iter())
            .chain(mozart_piano_23.iter())
            .collect();

        // We train the algorithm on one of the Mozart concertos, and the expectation is that the
        // tracks from the Miles Davis record will end up last.
        let opts = ForestOptions {
            n_trees: 1000,
            sample_size: 200,
            max_tree_depth: None,
            extension_level: 10,
        };
        let playlist: Vec<_> = closest_to_songs(
            &mozart_piano_19.iter().collect::<Vec<&Song>>(),
            &mut songs,
            &opts,
        )
        .collect();
        for e in &kind_of_blue {
            assert!(playlist[playlist.len() - 5..].contains(&e));
        }
    }
}
