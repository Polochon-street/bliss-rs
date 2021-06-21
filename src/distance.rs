//! Module containing various distance metric functions.
//!
//! All of these functions are intended to be used with the
//! [custom_distance](Song::custom_distance) method, or with
//! [playlist_from_songs_custom_distance](Library::playlist_from_song_custom_distance).
//!
//! They will yield different styles of playlists, so don't hesitate to
//! experiment with them if the default (euclidean distance for now) doesn't
//! suit you.
use crate::NUMBER_FEATURES;
#[cfg(doc)]
use crate::{Library, Song};
use ndarray::{Array, Array1};

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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::arr1;

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
