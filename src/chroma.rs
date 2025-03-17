//! Chroma feature extraction module.
//!
//! Contains functions to compute the chromagram of a song, and
//! then from this chromagram extract the song's tone and mode
//! (minor / major).
extern crate noisy_float;

use crate::utils::stft;
use crate::utils::{hz_to_octs_inplace, Normalize};
use crate::{BlissError, BlissResult};
use ndarray::{arr1, arr2, concatenate, s, Array, Array1, Array2, Axis, Zip};
use ndarray_stats::interpolate::Midpoint;
use ndarray_stats::QuantileExt;
use noisy_float::prelude::*;

/**
 * General object holding the chroma descriptor.
 *
 * Current chroma descriptors are interval features (see
 * https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf).
 *
 * Contrary to the other descriptors that can be used with streaming
 * without consequences, this one performs better if the full song is used at
 * once.
 */
#[doc(hidden)]
pub struct ChromaDesc {
    sample_rate: u32,
    n_chroma: u32,
    values_chroma: Array2<f64>,
}

impl Normalize for ChromaDesc {
    const MAX_VALUE: f32 = 0.12;
    const MIN_VALUE: f32 = 0.;
}

impl ChromaDesc {
    pub const WINDOW_SIZE: usize = 8192;

    pub fn new(sample_rate: u32, n_chroma: u32) -> ChromaDesc {
        ChromaDesc {
            sample_rate,
            n_chroma,
            values_chroma: Array2::zeros((n_chroma as usize, 0)),
        }
    }

    /**
     * Compute and store the chroma of a signal.
     *
     * Passing a full song here once instead of streaming smaller parts of the
     * song will greatly improve accuracy.
     */
    pub fn do_(&mut self, signal: &[f32]) -> BlissResult<()> {
        let mut stft = stft(signal, ChromaDesc::WINDOW_SIZE, 2205);
        let tuning = estimate_tuning(self.sample_rate, &stft, ChromaDesc::WINDOW_SIZE, 0.01, 12)?;
        let chroma = chroma_stft(
            self.sample_rate,
            &mut stft,
            ChromaDesc::WINDOW_SIZE,
            self.n_chroma,
            tuning,
        )?;
        self.values_chroma = concatenate![Axis(1), self.values_chroma, chroma];
        Ok(())
    }

    /**
     * Get the song's interval features.
     *
     * Return the 6 pitch class set categories, as well as the major, minor,
     * diminished and augmented triads.
     *
     * See this paper https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf
     * for more information ("Timbre-invariant Audio Features for Style Analysis of Classical
     * Music").
     */
    pub fn get_values(&mut self) -> Vec<f32> {
        chroma_interval_features(&self.values_chroma)
            .mapv(|x| self.normalize(x as f32))
            .to_vec()
    }
}

// Functions below are Rust versions of python notebooks by AudioLabs Erlang
// (https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html)
fn chroma_interval_features(chroma: &Array2<f64>) -> Array1<f64> {
    let chroma = normalize_feature_sequence(&chroma.mapv(|x| (x * 15.).exp()));
    let templates = arr2(&[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]);
    let interval_feature_matrix = extract_interval_features(&chroma, &templates);
    interval_feature_matrix.mean_axis(Axis(1)).unwrap()
}

fn extract_interval_features(chroma: &Array2<f64>, templates: &Array2<i32>) -> Array2<f64> {
    let mut f_intervals: Array2<f64> = Array::zeros((chroma.shape()[1], templates.shape()[1]));
    for (template, mut f_interval) in templates
        .axis_iter(Axis(1))
        .zip(f_intervals.axis_iter_mut(Axis(1)))
    {
        for shift in 0..12 {
            let mut vec: Vec<i32> = template.to_vec();
            vec.rotate_right(shift);
            let rolled = arr1(&vec);
            let power = Zip::from(chroma.t())
                .and_broadcast(&rolled)
                .map_collect(|&f, &s| f.powi(s))
                .map_axis_mut(Axis(1), |x| x.product());
            f_interval += &power;
        }
    }
    f_intervals.t().to_owned()
}

fn normalize_feature_sequence(feature: &Array2<f64>) -> Array2<f64> {
    let mut normalized_sequence = feature.to_owned();
    for mut column in normalized_sequence.columns_mut() {
        let mut sum = column.mapv(|x| x.abs()).sum();
        if sum < 0.0001 {
            sum = 1.;
        }
        column /= sum;
    }

    normalized_sequence
}

// All the functions below are more than heavily inspired from
// librosa"s code: https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py#L1165
// chroma(22050, n_fft=5, n_chroma=12)
//
// Could be precomputed, but it takes very little time to compute it
// on the fly compared to the rest of the functions, and we'd lose the
// possibility to tweak parameters.
fn chroma_filter(
    sample_rate: u32,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> BlissResult<Array2<f64>> {
    let ctroct = 5.0;
    let octwidth = 2.;
    let n_chroma_float = f64::from(n_chroma);
    let n_chroma2 = (n_chroma_float / 2.0).round() as u32;
    let n_chroma2_float = f64::from(n_chroma2);

    let frequencies = Array::linspace(0., f64::from(sample_rate), n_fft + 1);

    let mut freq_bins = frequencies;
    hz_to_octs_inplace(&mut freq_bins, tuning, n_chroma);
    freq_bins.mapv_inplace(|x| x * n_chroma_float);
    freq_bins[0] = freq_bins[1] - 1.5 * n_chroma_float;

    let mut binwidth_bins = Array::ones(freq_bins.raw_dim());
    binwidth_bins.slice_mut(s![0..freq_bins.len() - 1]).assign(
        &(&freq_bins.slice(s![1..]) - &freq_bins.slice(s![..-1])).mapv(|x| {
            if x <= 1. {
                1.
            } else {
                x
            }
        }),
    );

    let mut d: Array2<f64> = Array::zeros((n_chroma as usize, (freq_bins).len()));
    for (idx, mut row) in d.rows_mut().into_iter().enumerate() {
        row.fill(idx as f64);
    }
    d = -d + &freq_bins;

    d.mapv_inplace(|x| {
        (x + n_chroma2_float + 10. * n_chroma_float) % n_chroma_float - n_chroma2_float
    });
    d = d / binwidth_bins;
    d.mapv_inplace(|x| (-0.5 * (2. * x) * (2. * x)).exp());

    let mut wts = d;
    // Normalize by computing the l2-norm over the columns
    for mut col in wts.columns_mut() {
        let mut sum = col.mapv(|x| x * x).sum().sqrt();
        if sum < f64::MIN_POSITIVE {
            sum = 1.;
        }
        col /= sum;
    }

    freq_bins.mapv_inplace(|x| (-0.5 * ((x / n_chroma_float - ctroct) / octwidth).powi(2)).exp());

    wts *= &freq_bins;

    // np.roll(), np bro
    let mut uninit: Vec<f64> = vec![0.; (wts).len()];
    unsafe {
        uninit.set_len(wts.len());
    }
    let mut b = Array::from(uninit)
        .into_shape(wts.dim())
        .map_err(|e| BlissError::AnalysisError(format!("in chroma: {e}")))?;
    b.slice_mut(s![-3.., ..]).assign(&wts.slice(s![..3, ..]));
    b.slice_mut(s![..-3, ..]).assign(&wts.slice(s![3.., ..]));

    wts = b;
    let non_aliased = 1 + n_fft / 2;
    Ok(wts.slice_move(s![.., ..non_aliased]))
}

fn pip_track(
    sample_rate: u32,
    spectrum: &Array2<f64>,
    n_fft: usize,
) -> BlissResult<(Vec<f64>, Vec<f64>)> {
    let sample_rate_float = f64::from(sample_rate);
    let fmin = 150.0_f64;
    let fmax = 4000.0_f64.min(sample_rate_float / 2.0);
    let threshold = 0.1;

    let fft_freqs = Array::linspace(0., sample_rate_float / 2., 1 + n_fft / 2);

    let length = spectrum.len_of(Axis(0));

    // TODO>1.0 Make this a bitvec when that won't mean depending on a crate
    let freq_mask = fft_freqs
        .iter()
        .map(|&f| (fmin <= f) && (f < fmax))
        .collect::<Vec<bool>>();

    let ref_value = spectrum.map_axis(Axis(0), |x| {
        let first: f64 = *x.first().expect("empty spectrum axis");
        let max = x.fold(first, |acc, &elem| if acc > elem { acc } else { elem });
        threshold * max
    });

    // There will be at most taken_columns * length elements in pitches / mags
    let taken_columns = freq_mask
        .iter()
        .fold(0, |acc, &x| if x { acc + 1 } else { acc });
    let mut pitches = Vec::with_capacity(taken_columns * length);
    let mut mags = Vec::with_capacity(taken_columns * length);

    let beginning = freq_mask
        .iter()
        .position(|&b| b)
        .ok_or_else(|| BlissError::AnalysisError("in chroma".to_string()))?;
    let end = freq_mask
        .iter()
        .rposition(|&b| b)
        .ok_or_else(|| BlissError::AnalysisError("in chroma".to_string()))?;

    let zipped = Zip::indexed(spectrum.slice(s![beginning..end - 3, ..]))
        .and(spectrum.slice(s![beginning + 1..end - 2, ..]))
        .and(spectrum.slice(s![beginning + 2..end - 1, ..]));

    // No need to handle the last column, since freq_mask[length - 1] is
    // always going to be `false` for 22.5kHz
    zipped.for_each(|(i, j), &before_elem, &elem, &after_elem| {
        if elem > ref_value[j] && after_elem <= elem && before_elem < elem {
            let avg = 0.5 * (after_elem - before_elem);
            let mut shift = 2. * elem - after_elem - before_elem;
            if shift.abs() < f64::MIN_POSITIVE {
                shift += 1.;
            }
            shift = avg / shift;
            pitches.push(((i + beginning + 1) as f64 + shift) * sample_rate_float / n_fft as f64);
            mags.push(elem + 0.5 * avg * shift);
        }
    });

    Ok((pitches, mags))
}

// Only use this with strictly positive `frequencies`.
fn pitch_tuning(
    frequencies: &mut Array1<f64>,
    resolution: f64,
    bins_per_octave: u32,
) -> BlissResult<f64> {
    if frequencies.is_empty() {
        return Ok(0.0);
    }
    hz_to_octs_inplace(frequencies, 0.0, 12);
    frequencies.mapv_inplace(|x| f64::from(bins_per_octave) * x % 1.0);

    // Put everything between -0.5 and 0.5.
    frequencies.mapv_inplace(|x| if x >= 0.5 { x - 1. } else { x });

    let indexes = ((frequencies.to_owned() - -0.5) / resolution).mapv(|x| x as usize);
    let mut counts: Array1<usize> = Array::zeros(((0.5 - -0.5) / resolution) as usize);
    for &idx in indexes.iter() {
        counts[idx] += 1;
    }
    let max_index = counts
        .argmax()
        .map_err(|e| BlissError::AnalysisError(format!("in chroma: {e}")))?;

    // Return the bin with the most reoccuring frequency.
    Ok((-50. + (100. * resolution * max_index as f64)) / 100.)
}

fn estimate_tuning(
    sample_rate: u32,
    spectrum: &Array2<f64>,
    n_fft: usize,
    resolution: f64,
    bins_per_octave: u32,
) -> BlissResult<f64> {
    let (pitch, mag) = pip_track(sample_rate, spectrum, n_fft)?;

    let (filtered_pitch, filtered_mag): (Vec<N64>, Vec<N64>) = pitch
        .iter()
        .zip(&mag)
        .filter(|(&p, _)| p > 0.)
        .map(|(x, y)| (n64(*x), n64(*y)))
        .unzip();

    if pitch.is_empty() {
        return Ok(0.);
    }

    let threshold: N64 = Array::from(filtered_mag.to_vec())
        .quantile_axis_mut(Axis(0), n64(0.5), &Midpoint)
        .map_err(|e| BlissError::AnalysisError(format!("in chroma: {e}")))?
        .into_scalar();
    let mut pitch = filtered_pitch
        .iter()
        .zip(&filtered_mag)
        .filter_map(|(&p, &m)| if m >= threshold { Some(p.into()) } else { None })
        .collect::<Array1<f64>>();
    pitch_tuning(&mut pitch, resolution, bins_per_octave)
}

fn chroma_stft(
    sample_rate: u32,
    spectrum: &mut Array2<f64>,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> Result<Array2<f64>, BlissError> {
    spectrum.par_mapv_inplace(|x| x * x);
    let mut raw_chroma = chroma_filter(sample_rate, n_fft, n_chroma, tuning)?;

    raw_chroma = raw_chroma.dot(spectrum);
    for mut row in raw_chroma.columns_mut() {
        let mut sum = row.mapv(|x| x.abs()).sum();
        if sum < f64::MIN_POSITIVE {
            sum = 1.;
        }
        row /= sum;
    }
    Ok(raw_chroma)
}

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench {
    //! Re-exports of private functions for benchmarking purposes.

    use ndarray::{Array1, Array2};

    use crate::BlissResult;

    #[inline(always)]
    pub fn normalize_feature_sequence(feature: &Array2<f64>) -> Array2<f64> {
        super::normalize_feature_sequence(feature)
    }

    #[inline(always)]
    pub fn chroma_filter(
        sample_rate: u32,
        n_fft: usize,
        n_chroma: u32,
        tuning: f64,
    ) -> BlissResult<Array2<f64>> {
        super::chroma_filter(sample_rate, n_fft, n_chroma, tuning)
    }

    #[inline(always)]
    pub fn pip_track(
        sample_rate: u32,
        spectrum: &Array2<f64>,
        n_fft: usize,
    ) -> BlissResult<(Vec<f64>, Vec<f64>)> {
        super::pip_track(sample_rate, spectrum, n_fft)
    }

    #[inline(always)]
    pub fn pitch_tuning(
        frequencies: &mut Array1<f64>,
        resolution: f64,
        bins_per_octave: u32,
    ) -> BlissResult<f64> {
        super::pitch_tuning(frequencies, resolution, bins_per_octave)
    }

    #[inline(always)]
    pub fn estimate_tuning(
        sample_rate: u32,
        spectrum: &Array2<f64>,
        n_fft: usize,
        resolution: f64,
        bins_per_octave: u32,
    ) -> BlissResult<f64> {
        super::estimate_tuning(sample_rate, spectrum, n_fft, resolution, bins_per_octave)
    }

    #[inline(always)]
    pub fn chroma_stft(
        sample_rate: u32,
        spectrum: &mut Array2<f64>,
        n_fft: usize,
        n_chroma: u32,
        tuning: f64,
    ) -> BlissResult<Array2<f64>> {
        super::chroma_stft(sample_rate, spectrum, n_fft, n_chroma, tuning)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::ffmpeg::FFmpegDecoder as Decoder;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::Decoder as DecoderTrait;
    #[cfg(feature = "ffmpeg")]
    use crate::utils::stft;
    #[cfg(feature = "ffmpeg")]
    use crate::SAMPLE_RATE;
    use ndarray::{arr1, arr2, Array2};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    #[cfg(feature = "ffmpeg")]
    use std::path::Path;

    #[test]
    fn test_chroma_interval_features() {
        let file = File::open("data/chroma.npy").unwrap();
        let chroma = Array2::<f64>::read_npy(file).unwrap();
        let features = chroma_interval_features(&chroma);
        let expected_features = arr1(&[
            0.03860284, 0.02185281, 0.04224379, 0.06385278, 0.07311148, 0.02512566, 0.00319899,
            0.00311308, 0.00107433, 0.00241861,
        ]);
        for (expected, actual) in expected_features.iter().zip(&features) {
            assert!(0.00000001 > (expected - actual.abs()));
        }
    }

    #[test]
    fn test_extract_interval_features() {
        let file = File::open("data/chroma-interval.npy").unwrap();
        let chroma = Array2::<f64>::read_npy(file).unwrap();
        let templates = arr2(&[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]);

        let file = File::open("data/interval-feature-matrix.npy").unwrap();
        let expected_interval_features = Array2::<f64>::read_npy(file).unwrap();

        let interval_features = extract_interval_features(&chroma, &templates);
        for (expected, actual) in expected_interval_features
            .iter()
            .zip(interval_features.iter())
        {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }

    #[test]
    fn test_normalize_feature_sequence() {
        let array = arr2(&[[0.1, 0.3, 0.4, 0.], [1.1, 0.53, 1.01, 0.]]);
        let expected_array = arr2(&[
            [0.08333333, 0.36144578, 0.28368794, 0.],
            [0.91666667, 0.63855422, 0.71631206, 0.],
        ]);

        let normalized_array = normalize_feature_sequence(&array);

        assert!(!array.is_empty() && !expected_array.is_empty());

        for (expected, actual) in normalized_array.iter().zip(expected_array.iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_chroma_desc() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        let mut chroma_desc = ChromaDesc::new(SAMPLE_RATE, 12);
        chroma_desc.do_(&song.sample_array).unwrap();
        let expected_values = vec![
            -0.35661936,
            -0.63578653,
            -0.29593682,
            0.06421304,
            0.21852458,
            -0.581239,
            -0.9466835,
            -0.9481153,
            -0.9820945,
            -0.95968974,
        ];
        for (expected, actual) in expected_values.iter().zip(chroma_desc.get_values().iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_chroma_stft_decode() {
        let signal = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac"))
            .unwrap()
            .sample_array;
        let mut stft = stft(&signal, 8192, 2205);

        let file = File::open("data/chroma.npy").unwrap();
        let expected_chroma = Array2::<f64>::read_npy(file).unwrap();

        let chroma = chroma_stft(22050, &mut stft, 8192, 12, -0.04999999999999999).unwrap();

        assert!(!chroma.is_empty() && !expected_chroma.is_empty());

        for (expected, actual) in expected_chroma.iter().zip(chroma.iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }

    #[test]
    fn test_estimate_tuning() {
        let file = File::open("data/spectrum-chroma.npy").unwrap();
        let arr = Array2::<f64>::read_npy(file).unwrap();

        let tuning = estimate_tuning(22050, &arr, 2048, 0.01, 12).unwrap();
        assert!(0.000001 > (-0.09999999999999998 - tuning).abs());
    }

    #[test]
    fn test_chroma_estimate_tuning_empty_fix() {
        assert!(0. == estimate_tuning(22050, &Array2::zeros((8192, 1)), 8192, 0.01, 12).unwrap());
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_estimate_tuning_decode() {
        let signal = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac"))
            .unwrap()
            .sample_array;
        let stft = stft(&signal, 8192, 2205);

        let tuning = estimate_tuning(22050, &stft, 8192, 0.01, 12).unwrap();
        assert!(0.000001 > (-0.04999999999999999 - tuning).abs());
    }

    #[test]
    fn test_pitch_tuning() {
        let file = File::open("data/pitch-tuning.npy").unwrap();
        let mut pitch = Array1::<f64>::read_npy(file).unwrap();

        assert_eq!(-0.1, pitch_tuning(&mut pitch, 0.05, 12).unwrap());
    }

    #[test]
    fn test_pitch_tuning_no_frequencies() {
        let mut frequencies = arr1(&[]);
        assert_eq!(0.0, pitch_tuning(&mut frequencies, 0.05, 12).unwrap());
    }

    #[test]
    fn test_pip_track() {
        let file = File::open("data/spectrum-chroma.npy").unwrap();
        let spectrum = Array2::<f64>::read_npy(file).unwrap();

        let mags_file = File::open("data/spectrum-chroma-mags.npy").unwrap();
        let expected_mags = Array1::<f64>::read_npy(mags_file).unwrap();

        let pitches_file = File::open("data/spectrum-chroma-pitches.npy").unwrap();
        let expected_pitches = Array1::<f64>::read_npy(pitches_file).unwrap();

        let (mut pitches, mut mags) = pip_track(22050, &spectrum, 2048).unwrap();
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (expected_pitches, actual_pitches) in expected_pitches.iter().zip(pitches.iter()) {
            assert!(0.00000001 > (expected_pitches - actual_pitches).abs());
        }
        for (expected_mags, actual_mags) in expected_mags.iter().zip(mags.iter()) {
            assert!(0.00000001 > (expected_mags - actual_mags).abs());
        }
    }

    #[test]
    fn test_chroma_filter() {
        let file = File::open("data/chroma-filter.npy").unwrap();
        let expected_filter = Array2::<f64>::read_npy(file).unwrap();

        let filter = chroma_filter(22050, 2048, 12, -0.1).unwrap();

        for (expected, actual) in expected_filter.iter().zip(filter.iter()) {
            assert!(0.000000001 > (expected - actual).abs());
        }
    }
}
