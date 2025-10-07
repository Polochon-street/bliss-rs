extern crate rustfft;
use ndarray::{arr1, s, Array, Array1, Array2};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
#[cfg(feature = "ffmpeg")]
extern crate ffmpeg_next as ffmpeg;
use log::warn;
use std::f32::consts::PI;

pub(crate) fn reflect_pad(array: &[f32], pad: usize) -> Vec<f32> {
    let prefix = array[1..=pad].iter().rev().copied().collect::<Vec<f32>>();
    let suffix = array[(array.len() - 2) - pad + 1..array.len() - 1]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<f32>>();
    let mut output = Vec::with_capacity(prefix.len() + array.len() + suffix.len());

    output.extend(prefix);
    output.extend(array);
    output.extend(suffix);
    output
}

pub(crate) fn stft(signal: &[f32], window_length: usize, hop_length: usize) -> Array2<f64> {
    // Take advantage of raw-major order to have contiguous window for the
    // `assign`, reversing the axes to have the expected shape at the end only.
    let mut stft = Array2::zeros((
        (signal.len() as f32 / hop_length as f32).ceil() as usize,
        window_length / 2 + 1,
    ));
    let signal = reflect_pad(signal, window_length / 2);

    // Periodic, so window_size + 1
    let mut hann_window = Array::zeros(window_length + 1);
    for n in 0..window_length {
        hann_window[[n]] = 0.5 - 0.5 * f32::cos(2. * n as f32 * PI / (window_length as f32));
    }
    hann_window = hann_window.slice_move(s![0..window_length]);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_length);

    for (window, mut stft_col) in signal
        .windows(window_length)
        .step_by(hop_length)
        .zip(stft.rows_mut())
    {
        let mut signal = (arr1(window) * &hann_window).mapv(|x| Complex::new(x, 0.));
        match signal.as_slice_mut() {
            Some(s) => fft.process(s),
            None => {
                warn!("non-contiguous slice found for stft; expect slow performances.");
                fft.process(&mut signal.to_vec());
            }
        };
        stft_col.assign(
            &signal
                .slice(s![..window_length / 2 + 1])
                .mapv(|x| (x.re * x.re + x.im * x.im).sqrt() as f64),
        );
    }
    stft.permuted_axes((1, 0))
}

pub(crate) fn mean<T: Clone + Into<f32>>(input: &[T]) -> f32 {
    input.iter().map(|x| x.clone().into()).sum::<f32>() / input.len() as f32
}

pub(crate) trait Normalize {
    const MAX_VALUE: f32;
    const MIN_VALUE: f32;

    fn normalize(&self, value: f32) -> f32 {
        2. * (value - Self::MIN_VALUE) / (Self::MAX_VALUE - Self::MIN_VALUE) - 1.
    }
}

// Essentia algorithm
// https://github.com/MTG/essentia/blob/master/src/algorithms/temporal/zerocrossingrate.cpp
pub(crate) fn number_crossings(input: &[f32]) -> u32 {
    let mut crossings = 0;

    let mut was_positive = input[0] > 0.;

    for &sample in input {
        let is_positive = sample > 0.;
        if was_positive != is_positive {
            crossings += 1;
            was_positive = is_positive;
        }
    }

    crossings
}

// Only works for input of size 256 (or at least of size a multiple
// of 8), with values belonging to [0; 2^65].
// This finely optimized geometric mean courtesy of
// Jacques-Henri Jourdan (https://jhjourdan.mketjh.fr/)
pub(crate) fn geometric_mean(input: &[f32]) -> f32 {
    let mut exponents: i32 = 0;
    let mut mantissas: f64 = 1.;
    for ch in input.chunks_exact(8) {
        let mut m = (ch[0] as f64 * ch[1] as f64) * (ch[2] as f64 * ch[3] as f64);
        m *= 3.273390607896142e150; // 2^500 : avoid underflows and denormals
        m *= (ch[4] as f64 * ch[5] as f64) * (ch[6] as f64 * ch[7] as f64);
        if m == 0. {
            return 0.;
        }
        exponents += (m.to_bits() >> 52) as i32;
        mantissas *= f64::from_bits((m.to_bits() & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000);
    }

    let n = input.len() as u32;
    (((mantissas as f32).log2() + exponents as f32) / n as f32 - (1023. + 500.) / 8.).exp2()
}

pub(crate) fn hz_to_octs_inplace(
    frequencies: &mut Array1<f64>,
    tuning: f64,
    bins_per_octave: u32,
) -> &mut Array1<f64> {
    let a440 = 440.0 * 2_f64.powf(tuning / f64::from(bins_per_octave));

    *frequencies /= a440 / 16.;
    frequencies.mapv_inplace(f64::log2);
    frequencies
}

#[allow(dead_code)]
pub(crate) fn convolve(input: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let mut common_length = input.len() + kernel.len();
    if !common_length.is_multiple_of(2) {
        common_length -= 1;
    }
    let mut padded_input = Array::from_elem(common_length, Complex::zero());
    padded_input
        .slice_mut(s![..input.len()])
        .assign(&input.mapv(|x| Complex::new(x, 0.)));
    let mut padded_kernel = Array::from_elem(common_length, Complex::zero());
    padded_kernel
        .slice_mut(s![..kernel.len()])
        .assign(&kernel.mapv(|x| Complex::new(x, 0.)));

    let mut planner = FftPlanner::new();
    let forward = planner.plan_fft_forward(common_length);
    forward.process(padded_input.as_slice_mut().unwrap());
    forward.process(padded_kernel.as_slice_mut().unwrap());

    let mut multiplication = padded_input * padded_kernel;

    let mut planner = FftPlanner::new();
    let back = planner.plan_fft_inverse(common_length);
    back.process(multiplication.as_slice_mut().unwrap());

    let multiplication_length = multiplication.len() as f64;
    let multiplication = multiplication
        .slice_move(s![
            (kernel.len() - 1) / 2..(kernel.len() - 1) / 2 + input.len()
        ])
        .mapv(|x| x.re);
    multiplication / multiplication_length
}

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench {
    //! Re-exports of private functions for benchmarking purposes.
    use ndarray::{Array1, Array2};

    #[inline(always)]
    pub fn convolve(input: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
        super::convolve(input, kernel)
    }

    #[inline(always)]
    pub fn geometric_mean(input: &[f32]) -> f32 {
        super::geometric_mean(input)
    }

    #[inline(always)]
    pub fn reflect_pad(array: &[f32], pad: usize) -> Vec<f32> {
        super::reflect_pad(array, pad)
    }

    #[inline(always)]
    pub fn stft(signal: &[f32], window_length: usize, hop_length: usize) -> Array2<f64> {
        super::stft(signal, window_length, hop_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::ffmpeg::FFmpegDecoder as Decoder;
    #[cfg(feature = "ffmpeg")]
    use crate::song::decoder::Decoder as DecoderTrait;
    #[cfg(feature = "ffmpeg")]
    use ndarray::Array2;
    use ndarray::{arr1, Array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    #[cfg(feature = "ffmpeg")]
    use std::path::Path;

    #[test]
    fn test_convolve() {
        let file = File::open("data/convolve.npy").unwrap();
        let expected_convolve = Array1::<f64>::read_npy(file).unwrap();
        let input: Array1<f64> = Array::range(0., 1000., 0.5);
        let kernel: Array1<f64> = Array::ones(100);

        let output = convolve(&input, &kernel);
        for (expected, actual) in expected_convolve.iter().zip(output.iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }

        let input: Array1<f64> = Array::range(0., 1000., 0.5);
        let file = File::open("data/convolve_odd.npy").unwrap();
        let expected_convolve = Array1::<f64>::read_npy(file).unwrap();
        let kernel: Array1<f64> = Array::ones(99);

        let output = convolve(&input, &kernel);
        for (expected, actual) in expected_convolve.iter().zip(output.iter()) {
            assert!(0.0000001 > (expected - actual).abs());
        }
    }

    #[test]
    fn test_mean() {
        let numbers = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(2.0, mean(&numbers));
    }

    #[test]
    fn test_geometric_mean() {
        let numbers = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(0.0, geometric_mean(&numbers));

        let numbers = vec![4.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 2.0];
        assert!(0.0001 > (2.0 - geometric_mean(&numbers)).abs());

        // never going to happen, but just in case
        let numbers = vec![256., 4.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0];
        assert!(0.0001 > (3.668016172818685 - geometric_mean(&numbers)).abs());

        let subnormal = vec![4.0, 2.0, 1.0, 4.0, 2.0, 1.0, 2.0, 1.0e-40_f32];
        assert!(0.0001 > (1.8340080864093417e-05 - geometric_mean(&subnormal)).abs());

        let maximum = vec![2_f32.powi(65); 256];
        assert!(0.0001 > (2_f32.powi(65) - geometric_mean(&maximum).abs()));

        let input = [
            0.024454033,
            0.08809689,
            0.44554362,
            0.82753503,
            0.15822093,
            1.4442245,
            3.6971385,
            3.6789556,
            1.5981572,
            1.0172718,
            1.4436096,
            3.1457102,
            2.7641108,
            0.8395235,
            0.24896829,
            0.07063173,
            0.3554194,
            0.3520014,
            0.7973651,
            0.6619708,
            0.784104,
            0.8767957,
            0.28738266,
            0.04884128,
            0.3227065,
            0.33490747,
            0.18588875,
            0.13544942,
            0.14017746,
            0.11181582,
            0.15263161,
            0.22199312,
            0.056798387,
            0.08389257,
            0.07000965,
            0.20290329,
            0.37071738,
            0.23154318,
            0.02334859,
            0.013220183,
            0.035887096,
            0.02950549,
            0.09033857,
            0.17679504,
            0.08142187,
            0.0033268086,
            0.012269007,
            0.016257336,
            0.027027424,
            0.017253408,
            0.017230038,
            0.021678915,
            0.018645158,
            0.005417136,
            0.0066501745,
            0.020159671,
            0.026623515,
            0.0051667937,
            0.016880387,
            0.0099352235,
            0.011079361,
            0.013200151,
            0.0053205723,
            0.0050702896,
            0.008130498,
            0.009006041,
            0.0036024998,
            0.0064403876,
            0.004656151,
            0.0025131858,
            0.0030845597,
            0.008722531,
            0.017871628,
            0.022656294,
            0.017539924,
            0.0094395885,
            0.00308572,
            0.0013586166,
            0.0027467872,
            0.0054130103,
            0.004140312,
            0.00014358714,
            0.0013718408,
            0.004472961,
            0.003769122,
            0.0032591296,
            0.00363724,
            0.0024453322,
            0.00059036893,
            0.00064789865,
            0.001745297,
            0.0008671655,
            0.0021562362,
            0.0010756068,
            0.0020091995,
            0.0015373885,
            0.0009846204,
            0.00029200249,
            0.0009211624,
            0.0005351118,
            0.0014912765,
            0.0020651375,
            0.00066112226,
            0.00085005426,
            0.0019005901,
            0.0006395845,
            0.002262803,
            0.0030940182,
            0.0020891617,
            0.001215059,
            0.0013114084,
            0.000470959,
            0.0006654807,
            0.00143032,
            0.0017918893,
            0.00086320075,
            0.0005604455,
            0.00082841754,
            0.0006694539,
            0.000822765,
            0.0006165758,
            0.001189319,
            0.0007300245,
            0.0006237481,
            0.0012076444,
            0.0014746742,
            0.002033916,
            0.0015001699,
            0.00052051,
            0.00044564332,
            0.00055846275,
            0.00089778664,
            0.00080524705,
            0.00072653644,
            0.0006730526,
            0.0009940645,
            0.0011093937,
            0.0012950997,
            0.0009826822,
            0.0008766518,
            0.0016549287,
            0.00092906435,
            0.00029130623,
            0.00025049047,
            0.00022848802,
            0.00026967315,
            0.00023737509,
            0.0009694061,
            0.0010638118,
            0.00079342886,
            0.00059083506,
            0.0004763899,
            0.0009516641,
            0.00069223146,
            0.0005571137,
            0.0008517697,
            0.0010710277,
            0.0006102439,
            0.00074687623,
            0.00084989844,
            0.0004958062,
            0.000526994,
            0.00021524922,
            0.000096684314,
            0.0006545544,
            0.0012206973,
            0.0012103583,
            0.00092045433,
            0.0009248435,
            0.0008121284,
            0.00023953256,
            0.0009318224,
            0.0010439663,
            0.00048373415,
            0.00029895222,
            0.0004844254,
            0.0006668295,
            0.0009983985,
            0.0008604897,
            0.00018315323,
            0.0003091808,
            0.0005426462,
            0.0010403915,
            0.0007554566,
            0.0002846017,
            0.0006009793,
            0.0007650569,
            0.00056281046,
            0.00034661655,
            0.00023622432,
            0.0005987106,
            0.00029568427,
            0.00038697806,
            0.000584258,
            0.0005670976,
            0.0006136444,
            0.0005645493,
            0.00023538452,
            0.0002855746,
            0.00038535293,
            0.00043193565,
            0.0007312465,
            0.0006030728,
            0.0010331308,
            0.0011952162,
            0.0008245007,
            0.00042218363,
            0.00082176016,
            0.001132246,
            0.00089140673,
            0.0006351588,
            0.00037268156,
            0.00023035,
            0.0006286493,
            0.0008061599,
            0.00066162215,
            0.00022713901,
            0.00021469496,
            0.0006654577,
            0.000513901,
            0.00039176678,
            0.0010790947,
            0.0007353637,
            0.00017166573,
            0.00043964887,
            0.0002951453,
            0.00017704708,
            0.00018295897,
            0.00092653604,
            0.0008324083,
            0.0008041684,
            0.0011318093,
            0.0011871496,
            0.0008069488,
            0.00062862475,
            0.0005913861,
            0.0004721823,
            0.00016365231,
            0.00017787657,
            0.00042536375,
            0.0005736993,
            0.00043467924,
            0.00009028294,
            0.00017257355,
            0.0005019574,
            0.0006147168,
            0.0002167805,
            0.0001489743,
            0.000055081473,
            0.00029626413,
            0.00037805567,
            0.00014736196,
            0.00026251364,
            0.00016211842,
            0.0001853477,
            0.0001387354,
        ];
        assert!(0.00000001 > (0.0025750597 - geometric_mean(&input)).abs());
    }

    #[test]
    fn test_hz_to_octs_inplace() {
        let mut frequencies = arr1(&[32., 64., 128., 256.]);
        let expected = arr1(&[0.16864029, 1.16864029, 2.16864029, 3.16864029]);

        hz_to_octs_inplace(&mut frequencies, 0.5, 10)
            .iter()
            .zip(expected.iter())
            .for_each(|(x, y)| assert!(0.0001 > (x - y).abs()));
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_compute_stft() {
        let file = File::open("data/librosa-stft.npy").unwrap();
        let expected_stft = Array2::<f32>::read_npy(file).unwrap().mapv(|x| x as f64);

        let song = Decoder::decode(Path::new("data/piano.flac")).unwrap();

        let stft = stft(&song.sample_array, 2048, 512);

        assert!(!stft.is_empty() && !expected_stft.is_empty());
        for (expected, actual) in expected_stft.iter().zip(stft.iter()) {
            assert!(0.0001 > (expected - actual).abs());
        }
    }

    #[test]
    fn test_reflect_pad() {
        let array = Array::range(0., 100000., 1.);

        let output = reflect_pad(array.as_slice().unwrap(), 3);
        assert_eq!(&output[..4], &[3.0, 2.0, 1.0, 0.]);
        assert_eq!(&output[3..100003], array.to_vec());
        assert_eq!(&output[100003..100006], &[99998.0, 99997.0, 99996.0]);
    }
}
