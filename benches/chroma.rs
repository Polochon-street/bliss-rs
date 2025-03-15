//! Additional benchmarks for the chroma module.

use std::fs::File;
use std::time::Duration;

use bliss_audio::chroma::bench::{
    chroma_filter, chroma_stft, estimate_tuning, normalize_feature_sequence, pip_track,
    pitch_tuning,
};
use bliss_audio::utils::bench::stft;
use common::{test_file, DECODERS};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{arr2, Array1, Array2};
use ndarray_npy::ReadNpyExt;

mod common;

fn bench_estimate_tuning(c: &mut Criterion) {
    let file = File::open(test_file("spectrum-chroma.npy")).unwrap();
    let arr = Array2::<f64>::read_npy(file).unwrap();

    c.bench_function("chroma/estimate_tuning", |b| {
        b.iter(|| estimate_tuning(22050, &arr, 2048, 0.01, 12).unwrap());
    });
}

fn bench_pitch_tuning(c: &mut Criterion) {
    let file = File::open(test_file("pitch-tuning.npy")).unwrap();
    let pitch = Array1::<f64>::read_npy(file).unwrap();

    c.bench_function("chroma/pitch_tuning", |b| {
        b.iter_with_setup(
            || pitch.clone(),
            |mut pitch| {
                pitch_tuning(&mut pitch, 0.05, 12).unwrap();
            },
        );
    });
}

fn bench_pip_track(c: &mut Criterion) {
    let file = File::open(test_file("spectrum-chroma.npy")).unwrap();
    let spectrum = Array2::<f64>::read_npy(file).unwrap();

    c.bench_function("chroma/pip_track", |b| {
        b.iter(|| {
            pip_track(22050, &spectrum, 2048).unwrap();
        });
    });
}

fn bench_chroma_filter(c: &mut Criterion) {
    c.bench_function("chroma/chroma_filter", |b| {
        b.iter(|| {
            chroma_filter(22050, 2048, 12, -0.1).unwrap();
        });
    });
}

fn bench_normalize_feature_sequence(c: &mut Criterion) {
    let array = arr2(&[[0.1, 0.3, 0.4], [1.1, 0.53, 1.01]]);
    c.bench_function("chroma/normalize_feature_sequence", |b| {
        b.iter(|| {
            normalize_feature_sequence(&array);
        });
    });
}

#[cfg(all(feature = "ffmpeg", feature = "symphonia-flac"))]
fn bench_chroma_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("chroma/chroma_stft");

    for decoder in DECODERS {
        let path = test_file("s16_mono_22_5kHz.flac");
        let signal = (decoder.decode)(&path).unwrap().sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || stft(&signal, 8192, 2205),
                |mut stft| {
                    chroma_stft(22050, &mut stft, 8192, 12, -0.04999999999999999).unwrap();
                },
            );
        });
    }

    group.finish();
}

criterion_group!(name = stft_bench; config = Criterion::default().measurement_time(Duration::from_secs(40)); targets = bench_chroma_stft);
criterion_group!(
    chroma,
    bench_estimate_tuning,
    bench_pitch_tuning,
    bench_pip_track,
    bench_chroma_filter,
    bench_normalize_feature_sequence,
);
criterion_main!(chroma, stft_bench);
