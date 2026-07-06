//! Benchmarks for the full analysis pipeline

use common::{test_file, DECODERS};
use criterion::{criterion_group, criterion_main, Criterion};

mod common;

fn bench_analysis_pipeline_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/mono");
    let path = test_file("s16_mono_22_5kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_analysis_pipeline_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/stereo");
    let path = test_file("s16_stereo_22_5kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_analysis_pipeline_resample_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/resample_mono");
    let path = test_file("s32_mono_44_1_kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_analysis_pipeline_resample_multi(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/resample_multi");
    let path = test_file("s32_stereo_44_1_kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_analysis_pipeline_mp3(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/mp3");
    let path = test_file("s32_stereo_44_1_kHz.mp3");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_analysis_pipeline_long_song(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_pipeline/long_song");
    let path = test_file("5_mins_of_noise_stereo_48kHz.ogg");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.analyze,
                |analyze_fn| {
                    analyze_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

criterion_group!(name = long_song; config = Criterion::default().measurement_time(std::time::Duration::from_secs(90)); targets = bench_analysis_pipeline_long_song);
criterion_group!(
    name = pipelines;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = bench_analysis_pipeline_mono,
    bench_analysis_pipeline_stereo,
    bench_analysis_pipeline_resample_mono,
    bench_analysis_pipeline_resample_multi,
    bench_analysis_pipeline_mp3
);
criterion_main!(pipelines, long_song);
