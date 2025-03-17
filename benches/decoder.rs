//! Benchmarks for the decoders

use common::{test_file, DECODERS};
use criterion::{criterion_group, criterion_main, Criterion};

mod common;

fn bench_decode_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/decode_mono");
    let path = test_file("s16_mono_22_5kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_decode_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/decode_stereo");
    let path = test_file("s16_stereo_22_5kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_resample_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/resample_mono");
    let path = test_file("s32_mono_44_1_kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_resample_multi(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/resample_multi");
    let path = test_file("s32_stereo_44_1_kHz.flac");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_mp3(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/mp3");
    let path = test_file("s32_stereo_44_1_kHz.mp3");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_long_song(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder/long_song");
    let path = test_file("5_mins_of_noise_stereo_48kHz.ogg");

    for decoder in DECODERS {
        group.bench_with_input(decoder.name, &path, |b, path| {
            b.iter_with_setup(
                || decoder.decode, // done in setup so we aren't timing the vtable lookup
                |decoder_fn| {
                    decoder_fn(path).unwrap();
                },
            );
        });
    }

    group.finish();
}

criterion_group!(name = long_song; config = Criterion::default().measurement_time(std::time::Duration::from_secs(30)); targets = bench_long_song);
criterion_group!(
    name = decoders;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = bench_decode_mono,
    bench_decode_stereo,
    bench_resample_mono,
    bench_resample_multi,
    bench_mp3
);
criterion_main!(decoders, long_song);
