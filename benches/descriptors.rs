//! Benchmarks for all the descriptors used by bliss.

use bliss_audio::chroma::ChromaDesc;
use bliss_audio::decoder::PreAnalyzedSong;
use bliss_audio::misc::LoudnessDesc;
use bliss_audio::temporal::BPMDesc;
use bliss_audio::timbral::{SpectralDesc, ZeroCrossingRateDesc};
use common::{test_file, DecoderVTable};
use criterion::{criterion_group, criterion_main, Criterion};

mod common;

const SAMPLE_RATE: u32 = 22050;
fn song(decoder: DecoderVTable) -> PreAnalyzedSong {
    let path = test_file("s16_mono_22_5kHz.flac");
    (decoder.decode)(&path).unwrap()
}

fn bench_spectral_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptors/spectral descriptor");

    for decoder in common::DECODERS {
        let song = song(*decoder);
        let signal = song.sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || SpectralDesc::new(SAMPLE_RATE).unwrap(),
                |mut spectral_desc| {
                    spectral_desc.do_(&signal).unwrap();
                },
            );
        });
    }

    group.finish();
}

fn bench_zcr_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptors/zcr descriptor");

    for decoder in common::DECODERS {
        let song = song(*decoder);
        let signal = song.sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || ZeroCrossingRateDesc::new(SAMPLE_RATE),
                |mut zcr_desc| {
                    zcr_desc.do_(&signal);
                    zcr_desc.get_value();
                },
            );
        });
    }

    group.finish();
}

fn bench_bpm_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptors/bpm descriptor");

    for decoder in common::DECODERS {
        let song = song(*decoder);
        let signal = song.sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || BPMDesc::new(SAMPLE_RATE).unwrap(),
                |mut bpm_desc| {
                    bpm_desc.do_(&signal).unwrap();
                    bpm_desc.get_value();
                },
            );
        });
    }

    group.finish();
}

fn bench_loudness_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptors/loudness descriptor");

    for decoder in common::DECODERS {
        let song = song(*decoder);
        let signal = song.sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || LoudnessDesc::default(),
                |mut loudness_desc| {
                    loudness_desc.do_(&signal);
                    loudness_desc.get_value();
                },
            );
        });
    }

    group.finish();
}

fn bench_chroma_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptors/chroma descriptor");

    for decoder in common::DECODERS {
        let song = song(*decoder);
        let signal = song.sample_array;

        group.bench_with_input(decoder.name, &signal, |b, signal| {
            b.iter_with_setup(
                || ChromaDesc::new(SAMPLE_RATE, 12),
                |mut chroma_desc| {
                    chroma_desc.do_(&signal).unwrap();
                    chroma_desc.get_values().unwrap();
                },
            );
        });
    }

    group.finish();
}

criterion_group!(
    descriptors,
    bench_spectral_desc,
    bench_loudness_desc,
    bench_zcr_desc,
    bench_bpm_desc,
    bench_chroma_desc,
);
criterion_main!(descriptors);
