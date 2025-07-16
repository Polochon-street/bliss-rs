//! The default decoder module. It uses [ffmpeg](https://ffmpeg.org/) in
//! order to decode and resample songs. A very good choice for 99% of
//! the users.

use crate::decoder::{Decoder, PreAnalyzedSong};
use crate::{BlissError, BlissResult, CHANNELS, SAMPLE_RATE};
use ::log::warn;
use ffmpeg_next;
use ffmpeg_next::codec::threading::{Config, Type as ThreadingType};
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::error::Error;
use ffmpeg_next::util::error::EINVAL;
use ffmpeg_next::util::format::sample::{Sample, Type};
use ffmpeg_next::util::frame::audio::Audio;
use ffmpeg_next::util::log;
use ffmpeg_next::util::log::level::Level;
use ffmpeg_next::{media, util};
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;

use std::path::Path;

/// The actual FFmpeg decoder.
///
/// To use it, one might write `use FFmpegDecoder as Decoder;`,
/// `use super::decoder::Decoder as DecoderTrait;`, and then use
/// `Decoder::song_from_path`
pub struct FFmpegDecoder;

struct SendChannelLayout(ChannelLayout);
// Safe because the other thread just reads the channel layout
unsafe impl Send for SendChannelLayout {}

impl FFmpegDecoder {
    fn resample_frame(
        rx: Receiver<Audio>,
        in_codec_format: Sample,
        sent_in_channel_layout: SendChannelLayout,
        in_rate: u32,
        mut sample_array: Vec<f32>,
        empty_in_channel_layout: bool,
    ) -> BlissResult<Vec<f32>> {
        let in_channel_layout = sent_in_channel_layout.0;
        let mut resample_context = ffmpeg_next::software::resampling::context::Context::get(
            in_codec_format,
            in_channel_layout,
            in_rate,
            Sample::F32(Type::Packed),
            ffmpeg_next::util::channel_layout::ChannelLayout::MONO,
            SAMPLE_RATE,
        )
        .map_err(|e| {
            BlissError::DecodingError(format!(
                "while trying to allocate resampling context: {e:?}",
            ))
        })?;

        let mut resampled = ffmpeg_next::frame::Audio::empty();
        let mut something_happened = false;
        for mut decoded in rx.iter() {
            #[cfg(not(feature = "ffmpeg_7_0"))]
            let is_channel_layout_empty = decoded.channel_layout() == ChannelLayout::empty();
            #[cfg(feature = "ffmpeg_7_0")]
            let is_channel_layout_empty = decoded.channel_layout().is_empty();

            // If the decoded layout is empty, it means we forced the
            // "in_channel_layout" to something default, not that
            // the format is wrong.
            if empty_in_channel_layout && is_channel_layout_empty {
                decoded.set_channel_layout(in_channel_layout);
            } else if in_codec_format != decoded.format()
                || (in_channel_layout != decoded.channel_layout())
                || in_rate != decoded.rate()
            {
                warn!("received decoded packet with wrong format; file might be corrupted.");
                continue;
            }
            something_happened = true;
            resampled = ffmpeg_next::frame::Audio::empty();
            resample_context
                .run(&decoded, &mut resampled)
                .map_err(|e| {
                    BlissError::DecodingError(format!("while trying to resample song: {e:?}"))
                })?;
            FFmpegDecoder::push_to_sample_array(&resampled, &mut sample_array);
        }
        if !something_happened {
            return Ok(sample_array);
        }
        // TODO when ffmpeg-next will be active again: shouldn't we allocate
        // `resampled` again?
        loop {
            match resample_context.flush(&mut resampled).map_err(|e| {
                BlissError::DecodingError(format!("while trying to resample song: {e:?}"))
            })? {
                Some(_) => {
                    FFmpegDecoder::push_to_sample_array(&resampled, &mut sample_array);
                }
                None => {
                    if resampled.samples() == 0 {
                        break;
                    }
                    FFmpegDecoder::push_to_sample_array(&resampled, &mut sample_array);
                }
            };
        }
        Ok(sample_array)
    }

    fn push_to_sample_array(frame: &ffmpeg_next::frame::Audio, sample_array: &mut Vec<f32>) {
        if frame.samples() == 0 {
            return;
        }
        // Account for the padding
        let actual_size = util::format::sample::Buffer::size(
            Sample::F32(Type::Packed),
            CHANNELS,
            frame.samples(),
            false,
        );
        let f32_frame: Vec<f32> = frame.data(0)[..actual_size]
            .chunks_exact(4)
            .map(|x| {
                let mut a: [u8; 4] = [0; 4];
                a.copy_from_slice(x);
                f32::from_le_bytes(a)
            })
            .collect();
        sample_array.extend_from_slice(&f32_frame);
    }
}

impl Decoder for FFmpegDecoder {
    fn decode(path: &Path) -> BlissResult<PreAnalyzedSong> {
        ffmpeg_next::init().map_err(|e| {
            BlissError::DecodingError(format!(
                "ffmpeg init error while decoding file '{}': {:?}.",
                path.display(),
                e
            ))
        })?;
        log::set_level(Level::Quiet);
        let mut song = PreAnalyzedSong {
            path: path.into(),
            ..Default::default()
        };
        let mut ictx = ffmpeg_next::format::input(&path).map_err(|e| {
            BlissError::DecodingError(format!(
                "while opening format for file '{}': {:?}.",
                path.display(),
                e
            ))
        })?;
        let (mut decoder, stream, expected_sample_number) = {
            let input = ictx.streams().best(media::Type::Audio).ok_or_else(|| {
                BlissError::DecodingError(format!(
                    "No audio stream found for file '{}'.",
                    path.display()
                ))
            })?;
            let mut context = ffmpeg_next::codec::context::Context::from_parameters(
                input.parameters(),
            )
            .map_err(|e| {
                BlissError::DecodingError(format!(
                    "Could not load the codec context for file '{}': {:?}",
                    path.display(),
                    e
                ))
            })?;
            context.set_threading(Config {
                kind: ThreadingType::Frame,
                count: 0,
                #[cfg(not(feature = "ffmpeg_6_0"))]
                safe: true,
            });
            let decoder = context.decoder().audio().map_err(|e| {
                BlissError::DecodingError(format!(
                    "when finding decoder for file '{}': {:?}.",
                    path.display(),
                    e
                ))
            })?;

            // Add SAMPLE_RATE to have one second margin to avoid reallocating if
            // the duration is slightly more than estimated
            // TODO>1.0 another way to get the exact number of samples is to decode
            // everything once, compute the real number of samples from that,
            // allocate the array with that number, and decode again. Check
            // what's faster between reallocating, and just have one second
            // leeway.
            let expected_sample_number = (SAMPLE_RATE as f32 * input.duration() as f32
                / input.time_base().denominator() as f32)
                .ceil()
                + SAMPLE_RATE as f32;
            (decoder, input.index(), expected_sample_number)
        };
        let sample_array: Vec<f32> = Vec::with_capacity(expected_sample_number as usize);
        if let Some(title) = ictx.metadata().get("title") {
            song.title = match title {
                "" => None,
                t => Some(t.to_string()),
            };
        };
        if let Some(artist) = ictx.metadata().get("artist") {
            song.artist = match artist {
                "" => None,
                a => Some(a.to_string()),
            };
        };
        if let Some(album) = ictx.metadata().get("album") {
            song.album = match album {
                "" => None,
                a => Some(a.to_string()),
            };
        };
        if let Some(genre) = ictx.metadata().get("genre") {
            song.genre = match genre {
                "" => None,
                g => Some(g.to_string()),
            };
        };
        if let Some(track_number) = ictx.metadata().get("track") {
            song.track_number = match track_number {
                "" => None,
                t => t
                    .parse::<i32>()
                    .ok()
                    .or_else(|| t.split_once('/').and_then(|(n, _)| n.parse::<i32>().ok())),
            };
        };
        if let Some(disc_number) = ictx.metadata().get("disc") {
            song.disc_number = match disc_number {
                "" => None,
                t => t
                    .parse::<i32>()
                    .ok()
                    .or_else(|| t.split_once('/').and_then(|(n, _)| n.parse::<i32>().ok())),
            };
        };
        if let Some(album_artist) = ictx.metadata().get("album_artist") {
            song.album_artist = match album_artist {
                "" => None,
                t => Some(t.to_string()),
            };
        };

        #[cfg(not(feature = "ffmpeg_7_0"))]
        let is_channel_layout_empty = decoder.channel_layout() == ChannelLayout::empty();
        #[cfg(feature = "ffmpeg_7_0")]
        let is_channel_layout_empty = decoder.channel_layout().is_empty();

        let (empty_in_channel_layout, in_channel_layout) = {
            if is_channel_layout_empty {
                (true, ChannelLayout::default(decoder.channels().into()))
            } else {
                (false, decoder.channel_layout())
            }
        };
        decoder.set_channel_layout(in_channel_layout);

        let in_channel_layout_to_send = SendChannelLayout(in_channel_layout);

        let (tx, rx) = mpsc::channel();
        let in_codec_format = decoder.format();
        let in_codec_rate = decoder.rate();
        let child = thread::spawn(move || {
            FFmpegDecoder::resample_frame(
                rx,
                in_codec_format,
                in_channel_layout_to_send,
                in_codec_rate,
                sample_array,
                empty_in_channel_layout,
            )
        });
        for (s, packet) in ictx.packets() {
            if s.index() != stream {
                continue;
            }
            match decoder.send_packet(&packet) {
                Ok(_) => (),
                Err(Error::Other { errno: EINVAL }) => {
                    return Err(BlissError::DecodingError(format!(
                        "wrong codec opened for file '{}.",
                        path.display(),
                    )))
                }
                Err(Error::Eof) => {
                    warn!(
                        "Premature EOF reached while decoding file '{}'.",
                        path.display()
                    );
                    drop(tx);
                    song.sample_array = child.join().unwrap()?;
                    return Ok(song);
                }
                Err(e) => warn!("{} when decoding file '{}'", e, path.display()),
            };

            loop {
                let mut decoded = ffmpeg_next::frame::Audio::empty();
                match decoder.receive_frame(&mut decoded) {
                    Ok(_) => {
                        tx.send(decoded).map_err(|e| {
                        BlissError::DecodingError(format!(
                            "while sending decoded frame to the resampling thread for file '{}': {:?}",
                            path.display(),
                            e,
                        ))
                    })?;
                    }
                    Err(_) => break,
                }
            }
        }

        // Flush the stream
        let packet = ffmpeg_next::codec::packet::Packet::empty();
        match decoder.send_packet(&packet) {
            Ok(_) => (),
            Err(Error::Other { errno: EINVAL }) => {
                return Err(BlissError::DecodingError(format!(
                    "wrong codec opened for file '{}'.",
                    path.display()
                )))
            }
            Err(Error::Eof) => {
                warn!(
                    "Premature EOF reached while decoding file '{}'.",
                    path.display()
                );
                drop(tx);
                song.sample_array = child.join().unwrap()?;
                return Ok(song);
            }
            Err(e) => warn!("error while decoding {}: {}", path.display(), e),
        };

        loop {
            let mut decoded = ffmpeg_next::frame::Audio::empty();
            match decoder.receive_frame(&mut decoded) {
                Ok(_) => {
                    tx.send(decoded).map_err(|e| {
                        BlissError::DecodingError(format!(
                        "while sending decoded frame to the resampling thread for file '{}': {:?}",
                        path.display(),
                        e
                    ))
                    })?;
                }
                Err(_) => break,
            }
        }

        drop(tx);
        song.sample_array = child.join().unwrap()?;
        let duration_seconds = song.sample_array.len() as f32 / SAMPLE_RATE as f32;
        song.duration = Duration::from_nanos((duration_seconds * 1e9_f32).round() as u64);
        Ok(song)
    }
}

#[cfg(test)]
mod tests {
    use crate::decoder::ffmpeg::FFmpegDecoder as Decoder;
    use crate::decoder::Decoder as DecoderTrait;
    use crate::decoder::PreAnalyzedSong;
    use crate::BlissError;
    use crate::Song;
    use crate::SAMPLE_RATE;
    use adler32::RollingAdler32;
    use pretty_assertions::assert_eq;
    use std::num::NonZero;
    use std::path::Path;

    fn _test_decode(path: &Path, expected_hash: u32) {
        let song = Decoder::decode(path).unwrap();
        let mut hasher = RollingAdler32::new();
        for sample in song.sample_array.iter() {
            hasher.update_buffer(&sample.to_le_bytes());
        }

        assert_eq!(expected_hash, hasher.hash());
    }

    #[test]
    fn test_tags() {
        let song = Decoder::decode(Path::new("data/s16_mono_22_5kHz.flac")).unwrap();
        assert_eq!(song.artist, Some(String::from("David TMX")));
        assert_eq!(
            song.album_artist,
            Some(String::from("David TMX - Album Artist"))
        );
        assert_eq!(song.title, Some(String::from("Renaissance")));
        assert_eq!(song.album, Some(String::from("Renaissance")));
        assert_eq!(song.track_number, Some(2));
        assert_eq!(song.disc_number, Some(1));
        assert_eq!(song.genre, Some(String::from("Pop")));
        // Test that there is less than 10ms of difference between what
        // the song advertises and what we compute.
        assert!((song.duration.as_millis() as f32 - 11070.).abs() < 10.);
    }

    #[test]
    fn test_special_tags() {
        // This file has tags like `DISC: 02/05` and `TRACK: 06/24`.
        let song = Decoder::decode(Path::new("data/special-tags.mp3")).unwrap();
        assert_eq!(song.disc_number, Some(2));
        assert_eq!(song.track_number, Some(6));
    }

    #[test]
    fn test_unsupported_tags_format() {
        // This file has tags like `TRACK: 02test/05`.
        let song = Decoder::decode(Path::new("data/unsupported-tags.mp3")).unwrap();
        assert_eq!(song.track_number, None);
    }

    #[test]
    fn test_empty_tags() {
        let song = Decoder::decode(Path::new("data/no_tags.flac")).unwrap();
        assert_eq!(song.artist, None);
        assert_eq!(song.title, None);
        assert_eq!(song.album, None);
        assert_eq!(song.track_number, None);
        assert_eq!(song.disc_number, None);
        assert_eq!(song.genre, None);
    }

    #[test]
    fn test_resample_mono() {
        let path = Path::new("data/s32_mono_44_1_kHz.flac");
        let expected_hash = 0xa0f8b8af;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_resample_multi() {
        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let expected_hash = 0xbbcba1cf;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_resample_stereo() {
        let path = Path::new("data/s16_stereo_22_5kHz.flac");
        let expected_hash = 0x1d7b2d6d;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_decode_mono() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.flac -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash addler32 -
        let expected_hash = 0x5e01930b;
        _test_decode(&path, expected_hash);
    }

    #[test]
    fn test_decode_mp3() {
        let path = Path::new("data/s32_stereo_44_1_kHz.mp3");
        // Obtained through
        // ffmpeg -i data/s16_mono_22_5kHz.mp3 -ar 22050 -ac 1 -c:a pcm_f32le
        // -f hash -hash addler32 -
        let expected_hash = 0x69ca6906;
        _test_decode(&path, expected_hash);
    }

    #[test]
    #[cfg(feature = "ffmpeg")]
    fn test_dont_panic_no_channel_layout() {
        let path = Path::new("data/no_channel.wav");
        Decoder::decode(&path).unwrap();
    }

    #[test]
    fn test_decode_right_capacity_vec() {
        let path = Path::new("data/s16_mono_22_5kHz.flac");
        let song = Decoder::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len() + SAMPLE_RATE as usize,
            sample_array.capacity()
        );

        let path = Path::new("data/s32_stereo_44_1_kHz.flac");
        let song = Decoder::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert_eq!(
            sample_array.len() + SAMPLE_RATE as usize,
            sample_array.capacity()
        );

        let path = Path::new("data/capacity_fix.ogg");
        let song = Decoder::decode(&path).unwrap();
        let sample_array = song.sample_array;
        assert!(sample_array.len() as f32 / sample_array.capacity() as f32 > 0.90);
        assert!(sample_array.len() as f32 / (sample_array.capacity() as f32) < 1.);
    }

    #[test]
    fn test_decode_errors() {
        assert_eq!(
        Decoder::decode(Path::new("nonexistent")).unwrap_err(),
        BlissError::DecodingError(String::from(
            "while opening format for file 'nonexistent': ffmpeg::Error(2: No such file or directory)."
        )),
    );
        assert_eq!(
            Decoder::decode(Path::new("data/picture.png")).unwrap_err(),
            BlissError::DecodingError(String::from(
                "No audio stream found for file 'data/picture.png'."
            )),
        );
    }

    #[test]
    fn test_decode_wav() {
        let expected_hash = 0xde831e82;
        _test_decode(Path::new("data/piano.wav"), expected_hash);
    }

    #[test]
    fn test_try_from() {
        let pre_analyzed_song = PreAnalyzedSong::default();
        assert!(<PreAnalyzedSong as TryInto<Song>>::try_into(pre_analyzed_song).is_err());
    }

    #[test]
    fn test_analyze_paths() {
        let analysis = Decoder::analyze_paths(["data/nonexistent", "data/piano.flac"])
            .map(|s| s.1.is_ok())
            .collect::<Vec<_>>();
        assert_eq!(analysis, vec![false, true]);
    }

    #[test]
    fn test_analyze_paths_with_cores() {
        // Analyze with a number of cores greater than the system's number of cores.
        let analysis = Decoder::analyze_paths_with_cores(
            [
                "data/nonexistent",
                "data/piano.flac",
                "data/nonexistent.cue",
            ],
            NonZero::new(usize::MAX).unwrap(),
        )
        .map(|s| s.1.is_ok())
        .collect::<Vec<_>>();
        assert_eq!(analysis, vec![false, true, false]);
    }

    #[test]
    fn test_analyze_paths_with_cores_empty_paths() {
        let analysis =
            Decoder::analyze_paths_with_cores::<&str, [_; 0]>([], NonZero::new(1).unwrap())
                .collect::<Vec<_>>();
        assert_eq!(analysis, vec![]);
    }
}
