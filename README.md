[![crate](https://img.shields.io/crates/v/bliss-audio.svg)](https://crates.io/crates/bliss-audio)
[![build](https://github.com/Polochon-street/bliss-rs/workflows/Rust/badge.svg)](https://github.com/Polochon-street/bliss-rs/actions)
[![doc](https://docs.rs/bliss-audio/badge.svg)](https://docs.rs/bliss-audio/)

# bliss music analyzer - Rust version
bliss-rs is the Rust improvement of [bliss](https://github.com/Polochon-street/bliss), a
library used to make playlists by analyzing songs, and computing distance between them.

Like bliss, it eases the creation of « intelligent » playlists and/or continuous
play, à la Spotify/Grooveshark Radio, as well as easing creating plug-ins for
existing audio players. For instance, you can use it to make calm playlists
to help you sleeping, fast playlists to get you started during the day, etc.

For now (and if you're looking for an easy-to use smooth play experience),
[blissify](https://crates.io/crates/blissify) implements bliss for
[MPD](https://www.musicpd.org/).

There are also [python](https://pypi.org/project/bliss-audio/) bindings.
The wheels are compiled used [maturin](https://github.com/PyO3/maturin/); the
sources [are available here](https://github.com/Polochon-street/bliss-python)
for inspiration.

Note 1: the features bliss-rs outputs is not compatible with the ones
used by C-bliss, since it uses
different, more accurate values, based on
[actual literature](https://lelele.io/thesis.pdf). It is also faster.

## Dependencies

To use bliss-rs, you'll need a few packages: a C linker, `ffmpeg`, and the associated
development packages (libavformat, libavutil, libavcodec, libavfilter, libavdevice),
as well as the clang development packages. These steps are necessary to e.g. run the
examples below.

On Ubuntu:

    $ sudo apt install build-essential pkg-config libavutil-dev libavformat-dev \
    libavfilter-dev libavdevice-dev libclang-dev 

On Archlinux:

    $ sudo pacman -S base-devel clang ffmpeg sqlite

## Examples

For simple analysis / distance computing, take a look at `examples/distance.rs` and
`examples/analyze.rs`.

If you simply want to try out making playlists from a folder containing songs,
[this example](https://github.com/Polochon-street/bliss-rs/blob/master/examples/playlist.rs)
contains all you need. Usage:

        cargo run --features=serde --release --example=playlist /path/to/folder /path/to/first/song

Don't forget the `--release` flag!

By default, it outputs the playlist to stdout, but you can use `-o <path>`
to output it to a specific path.

To avoid having to analyze the entire folder
several times, it also stores the analysis in `/tmp/analysis.json`. You can customize
this behavior by using `-a <path>` to store this file in a specific place.

Ready to use code examples:

### Compute the distance between two songs

```
use bliss_audio::decoder::bliss_ffmpeg::FFmpeg as Decoder;
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::BlissError;

fn main() -> Result<(), BlissError> {
    let song1 = Decoder::from_path("/path/to/song1")?;
    let song2 = Decoder::from_path("/path/to/song2")?;
        
    println!("Distance between song1 and song2 is {}", song1.distance(&song2));
    Ok(())
}
```

### Make a playlist from a song

```
use bliss_audio::decoder::bliss_ffmpeg::FFmpeg as Decoder;
use bliss_audio::decoder::Decoder as DecoderTrait;
use bliss_audio::{BlissError, Song};
use noisy_float::prelude::n32;

fn main() -> Result<(), BlissError> {
    let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
    let mut songs: Vec<Song> = paths
        .iter()
        .map(|path| Decoder::song_from_path(path))
        .collect::<Result<Vec<Song>, BlissError>>()?;

    // Assuming there is a first song
    let first_song = songs.first().unwrap().to_owned();

    songs.sort_by_cached_key(|song| n32(first_song.distance(&song)));
    println!(
        "Playlist is: {:?}",
        songs
            .iter()
            .map(|song| &song.path)
            .collect::<Vec<&String>>()
    );
    Ok(())
}
```

## Further use

Instead of reinventing ways to fetch a user library, play songs, etc,
and embed that into bliss, it is easier to look at the
[library](https://docs.rs/bliss-audio/latest/bliss_audio/library/index.html) module.
It implements common analysis functions, and allows analyzed songs to be put
in a sqlite database seamlessly.

See [blissify](https://crates.io/crates/blissify) for a reference
implementation.

If you want to experiment with e.g. distance metrics, taking a look at
the [mahalanobis distance](https://docs.rs/bliss-audio/latest/bliss_audio/playlist/fn.mahalanobis_distance.html) function is a good idea. It allows customizing the distance metric,
putting more emphasis on certain parts of the features, e.g. making the tempo
twice as prominent as the timbral features.

It is also possible to perform metric learning, i.e. making playlists tailored
to your specific tastes by learning a distance metric for the mahalanobis distance
using the code in the [metric learning](https://github.com/Polochon-street/bliss-metric-learning/)
repository. [blissify-rs](https://github.com/Polochon-street/blissify-rs/)
is made to work with it, but it should be fairly straightforward to implement
it for other uses. The metric learning repo also contains a bit more context
on what metric learning is.

## Cross-compilation

To cross-compile bliss-rs from linux to x86_64 windows, install the
`x86_64-pc-windows-gnu` target via:

        rustup target add x86_64-pc-windows-gnu

Make sure you have `x86_64-w64-mingw32-gcc` installed on your computer.

Then after downloading and extracting [ffmpeg's prebuilt binaries](https://www.gyan.dev/ffmpeg/builds/),
running:

        FFMPEG_DIR=/path/to/prebuilt/ffmpeg cargo build --target x86_64-pc-windows-gnu --release

Will produce a `.rlib` library file. If you want to generate a shared `.dll`
library, add:

        [lib]
        crate-type = ["cdylib"]

to `Cargo.toml` before compiling, and if you want to generate a `.lib` static
library, add:

        [lib]
        crate-type = ["staticlib"]

You can of course test the examples yourself by compiling them as .exe:

        FFMPEG_DIR=/path/to/prebuilt/ffmpeg cargo build --target x86_64-pc-windows-gnu --release --examples

WARNING: Doing all of the above and making it work on windows requires to have
ffmpeg's dll on your Windows `%PATH%` (`avcodec-59.dll`, etc).
Usually installing ffmpeg on the target windows is enough, but you can also just
extract them from `/path/to/prebuilt/ffmpeg/bin` and put them next to the thing
you generated from cargo (either bliss' dll or executable).

## Acknowledgements

* This library relies heavily on [aubio](https://aubio.org/)'s
  [Rust bindings](https://crates.io/crates/aubio-rs) for the spectral /
  timbral analysis, so a big thanks to both the creators and contributors
  of librosa, and to @katyo for making aubio bindings for Rust.
* The first part of the chroma extraction is basically a rewrite of
  [librosa](https://librosa.org/doc/latest/index.html)'s
  [chroma feature extraction](https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html?highlight=chroma#librosa.feature.chroma_stftfrom)
  from python to Rust, with just as little features as needed. Thanks
  to both creators and contributors as well.
* Finally, a big thanks to
  [Christof Weiss](https://www.audiolabs-erlangen.de/fau/assistant/weiss)
  for pointing me in the right direction for the chroma feature summarization,
  which are basically also a rewrite from Python to Rust of some of the
  awesome notebooks by AudioLabs Erlangen, that you can find
  [here](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html).
