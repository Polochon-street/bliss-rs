[![crate](https://img.shields.io/crates/v/bliss-audio.svg)](https://crates.io/crates/bliss-audio)
[![build](https://github.com/Polochon-street/bliss-rs/workflows/Rust/badge.svg)](https://github.com/Polochon-street/bliss-rs/actions)
[![doc](https://docs.rs/bliss-rs/badge.svg)](https://docs.rs/bliss-audio/)

# bliss music analyser - Rust version
bliss-rs is the Rust improvement of [bliss](https://github.com/Polochon-street/bliss), a
library used to make playlists by analyzing songs, and computing distance between them.

Like bliss, it eases the creation of « intelligent » playlists and/or continuous
play, à la Spotify/Grooveshark Radio, as well as easing creating plug-ins for
existing audio players.

For now (and if you're looking for an easy-to use smooth play experience),
[blissify](https://crates.io/crates/blissify) implements bliss for
[MPD](https://www.musicpd.org/).

There are also [python](https://pypi.org/project/bliss-audio/) bindings.

Note 1: the features bliss-rs outputs is not compatible with the ones
used by C-bliss, since it uses
different, more accurate values, based on
[actual literature](https://lelele.io/thesis.pdf). It is also faster.

Note 2: The `bliss-rs` crate is outdated. You should use `bliss-audio`
(this crate) instead.

## Examples
For simple analysis / distance computing, a look at `examples/distance.rs` and
`examples/analyse.rs`.

Ready to use examples:

### Compute the distance between two songs
```
use bliss_audio::Song;

fn main() {
        let song1 = Song::new("/path/to/song1");
        let song2 = Song::new("/path/to/song2");

        println!("Distance between song1 and song2 is {}", song1.distance(song2));
}
```

### Make a playlist from a song
```
use bliss_rs::{BlissError, Song};
use ndarray::{arr1, Array};
use noisy_float::prelude::n32;

fn main() -> Result<(), BlissError> {
    let paths = vec!["/path/to/song1", "/path/to/song2", "/path/to/song3"];
    let mut songs: Vec<Song> = paths
        .iter()
        .map(|path| Song::new(path))
        .collect::<Result<Vec<Song>, BlissError>>()?;

    // Assuming there is a first song
    let analysis_first_song = arr1(&songs[0].analysis);

    // Identity matrix used to compute the distance.
    // Note that it can be changed to alter feature ponderation, which
    // may yield to better playlists (subjectively).
    let m = Array::eye(analysis_first_song.len());

    songs.sort_by_cached_key(|song| {
        n32((arr1(&song.analysis) - &analysis_first_song)
            .dot(&m)
            .dot(&(arr1(&song.analysis) - &analysis_first_song)))
    });
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
[Library](https://github.com/Polochon-street/bliss-rs/blob/master/src/library.rs#L12)
trait.

By implementing a few functions to get songs from a media library, and store
the resulting analysis, you get access to functions to analyze an entire
library (with multithreading), and to make playlists easily.

See [blissify](https://crates.io/crates/blissify) for a reference
implementation.

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
