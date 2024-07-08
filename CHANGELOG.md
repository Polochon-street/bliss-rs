# Changelog

## bliss 0.8.0
* Remove the `number_songs` option from `Library::playlist_from_custom`.
  Since it now returns an Iterator, `number_songs` can just be replaced by `take`.
* `Library::playlist_from_custom` now returns an Iterator instead of a Vec.
* `Library::playlist_from_custom` now also returns the song(s) the playlist
  was built from.

## bliss 0.7.1
* Bump rust-ffmpeg to 7.0.2, to allow building on FreeBSD systems.

## bliss 0.7.0
* Make ffmpeg an optional dependency, to decouple bliss from ffmpeg:
  - Remove Song::from_path
  - Add a specific `song` and `decoder` module
  - Add a `Decoder` trait to make implementing decoders other than ffmpeg more easily
  - Add an `FFmpeg` struct implementing the previous decoding behavior with ffmpeg
  Existing code will need to be updated by replacing `Song::from_path` by
  `song::decoder::bliss_ffmpeg::FFmpeg::song_from_path`, and the other
  corresponding functions (see the updated examples for more details).
* Put the decoding logic in its own module.
* Bump ffmpeg-next version to 7.0.
* Bump aubio-rs custom crate to disable compiling it with -std=c99.
* Add the possibility to make playlists based on multiple songs using extended
  isolation forest (Thanks @SimonTeixidor!).
* Remove *_by_key family of functions (Thanks @SimonTeixidor!).
* Remove circular dependency between playlist and song by removing distances
  from the `Song` struct (Thanks @SimonTeixidor!).

## bliss 0.6.11
* Bump rust-ffmpeg to 6.1.1 to fix build for raspberry pis.

## bliss 0.6.10
* Make the `analyze` function public, for people who don't want to use
  ffmpeg
* Run `cargo update`, bump ffmpeg to 6.1
* Fix the library module erroring when wrong UTF-8 ends up in the database.

## bliss 0.6.9
* Add a feature flag for compilation on raspberry pis.

## bliss 0.6.8
* Add an `update-aubio-bindings` feature.

## bliss 0.6.7
* Fix compatibility for ffmpeg 6.0, and bump ffmpeg version to 6.0.
* Update and remove extraneous dependencies.

## bliss 0.6.6
* Add a `delete_everything_else` function in `library`'s update functions.
* Use Rust 2021.

## bliss 0.6.5
* Fix library update performance issues.
* Pretty-print JSON in the config file.

## bliss 0.6.4
* Fix a bug in the customizable CPU number option in `library`.

## bliss 0.6.3
* Add customizable CPU number in the `library` module.

## bliss 0.6.2
* Add a `library` module, that greatly helps when making player plug-ins.

## bliss 0.6.1
* Fix a decoding bug while decoding certain WAV files.

## bliss 0.6.0
* Change String to PathBuf in `analyze_paths`.
* Add `analyze_paths_with_cores`.

## bliss 0.5.2
* Fix a bug with some broken MP3 files.
* Bump ffmpeg to 5.1.0.

## bliss 0.5.0
* Add support for CUE files.
* Add `album_artist` and `duration` to `Song`.
* Fix a bug in `estimate_tuning` that led to empty chroma errors.
* Remove the unusued Library trait, and extract a few useful functions from
  there (`analyze_paths`, `closest_to_album_group`).
* Rename `distance` module to `playlist`.
* Remove all traces of the "analyse" word vs "analyze" to make the codebase
  more coherent.
* Rename `Song::new` to `Song::from_path`.

## bliss 0.4.6
* Bump ffmpeg crate version to allow for cross-compilation.

## bliss 0.4.5
* Bump ffmpeg crate version.
* Add an "ffmpeg-static" option.

## bliss 0.4.4
* Make features' version public.

## bliss 0.4.3
* Add features' version on each Song instance.

## bliss 0.4.2
* Add a binary example to easily make playlists.

## bliss 0.4.1
* Add a function to make album playlists.

## bliss 0.4.0
* Make the song-to-song custom sorting method faster.
* Rename `to_vec` and `to_arr1` to `as_vec` and `as_arr1` .
* Add a playlist_dedup function. 

## bliss 0.3.5
* Add custom sorting methods for playlist-making.

## bliss 0.3.4
* Bump ffmpeg's version to avoid building ffmpeg when building bliss.

## bliss 0.3.3
* Add a streaming analysis function, to help libraries displaying progress.

## bliss 0.3.2
* Fixed a rare ffmpeg multithreading bug.

## bliss 0.3.1
* Show error message when song storage fails in the Library trait.
* Added a `distance` module containing euclidean and cosine distance.
* Added various custom_distance functions to avoid being limited to the
  euclidean distance only.

## bliss 0.3.0
* Changed `Song.path` from `String` to `PathBuf`.
* Made `Song` metadata (artist, album, etc) `Option`s.
* Added a `BlissResult` error type.

## bliss 0.2.6
* Fixed an allocation bug in Song::decode that potentially caused segfaults.

## bliss 0.2.5
* Updates to docs

## bliss 0.2.4
* Make `Analysis::to_vec()` public.

## bliss 0.2.3

* Made `NUMBER_FEATURES` public.

## bliss 0.2.1

* Made `Analysis::new` public.
* Made `Analysis` serializable.

## bliss 0.2.0

* Added an `Analysis` struct to `Song`, as well as an `AnalysisIndex` to
  index it easily.
* Changed some logging parameters for the Library trait.
