#Changelog

## bliss 0.5.0
* Add support for CUE files.
* Fix a bug in `estimate_tuning` that led to empty chroma errors.
* Remove the Library trait, which was too convoluted, and extract a few useful
  functions from there into different modules
  (`analyze_paths` to `lib.rs`, `closest_to_album_group` to `playlist.rs`).
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
