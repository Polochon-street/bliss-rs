# Changelog

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
