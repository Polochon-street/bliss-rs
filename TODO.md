# TODO for bliss-rs

This is a todo-list of what's left to do for bliss-rs.
While most of the features are already there, there is still some work to be
done, mostly around optimizing, but also in feature summarization, feature
evaluation, and distance metrics.

Feel free to submit a PR editing this list if you have some wishes, or to
ask questions if you want to tackle an item.

## Actual TODO

- Optimize / lower RAM consumption: chroma features can be streamed with very
  little precision loss. Need to investigate the most lossless way to stream
  the tuning.

### New features

- Find a proper "objective" way to evaluate how good bliss playlists are. Probably
  no better way than making mini bliss playlists and make local surveys so each
  users can evaluate these playlists.
- A way to learn a metric with a "user survey" on their own libraries using code from the thesis:
  in progress here https://github.com/Polochon-street/bliss-metric-learning/.
- A waypoint feature: go from song1 to song2, both picked by the users, in n songs, without any repetitions between playlist 1 and playlist 2.
- A direction feature ("I want the tempo to go down or stay the same").
- Publish the genre clustering / evaluation code, and try to enhance it.
- Make the "m" distance matrix customizable, using an M that neutralizes
  feature scaling imbalance.
- Use genre clustering (cf already existing code) to find an appropriate M matrix, and put it as alternative.
  Hopefully will make playlists not drift.
- Have some presets like "upbeat music" "melancholic music", etc.
- Have a "similar artists to the current one" playlist option.

### "Under-the hood" features

- Better duplicate finding in the playlist module (sometimes songs have different across albums but should have the same fingerprint).
- Take a look at https://nnethercote.github.io/perf-book/ to see if optimizing is possible.
- Find a way to differenciate classic vs contemporary music (new feature? Or better use of existing features?)
  For instance, Cigarettes after sex / Sweet doesn't seem to give similar enough tracks?
- Get the feature processing examined by a data scientist.
- Should library really use `indicatif`? And not leave it up to the CLI program itself?
- Library: the database should maybe have errored_songs in a separate column (and remove
  the "analyzed" flag?)
- Library: Add a command to dump the contents of the library?

### Maintenance tasks

- Add a list of dependencies / installation guide for windows and mac
- Maybe add playlist functions for single songs as convenience methods?
- Regularly update the python bindings with the new code.
- Freebsd support? (see https://github.com/Polochon-street/bliss-rs/issues/60)
- Try to trim out the crates (it is a bit too big right now).
- grep TODO and see what can be fixed.
- Split library's test module (the file is too big right now).
- Improve bliss-python somehow / use it in a small demo project maybe?
  A blissify in python?
- Investigate what type SAMPLE_RATE is in Aubio - maybe u16 is enough.
- Add a proper feature explanation page.

## Done

- Split out ffmpeg (see https://github.com/Polochon-street/bliss-rs/issues/63 and https://users.rust-lang.org/t/proper-way-to-abstract-a-third-party-provider/107076/8)
  - Make ffmpeg an optional (but default) feature 
  - The library trait must be Decoder-agnostic, and not depend on FFmpeg
  - Make the tests that don't need it not dependent of ffmpeg
  - Make the Song::from_path a trait that is by default implemented with the
    ffmpeg feature (so you can theoretically implement the library trait without ffmpeg)
- The album playlist should take into account multi-CDs albums
- A "song group" feature (I want to make a playlist that's in the vibe of these n songs [like 4-5])
  (can probably recycle the "album" feature) - done https://github.com/Polochon-street/bliss-rs/pull/72
- Make it possible to return failed songs from the library module, so plugins can
  store the errors.
- Check the chroma feature for anomalies (the last 4 numbers look anomalous in a lot of cases -
  compare with https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html etc).
