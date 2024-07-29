# TODO for bliss-rs

This is a todo-list of what's left to do for bliss-rs.
While most of the features are already there, there is still some work to be
done, mostly around optimizing, but also in feature summarization, feature
evaluation, and distance metrics.

Feel free to submit a PR editing this list if you have some wishes, and to
ask questions if you want to tackle an item.

## Actual TODO

- The album playlist should take into account multi-CDs albums
- Add a list of dependencies / installation guide
- Maybe add playlist functions for single songs as convenience methods?
- Regularly update the python bindings with the new code
- Check the chroma feature for anomalies (the last 4 numbers look anomalous in most of my cases -
  compare with https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html etc)
- Freebsd support? (see https://github.com/Polochon-street/bliss-rs/issues/60)
- Try to trim out the crates (it is a bit too big right now)
- grep TODO and see what can be fixed
- Better duplicate finding in the playlist module (sometimes songs have different across albums but should have the same footprint)
- Split library's test module (the file is too big right now)
- Optimize / lower RAM consumption
- Take a look at https://nnethercote.github.io/perf-book/ to see if optimizing is possible
- Find a way to differenciate classic vs contemporary music (new feature? Or better use of existing features?)
- Publish the genre clustering / evaluation code, and try to enhance it.
  For instance, Cigarettes after sex / Sweet doesn't seem to give similar enough tracks?
- A waypoint feature: go from song1 to song2, both picked by the users, in n songs, without any repetitions between playlist 1 and playlist 2
- A direction feature ("I want the tempo to go down or stay the same")
- A "song group" feature (I want to make a playlist that's in the vibe of these n songs [like 4-5])
  (can probably recycle the "album" feature)
- Use genre clustering (cf already existing code) to find an appropriate M matrix, and put it as alternative
  Hopefully will make playlists not drift
- bliss: A way to learn a metric with a "user survey" on their own libraries using code from the thesis
  (probably reuse the interactive-playlist in blissify?)
- Improve bliss-python somehow / use it in a small demo project maybe?
  A blissify in python?

## Done
- Split out ffmpeg (see https://github.com/Polochon-street/bliss-rs/issues/63 and https://users.rust-lang.org/t/proper-way-to-abstract-a-third-party-provider/107076/8)
  - Make ffmpeg an optional (but default) feature 
  - The library trait must be Decoder-agnostic, and not depend on FFmpeg
  - Make the tests that don't need it not dependent of ffmpeg
  - Make the Song::from_path a trait that is by default implemented with the
    ffmpeg feature (so you can theoretically implement the library trait without ffmpeg)

