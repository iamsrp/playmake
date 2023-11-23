# PlayMake

## Overview

A simple playlist generator which uses audio similarity to walk the song space.

## Description

PlayMake takes a standard `m3u` playlist and puts all the songs therein into an
embedding space. Once it has these, it walks the space, finding random close
(Euclidean distance) neighbouring songs, appending them to a playlist as it does
so.

For example:

```bash
playmake.py -G -p $HOME/Music/mp3/Pop/Pop.m3u -s 'The_The/45_RPM_-_The_Singles_Of_The_The/13-December_Sunlight_(Cried_Out).mp3'
```

It's not fast (even if you have a decent GPU), it is memory hungry, and it's
also far from perfect. But it kinda works.

## Dependencies

PlayMake requires:
  - [argh](https://pypi.org/project/argh/) for argument parsing.
  - [mutagen](https://pypi.org/project/mutagen/) for reading song metadata.
  - [openl3](https://github.com/marl/openl3) to generate the embeddings in the
    song space.
