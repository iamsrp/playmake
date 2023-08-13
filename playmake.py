#!/usr/bin/env python3
'''
Make playlists from similar audio files.
'''

import argh
import json
import logging
import math
import numpy as np
import os
import random

from   threading import Thread, Lock

# ------------------------------------------------------------------------------

LOG = logging
LOG.basicConfig(
    format='[%(asctime)s %(threadName)s %(filename)s:%(lineno)d %(levelname)s] %(message)s',
    level=logging.INFO
)

# ------------------------------------------------------------------------------

# Used to protect state when reading or writing the state
_STATE_LOCK = Lock()

def save_data(data, fn):
    '''
    Save out the state.
    '''
    # Save them all out
    LOG.info(f"Saving {fn}")
    with open(fn, 'w') as fh:
        with _STATE_LOCK:
            json.dump(data, fh)


def compute_embeddings(pending, root, data, data_fn, save_period):
    '''
    Work through the list of pending files and compute their
    embeddings. Each time we do this we save them out since they are expensive
    to compute.
    '''
    import openl3
    import soundfile as sf

    # Keep going until there is nothing left
    count = 0
    while len(pending) > 0:
        fn = pending.pop()
        if not fn.startswith('#') and fn not in data['embeddings']:
            LOG.info(f"Handling {fn}")
            try:
                # Compute the embedding
                audio,     sr = sf.read(os.path.join(root, fn))
                embedding, _  = openl3.get_audio_embedding(audio,
                                                           sr,
                                                           embedding_size=512,
                                                           hop_size=10)
                # Store it
                with _STATE_LOCK:
                    count += 1
                    data['embeddings'][fn] = [
                        float(v) for v in np.mean(embedding, axis=0)
                    ]

                # Save them out?
                if save_period <= 0 or (count % save_period) == 0:
                    with _STATE_LOCK:
                        save_data(data, data_fn)

            except Exception as e:
                LOG.warning(f'Failed to handle {fn}: {e}')

    # And one for luck, if we need to and haven't just done so
    if count > 0 and save_period > 0 and (count % save_period) != 0:
        save_data(data, data_fn)


def generate_playlist(count, start, files, data, data_fn):
    '''
    Generate the list of songs, starting at the given one.
    '''
    LOG.info("Starting a playlist with %s", start)

    # True if we need to save out the mutated data
    dirty = False

    # Local handles
    distances  = data['distances' ]
    embeddings = data['embeddings']

    # What we give back, starting with the first one
    result = []
    result.append(start)

    # The distance matrix entry key format. This needs to be a string since we
    # need to dump them via JSON and also be able to use them as a key in a
    # dict.
    keyfmt = '%s<=>%s'

    # And go..!
    song = start
    seen = set()
    for i in range(count):
        seen.add(song)
        if song not in embeddings:
            raise ValueError(f'{song} not in embeddings')
        s_e = embeddings[song]

        # Compute the distances between this file and all the others
        dists = []
        for fn in files:
            if fn not in embeddings:
                continue
            if fn in seen:
                continue

            # The distance matrix entries, these needs to be a string since we
            # need to dump them via JSON and also be able to use them as a key
            # in a dict.
            pair   = keyfmt % (song, fn)
            rpair  = keyfmt % (fn, song)
            if pair not in distances:
                # We'll use the Euclidian distance
                diffs = [(a-b) for (a, b) in zip(s_e, embeddings[fn])]
                distance = math.sqrt(sum([d * d for d in diffs]))

                # Put the distance into the state
                with _STATE_LOCK:
                    distances[pair] = distances[rpair] = distance
                    dirty = True

            else:
                distance = distances[pair]

            # And save
            dists.append((fn, distance))

        # Now compute
        dists = sorted(dists, key=lambda pair:pair[1])
        limit = 10
        prev = song
        LOG.debug("Closest to %s:  %s", song, dists[:3])
        LOG.debug("Furthest to %s: %s", song, dists[-3:])
        while song in seen and limit < len(dists):
            song = random.choice(dists[:limit])[0]
            limit += 1
        result.append(song)

        LOG.info("Added %s with distance %0.5f",
                 song, distances[keyfmt % (prev, song)])

    # Save out the data, if we added distances
    if dirty:
        save_data(data, data_fn)

    # And give it all back
    return result

# ------------------------------------------------------------------------------

@argh.arg('--count', '-c',
          help="How many entries to generate in the playlist")
@argh.arg('--distance', '-d',
          help="How many songs to look at away from the current one")
@argh.arg('--generate_only', '-G',
          help="Don't compute distances, just generate")
@argh.arg('--num_compute_threads', '-t',
          help="How many threads to use to compute the embeddings")
@argh.arg('--playlist', '-p',
          help="The path of the playlist file")
@argh.arg('--save_period', '-S',
          help="How frequently to save out the embeddings to disk")
@argh.arg('--start', '-s',
          help="The entry (text) from the playlist file to start at")
def main(playlist=None,
         count=100,
         distance=25,
         generate_only=False,
         num_compute_threads=10,
         save_period=10,
         start=None):
    '''
    Main entry point.
    '''
    # We need a playlist
    if not playlist:
        raise ValueError("No playlist given")

    # Where we get and save data
    if playlist.endswith('.m3u'):
        data_fn = playlist.replace('.m3u', '.pmk')
    else:
        data_fn = playlist + '.pmk'

    # Where on disk to find the playlist entries
    dirname = os.path.dirname(playlist)

    # The state
    if os.path.exists(data_fn):
        # Pull in the current data
        LOG.info(f"Loading {data_fn}")
        with open(data_fn, 'r') as fh:
            data = json.load(fh)
        LOG.info(f"Loaded {data_fn}")
    else:
        data = { 'embeddings' : dict(),
                 'distances'  : dict() }

    # Parse the playlist
    with open(playlist, 'r') as fh:
        LOG.info(f"Loading {playlist}")
        files = tuple(fn.strip() for fn in fh.readlines())
    LOG.info('Had %d files in %s' % (len(files), playlist))

    # Need to choose a starting point?
    if not start:
        start = random.choice(files)

    # Compute the embeddings do this in a bunch of threads for speed
    if not generate_only:
        pending = list(files)
        pending.reverse()

        # And spawn the compute threads
        def compute():
            compute_embeddings(pending, dirname, data, data_fn, save_period)
        threads = []
        for i in range(num_compute_threads):
            t = Thread(target=compute)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    # Now generate the playlist
    playlist = generate_playlist(count, start, files, data, data_fn)

    # And print it!
    for entry in playlist:
        print(entry)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        argh.dispatch_command(main)
    except Exception as e:
        print("%s" % e)
