import math
import sys
import os
from multiprocessing import Pool
from pathlib import Path

from esper.prelude import par_for
from query.models import Video

sys.path.append('/app/deps/caption-index')

import captions.util as caption_util
from captions.index import Documents, Lexicon, InvertedIndex, DocumentData, MetadataIndex


INDEX_DIR = '/app/data/index10'
DOCUMENTS_PATH = os.path.join(INDEX_DIR, 'docs.list')
LEXICON_PATH = os.path.join(INDEX_DIR, 'words.lex')
INDEX_PATH = os.path.join(INDEX_DIR, 'index.bin')
DOCUMENT_DATA_PATH = os.path.join(INDEX_DIR, 'docs.bin')
METADATA_PATH = os.path.join(INDEX_DIR, 'meta.bin')

print('Loading the document list and lexicon', file=sys.stderr)
DOCUMENTS = Documents.load(DOCUMENTS_PATH)
LEXICON = Lexicon.load(LEXICON_PATH)
INDEX = InvertedIndex(INDEX_PATH, LEXICON, DOCUMENTS)
DOCUMENT_DATA = DocumentData(DOCUMENT_DATA_PATH, LEXICON, DOCUMENTS)


def _get_video_name(p):
    """Only the filename without exts"""
    return Path(p).name.split('.')[0]


def _init_doc_id_to_vid_id():
    video_name_to_id = {_get_video_name(v.path) : v.id for v in Video.objects.all()}
    doc_id_to_vid_id = {}
    num_docs_with_no_videos = 0
    for d in DOCUMENTS:
        video_name = _get_video_name(d.name)
        video_id = video_name_to_id.get(video_name, None)
        if video_id is not None:
            doc_id_to_vid_id[d.id] = video_id
        else:
            num_docs_with_no_videos += 1
    print('{} documents have no videos'.format(num_docs_with_no_videos), file=sys.stderr)
    print('{} videos have no documents'.format(len(video_name_to_id) - len(doc_id_to_vid_id)),
          file=sys.stderr)
    return doc_id_to_vid_id
            

DOCUMENT_ID_TO_VIDEO_ID = _init_doc_id_to_vid_id()
VIDEO_ID_TO_DOCUMENT_ID = {v: k for k, v in DOCUMENT_ID_TO_VIDEO_ID.items()}



def _doc_ids_to_video_ids(results):
    def wrapper(document_results):
        for d in document_results:
            video_id = DOCUMENT_ID_TO_VIDEO_ID.get(d.id, None)
            if video_id is not None:
                yield d._replace(id=video_id)
    return results._replace(documents=wrapper(results.documents))
        

def topic_search(phrases, dilate=60):
    if not isinstance(phrases, list):
        raise TypeError('phrases should be a list of phrases/n-grams')
    return _doc_ids_to_video_ids(caption_util.topic_search(phrases, INDEX, window_size=dilate))
                            

def phrase_search(query):
    return _doc_ids_to_video_ids(INDEX.search(query))
    

# Set before forking, this is a hack
LOWER_CASE_ALPHA_IDS = None

    
def _get_lowercase_segments(video_id, dilate=1, verbose=False):
    doc_id = VIDEO_ID_TO_DOCUMENT_ID.get(video_id, None)
    if doc_id is None:
        if verbose:
            print('No document for video id: {}'.format(video_id), file=sys.stderr)
        return []
    
    def has_lowercase(tokens):
        for t in tokens:
            if t in LOWER_CASE_ALPHA_IDS:
                return True
        return False
    
    lowercase_segments = []
    curr_interval = None
    for interval in DOCUMENT_DATA.token_intervals(doc_id, 0, DOCUMENTS[doc_id].duration):
        if has_lowercase(interval.tokens):
            if curr_interval is None:
                curr_interval = (interval.start - dilate, interval.end + dilate)
            else:
                curr_start, curr_end = curr_interval
                if min(interval.end + dilate, curr_end) - max(interval.start - dilate, curr_start) > 0:
                    curr_interval = (
                        min(interval.start - dilate, curr_start), 
                        max(interval.end + dilate, curr_end)
                    )
                else:
                    lowercase_segments.append(curr_interval)
                    curr_interval = (interval.start - dilate, interval.end + dilate)
    if curr_interval is not None:
        lowercase_segments.append(curr_interval)
    return lowercase_segments
    

def get_lowercase_segments(video_ids=None):
    if video_ids is None:
        video_ids = [v.id for v in Video.objects.filter(threeyears_dataset=True)]
    elif not isinstance(video_ids, list):
        video_ids = list(video_ids)
    
    def has_lower_alpha(word):
        for c in word:
            if c.isalpha() and c.islower():
                return True
        return False

    lowercase_alpha_ids = {w.id for w in LEXICON if has_lower_alpha(w.token)}
    global LOWER_CASE_ALPHA_IDS
    LOWER_CASE_ALPHA_IDS = lowercase_alpha_ids
    with Pool(os.cpu_count()) as pool:
        results = pool.map(_get_lowercase_segments, video_ids)
    return zip(video_ids, results)

NGRAM_LEXICON_IDS = None

def _scan_for_ngrams_in_parallel(video_id, verbose=None):
    ngram_intervals = [[] for _ in NGRAM_LEXICON_IDS]
    doc_id = VIDEO_ID_TO_DOCUMENT_ID.get(video_id, None)
    if doc_id is None:
        if verbose:
            print('No document for video id: {}'.format(video_id), file=sys.stderr)
        return ngram_intervals
    for interval in DOCUMENT_DATA.token_intervals(doc_id, 0, DOCUMENTS[doc_id].duration):
        found_token = [[False for _ in ids] for ids in NGRAM_LEXICON_IDS]
        for token in interval.tokens:
            for i, ngram in enumerate(NGRAM_LEXICON_IDS):
                for j, ngram_id in enumerate(ngram):
                    if token == ngram_id:
                        found_token[i][j] = True
        for i, token_markers in enumerate(found_token):
            found = True
            for marker in token_markers:
                if not marker:
                    found = False
                    break
            if found:
                ngram_intervals[i].append((interval.start, interval.end))

    return ngram_intervals

def scan_for_ngrams_in_parallel(ngram_list, video_ids=None):
    """
    Scans through video transcripts for the terms in the ngrams.

    ngramlist is a list of ngrams to search for, for example
    ["JOINING US NOW", "VERMONT SENATOR", "THIS IS CNN"].

    This function scans through the transcripts of the videos in video_ids and
    returns a list of tuples where the first tuple is the video id, and the
    second tuple is a list of lists of intervals, one list for every ngram.

    Will return all intervals where all words in the ngram appear.
    """
    if video_ids is None:
        video_ids = [v.id for v in Video.objects.filter(threeyears_dataset=True)]
    elif not isinstance(video_ids, list):
        video_ids = list(video_ids)

    ngram_lexicon_ids = [
        [LEXICON[ngram].id for ngram in ngrams.split(" ")]
        for ngrams in ngram_list
    ]
    global NGRAM_LEXICON_IDS
    NGRAM_LEXICON_IDS = ngram_lexicon_ids

    with Pool(os.cpu_count()) as pool:
        results = pool.map(_scan_for_ngrams_in_parallel, video_ids)
    return zip(video_ids, results)
