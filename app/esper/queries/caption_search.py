from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Caption search (rekall)")
def caption_search():
    from esper.captions import topic_search
    from rekall.video_interval_collection import VideoIntervalCollection
    from esper.rekall import intrvllists_to_result_with_objects
    
    phrases = [
        'may the Force be with you',
        'may the force be with you',
        'May the Force be with you',
        'May the force be with you'
    ]
    results = topic_search(phrases,
        window_size=0,
        video_ids = [vid.id for vid in Video.objects.filter(name__contains="star wars").all()])
    caption_results = VideoIntervalCollection({
        r.id: [(
            (p.start * Video.objects.get(id=r.id).fps),
            (p.end * Video.objects.get(id=r.id).fps),
            0)
            for p in r.postings]
        for r in results
    })
    
    return intrvllists_to_result_with_objects(caption_results, lambda a, b: [])
