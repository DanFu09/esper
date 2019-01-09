from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("All aligned captions (rekall)")
def all_captions():
    from esper.captions import get_all_segments
    from rekall.video_interval_collection import VideoIntervalCollection
    from esper.rekall import intrvllists_to_result_with_objects
    
    video_ids = [1]

    # Only aligned captions are in the caption index
    results = get_all_segments(video_ids)
    caption_results = VideoIntervalCollection({
        video_id: [(
            word[0] * Video.objects.get(id=video_id).fps, # start frame
            word[1] * Video.objects.get(id=video_id).fps, # end frame
            word[2]) # payload is the word (string)
            for word in words]
        for video_id, words in results
    })
    
    return intrvllists_to_result_with_objects(caption_results, lambda a, b: [])
