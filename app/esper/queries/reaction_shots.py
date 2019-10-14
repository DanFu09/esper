from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Reaction shots in Apollo 13 (rekall)")
def reaction_shots_apollo_13():
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.temporal_predicates import overlaps
    from rekall.parsers import in_array, merge_dict_parsers, bbox_payload_parser, dict_payload_parser
    from esper.caption_metadata import caption_metadata_for_video
    from esper.captions import get_all_segments
    from esper.rekall import intrvllists_to_result_with_objects
    from query.models import FaceCharacterActor, Shot
    
    videos = Video.objects.filter(name__contains="apollo 13").all()
    
    # Load script data
    metadata = VideoIntervalCollection({
        video.id: caption_metadata_for_video(video.id)
        for video in videos
    }).filter(
        lambda meta_interval: (meta_interval.payload['speaker'] is not None and
                                "man's voice" not in meta_interval.payload['speaker'] and
                                meta_interval.payload['speaker'].strip() != "gene krantz")
    )
    
    all_segments = get_all_segments([video.id for video in videos])
    
    captions_interval_collection = VideoIntervalCollection({
        video: intervals
        for video, intervals in all_segments
    })
    
    captions_with_speaker_id = captions_interval_collection.overlaps(
        metadata.filter(payload_satisfies(lambda p: p['aligned'])),
        payload_merge_op=lambda word, script_meta: (word[0], script_meta['speaker'])
    )
    
    # Annotate face rows with start and end frames and the video ID
    faces_with_character_actor_qs = FaceCharacterActor.objects.annotate(
        min_frame=F('face__frame__number'),
        max_frame=F('face__frame__number'),
        video_id=F('face__frame__video_id'),
        character_name=F('characteractor__character__name')
    ).filter(video_id__in=[v.id for v in videos])

    frames_with_identity = VideoIntervalCollection.from_django_qs(
        faces_with_character_actor_qs,
        with_payload=in_array(
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'character': 'character_name' }),
        )
    ).coalesce(payload_merge_op=payload_plus)
    
    # Annotate shots with all the people in them
    shots_qs = Shot.objects.filter(
        cinematic=True, video_id__in=[v.id for v in videos]
    ).annotate(fps=F('video__fps'))
    shots = VideoIntervalCollection.from_django_qs(shots_qs, with_payload=lambda shot:shot.fps)

    # Annotate shots with mode shot scale
    frames_with_shot_scale_qs = Frame.objects.filter(
        regularly_sampled=True,
        video_id__in=[v.id for v in videos]
    ).annotate(
        min_frame=F('number'),
        max_frame=F('number'),
        shot_scale_name=F('shot_scale__name')
    ).all()
    frames_with_shot_scale = VideoIntervalCollection.from_django_qs(
        frames_with_shot_scale_qs,
        with_payload=lambda f: f.shot_scale_name
    )

    def get_mode(items):
        return max(set(items), key=items.count)
    shots_with_scale = shots.merge(
        frames_with_shot_scale,
        predicate=overlaps(),
        payload_merge_op=lambda shot_fps, shot_scale: [(shot_fps, shot_scale)]
    ).coalesce(
        payload_merge_op = payload_plus
    ).map(
        lambda intrvl: (intrvl.start, intrvl.end, {
            'fps': intrvl.payload[0][0],
            'shot_scale': get_mode([p[1] for p in intrvl.payload])
        })
    )

    shots_with_people_in_them = shots_with_scale.overlaps(
        frames_with_identity,
        payload_merge_op=lambda shot_payload, identities: (shot_payload, identities),
        working_window=1
    ).coalesce(payload_merge_op=lambda p1, p2: (p1[0], p1[1] + p2[1])).map(
        lambda intrvl: (intrvl.start / intrvl.payload[0]['fps'], intrvl.end / intrvl.payload[0]['fps'], {
            'fps': intrvl.payload[0]['fps'],
            'shot_scale': intrvl.payload[0]['shot_scale'],
            'characters': set([
                name.strip().split(' ')[0].strip()
                for d in intrvl.payload[1]
                for name in d['character'].split('/')
                if len(name.strip()) > 0
            ])
        })
    )

    reaction_shots = captions_with_speaker_id.overlaps(
        shots_with_people_in_them.filter(
            payload_satisfies(lambda p: p['shot_scale'] in ['medium_close_up', 'close_up', 'extreme_close_up'])
        ),
        predicate = lambda captions, shots: captions.payload[1].strip().split(' ')[0] not in shots.payload['characters'],
        payload_merge_op = lambda word_and_speaker, fps_and_characters: (fps_and_characters['fps'], word_and_speaker)
    ).map(
        lambda intrvl: (
            int(intrvl.start * intrvl.payload[0]), 
            int(intrvl.end * intrvl.payload[0]), 
            [intrvl.payload[1]]
        )
    ).dilate(12).coalesce(payload_merge_op=payload_plus).dilate(-12).filter_length(min_length=12)
    
    return intrvllists_to_result_with_objects(reaction_shots, lambda a, b: [])
