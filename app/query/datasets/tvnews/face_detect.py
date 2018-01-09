from query.datasets.prelude import *
from scannerpy.stdlib import pipelines, parsers

LABELER, _ = Labeler.objects.get_or_create(name='tinyfaces')
LABELED_TAG, _ = Tag.objects.get_or_create(name='tinyfaces:labeled')
cwd = os.path.dirname(os.path.abspath(__file__))

METHOD = 'mtcnn'


def face_detect(videos, all_frames, force=False):
    existing_frames = Frame.objects.filter(
        video=videos[0], number__in=all_frames[0], tags=LABELED_TAG).count()
    needed_frames = len(all_frames[0])
    if force or existing_frames != needed_frames:
        log.debug('Faces not cached, missing {}/{} frames'.format(needed_frames - existing_frames,
                                                                  needed_frames))

        def output_name(video, frames):
            return video.path + '_faces_' + str(hash(tuple(frames)))

        with make_scanner_db(kube=True) as db:
            # ingest_if_missing(db, videos)

            def remove_already_labeled(video, frames):
                already_labeled = set([
                    f['number']
                    for f in Frame.objects.filter(video=video, tags=LABELED_TAG).values('number')
                ])
                return sorted(list(set(frames) - already_labeled))

            filtered_frames = [
                remove_already_labeled(video, vid_frames) if not force else vid_frames
                for video, vid_frames in zip(videos, all_frames)
            ]
            to_compute = [(video, vid_frames) for video, vid_frames in zip(videos, filtered_frames)
                          if force or not db.has_table(output_name(video, vid_frames))
                          or not db.table(output_name(video, vid_frames)).committed()]

            if len(to_compute) > 0:
                if METHOD == 'mtcnn':
                    device = DeviceType.CPU
                    db.register_op('MTCNN', [('frame', ColumnType.Video)], ['bboxes'])
                    db.register_python_kernel('MTCNN', device, cwd + '/mtcnn_kernel.py', batch=50)

                    frame = db.ops.FrameInput()
                    frame_strided = frame.sample()
                    bboxes = db.ops.MTCNN(frame=frame_strided, device=device)
                    output = db.ops.Output(columns=[bboxes])

                    jobs = [
                        Job(op_args={
                            frame: db.table(video.path).column('frame'),
                            frame_strided: db.sampler.gather(vid_frames),
                            output: output_name(video, vid_frames)
                        }) for video, vid_frames in to_compute
                    ]

                    log.debug('Running face detect on {} jobs'.format(len(jobs)))
                    db.run(
                        BulkJob(output=output, jobs=jobs),
                        force=True,
                        io_packet_size=50000,
                        work_packet_size=500,
                        pipeline_instances_per_node=1)
                    log.debug('Done!')
                    exit()

                elif METHOD == 'tinyfaces':
                    pipelines.detect_faces(
                        db, [db.table(video.path).column('frame') for video, _ in to_compute], [
                            db.sampler.gather(vid_frames) for _, vid_frames in to_compute
                        ], [output_name(video, vid_frames) for video, vid_frames in to_compute])
                else:
                    raise Exception("Invalid face detect method {}".format(METHOD))

            log.debug('Saving metadata')
            for video, video_frames in tqdm(zip(videos, filtered_frames)):
                video_faces = list(
                    db.table(output_name(video, video_frames)).load(
                        ['bboxes'], lambda lst, db: parsers.bboxes(lst[0], db.protobufs)))

                frames = [Frame(video=video, number=n) for n in video_frames]
                Frame.objects.bulk_create(frames)

                people = []
                tags = []
                for (_, frame_faces), frame in zip(video_faces, frames):
                    tags.append(
                        Frame.tags.through(tvnews_frame_id=frame.pk, tvnews_tag_id=LABELED_TAG.pk))
                    for bbox in frame_faces:
                        people.append(Person(frame=frame))
                Frame.tags.through.objects.bulk_create(tags)
                Person.objects.bulk_create(people)

                faces_to_save = []
                p_idx = 0
                for (_, frame_faces) in video_faces:
                    for bbox in frame_faces:
                        faces_to_save.append(
                            Face(
                                person=people[p_idx],
                                bbox_x1=bbox.x1 / video.width,
                                bbox_x2=bbox.x2 / video.width,
                                bbox_y1=bbox.y1 / video.height,
                                bbox_y2=bbox.y2 / video.height,
                                bbox_score=bbox.score,
                                labeler=LABELER))
                        p_idx += 1

                Face.objects.bulk_create(faces_to_save)

    return [
        group_by_frame(
            list(
                Face.objects.filter(
                    person__frame__video=video,
                    person__frame__number__in=vid_frames,
                    labeler=LABELER).select_related('person', 'person__frame')),
            lambda f: f.person.frame.number,
            lambda f: f.id,
            include_frame=False) for video, vid_frames in zip(videos, all_frames)
    ]
