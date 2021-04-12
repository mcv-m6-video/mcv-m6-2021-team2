import os
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

import pickle

from collections import defaultdict

from src.readers.ai_city_reader import parse_annotations, group_by_id, group_by_frame
from sklearn.cluster import DBSCAN


def is_static(track, thresh=50):
    std = np.std([det.center for det in track], axis=0)
    return np.all(std < thresh)


def get_track_embedding(track, cap, encoder, max_views=32):
    batch = []
    for det in np.random.permutation(track)[:max_views]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame)
        ret, img = cap.read()
        img = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
        if img.size > 0:
            batch.append(img)

    embeddings = encoder.get_embeddings(batch)

    # combine track embeddings by averaging them
    embedding = embeddings.mean(axis=0)

    return embedding


def get_track_embeddings(tracks_by_cam, cap, encoder, batch_size=512, save_path=None):
    embeddings = defaultdict(dict)
    for cam in tqdm(tracks_by_cam, desc='Computing embeddings', leave=True):
        # process camera detections frame by frame
        detections = [det for track in tracks_by_cam[cam].values() for det in track]
        detections_by_frame = group_by_frame(detections)

        cap[cam].set(cv2.CAP_PROP_POS_FRAMES, 0)
        length = int(cap[cam].get(cv2.CAP_PROP_FRAME_COUNT))

        track_embeddings = defaultdict(list)
        batch = []
        ids = []
        for _ in tqdm(range(length), desc=f'cam={cam}', leave=False):
            # read frame
            frame = int(cap[cam].get(cv2.CAP_PROP_POS_FRAMES))
            _, img = cap[cam].read()
            if frame not in detections_by_frame:
                continue

            # crop and accumulate frame detections
            for det in detections_by_frame[frame]:
                crop = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                if crop.size > 0:
                    batch.append(crop)
                    ids.append(det.id)

            # compute embeddings if enough detections in batch
            if len(batch) >= batch_size:
                embds = encoder.get_embeddings(batch)
                for id, embd in zip(ids, embds):
                    track_embeddings[id].append(embd)
                batch.clear()
                ids.clear()

        # compute embeddings of last batch
        if len(batch) > 0:
            embds = encoder.get_embeddings(batch)
            for id, embd in zip(ids, embds):
                track_embeddings[id].append(embd)

        # combine track embeddings by averaging them
        for id, embds in track_embeddings.items():
            embeddings[cam][id] = np.stack(embds).mean(axis=0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings


class Encoder:

    def __init__(self):
        self.length = 512

    def get_embeddings(self, batch):
        return np.random.random((len(batch), self.length))


def get_encoder():
    return Encoder()


def task2(sequence: str, metric: str = 'euclidean'):
    seq_path = str(Path(f'train/{sequence}'))
    cams = sorted(os.listdir(seq_path))
    tracks_by_cam = {
        cam: group_by_id(parse_annotations(os.path.join(seq_path, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt')))
        for cam in cams
    }
    cap = {
        cam: cv2.VideoCapture(os.path.join(seq_path, cam, 'vdo.avi'))
        for cam in cams
    }

    # filter out static tracks
    for cam in cams:
        tracks_by_cam[cam] = dict(filter(lambda x: not is_static(x[1]), tracks_by_cam[cam].items()))

    # tracks_by_cam have a set of tracks (Detections for individual ids for each camera)

    # print(f' tracks_by_cam {tracks_by_cam}')
    encoder = get_encoder()

    embeddings_file = os.path.join('./embeddings', f'dummy_{sequence}.pkl')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_track_embeddings(tracks_by_cam, cap, encoder, save_path=embeddings_file)
    # print(f' embeddings-1 {embeddings}')

    embeddings = {(cam, id): embd for cam in embeddings for id, embd in embeddings[cam].items()}
    # at this stage we have `embeddings` mapped by pairs (cam, id): That means, that for a camera and id pair we only
    # have a single embedding
    print(f' embeddings {embeddings.keys()}')
    print(f' embedding {embeddings[("c001", 10)]}')
    clustering = DBSCAN(eps=0.3, min_samples=2, metric=metric)
    clustering.fit(np.stack(list(embeddings.values())))

    print(f' embeddings.keys() {embeddings.keys()}, {len(embeddings.keys())}')
    print(f' clustering.labels_ {clustering.labels_}, {len(clustering.labels_)}')

    groups = defaultdict(list)
    for id, label in zip(embeddings.keys(), clustering.labels_):
        groups[label].append(id)
    groups = list(groups.values())

    print(f' groups {groups}')

    results = defaultdict(list)
    for global_id, group in enumerate(groups):
        for cam, id in group:
            track = tracks_by_cam[cam][id]
            for det in track:
                det.id = global_id
            results[cam].append(track)

    # print(f' results {results}')
    return results


def get_embeddings(frames_cam_ids_dicts, cap, encoder, save_path=None):
    embeddings = defaultdict(dict)
    for frame in tqdm(frames_cam_ids_dicts, desc=f'Computing embeddings for frame', leave=True):
        for cam in tqdm(frames_cam_ids_dicts[frame], desc=f'Computing embeddings for cam', leave=False):
            # process camera detections frame by frame
            detections = frames_cam_ids_dicts[frame][cam]
            embeddings[frame][cam] = {}

            cap[cam].set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, img = cap[cam].read()
            batch = []
            ids = []
            for det in detections:
                crop = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                if crop.size > 0:
                    batch.append(crop)
                    ids.append(det.id)

            embds = encoder.get_embeddings(batch)
            for id, embd in zip(ids, embds):
                embeddings[frame][cam][id] = embd

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings


def task_5_2(sequence: str, metric: str = 'euclidean'):
    seq_path = str(Path(f'train/{sequence}'))
    cams = sorted(os.listdir(seq_path))

    frames_by_cam = {
        cam: group_by_frame(parse_annotations(os.path.join(seq_path, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt')))
        for cam in cams
    }
    cap = {
        cam: cv2.VideoCapture(os.path.join(seq_path, cam, 'vdo.avi'))
        for cam in cams
    }

    frames_cam_ids_dicts = defaultdict(dict)

    for cam in frames_by_cam:
        for frame in frames_by_cam[cam]:
            frames_cam_ids_dicts[frame][cam] = frames_by_cam[cam][frame]

    """
          {
              frame_id: 
                  {
                    'cam1': [dets]
                    'cam2': [dets]
                    'cam3': [dets]
                  }
              ...
          }
    """

    print(f' frames_cam_ids_dicts {frames_cam_ids_dicts.keys()}')
    print(f' frames_cam_ids_dicts[1] {frames_cam_ids_dicts[1].keys()}')
    print(f' frames_cam_ids_dicts[1]["c001"] {frames_cam_ids_dicts[1]["c001"]}')

    encoder = get_encoder()

    embeddings_file = os.path.join('./embeddings', f'dummy_{sequence}.pkl')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_embeddings(frames_cam_ids_dicts, cap, encoder, save_path=embeddings_file)

    print(f' embeddings {embeddings.keys()}')
    print(f' embeddings[1] {embeddings[1].keys()}')
    print(f' embeddings[1]["c001"] {embeddings[1]["c001"].keys()}')

    # Apply min graph algorithm for every frame


if __name__ == '__main__':
    task_5_2('S01')
