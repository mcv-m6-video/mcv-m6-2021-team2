import os
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

import pickle

from collections import defaultdict

from src.readers.ai_city_reader import parse_annotations, group_by_id, group_by_frame, parse_annotations_from_txt
from sklearn.metrics.pairwise import pairwise_distances
from src.metrics.mot_metrics import IDF1Computation
from src.video import get_frames_from_video

import networkx as nx


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


class MetricLearningEncoder:

    def __init__(self):
        self.length = 512

    def get_embeddings(self, batch):
        return np.random.random((len(batch), self.length))


class HistogramEncoder:

    def get_embeddings(self, batch):
        # exctract histogram
        histograms = []
        for image in batch:
            histogram = []
            for channel in [0, 1, 2]:
                histogram_channel = np.transpose(cv2.calcHist([image], [channel], None, [64], [0, 256]))
                histogram.append(histogram_channel)
            histograms.append(np.hstack(histogram))

        ret = np.vstack(histograms)
        assert ret.shape == (len(batch), 64 * 3)
        return ret


class RandomEncoder:
    def __init__(self):
        self.length = 512

    def get_embeddings(self, batch):
        # exctract histogram
        return np.random.random((len(batch), self.length))


def get_encoder(method):
    if method == 'histogram':
        return HistogramEncoder()
    elif method == 'dummy':
        return RandomEncoder()
    elif method == 'metric':
        return MetricLearningEncoder()


def get_distances(metric, embed_cam1, embed_cams2):
    if metric == 'euclidean':
        dist = pairwise_distances([embed_cam1], embed_cams2, metric).flatten()
    elif metric == 'histogram_correl':
        dist = []
        for embed_cam2 in embed_cams2:
            d = cv2.compareHist(embed_cam1, embed_cam2, cv2.HISTCMP_CORREL)
            dist.append(d)
        dist = np.array(dist)

    return dist


def get_correspondances(sequence: str, metric: str = 'euclidean', method: str = 'dummy', thresh=300):
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

    encoder = get_encoder(method)

    embeddings_file = os.path.join('./embeddings', f'{method}_{sequence}.pkl')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_track_embeddings(tracks_by_cam, cap, encoder, save_path=embeddings_file)

    embeddings = {(cam, id): embd for cam in embeddings for id, embd in embeddings[cam].items()}
    # at this stage we have `embeddings` mapped by pairs (cam, id): That means, that for a camera and id pair we only
    # have a single embedding
    G = nx.Graph()
    for cam1 in cams:
        for id1, track1 in tracks_by_cam[cam1].items():
            candidates = []

            for cam2 in cams:
                if cam2 == cam1:
                    continue
                for id2, track2 in tracks_by_cam[cam2].items():
                    candidates.append((cam2, id2))

            if len(candidates) > 0:
                dist = get_distances(metric, embeddings[(cam1, id1)],
                                     [embeddings[(cam2, id2)] for cam2, id2 in candidates])
                # dist = pairwise_distances([embeddings[(cam1, id1)]],
                #                           [embeddings[(cam2, id2)] for cam2, id2 in candidates],
                #                           metric).flatten()

                ind = dist.argmin()
                if dist[ind] < thresh:
                    cam2, id2 = candidates[ind]
                    G.add_edge((cam1, id1), (cam2, id2))

    groups = []
    while G.number_of_nodes() > 0:
        cliques = nx.find_cliques(G)
        maximal = max(cliques, key=len)
        groups.append(maximal)
        G.remove_nodes_from(maximal)

    results = defaultdict(list)
    for global_id, group in enumerate(groups):
        for cam, id in group:
            track = tracks_by_cam[cam][id]
            for det in track:
                det.id = global_id
            results[cam].append(track)

    return results


def write_results(results, path):
    for cam, tracks in results.items():
        lines = []
        for track in tracks:
            for det in track:
                lines.append((det.frame, det.id, int(det.xtl), int(det.ytl), int(det.width), int(det.height),
                              det.score, '-1', '-1', '-1'))
        lines = sorted(lines, key=lambda x: x[0])

        filename = os.path.join(path, cam, 'results.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            for line in lines:
                file.write(','.join(list(map(str, line))) + '\n')


def task2(sequence: str, metric: str = 'euclidean', method: str = 'dummy', thresh=500):
    # obtain reid results
    seq_path = str(Path(f'train/{sequence}'))
    path_results = f'./results/week5/{method}_{metric}_{thresh}'
    results = get_correspondances(sequence=sequence, method=method, metric=metric, thresh=thresh)
    write_results(results, path=path_results)

    # compute metrics
    idf1_computation = IDF1Computation()
    for cam in os.listdir(seq_path):
        dets_true = group_by_frame(parse_annotations_from_txt(os.path.join(seq_path, cam, 'gt', 'gt.txt')))
        dets_pred = group_by_frame(parse_annotations_from_txt(os.path.join(path_results, cam, 'results.txt')))
        for frame in dets_true.keys():
            y_true = dets_true.get(frame, [])
            y_pred = dets_pred.get(frame, [])
            idf1_computation.add_frame_detections(y_true, y_pred)
    print(f'Metrics: {idf1_computation.get_computation()}')

    cams = sorted(os.listdir(seq_path))
    import imageio
    colors = {}
    for cam in cams:
        writer = imageio.get_writer(
            os.path.join(path_results, f'{cam}.gif'), fps=10)
        video_path = os.path.join(seq_path, cam, 'vdo.avi')
        detections = group_by_frame(parse_annotations_from_txt(os.path.join(path_results, cam, 'results.txt')))
        for frame_idx, current_frame in get_frames_from_video(str(video_path), start_frame=600, end_frame=800):
            for det in detections.get(frame_idx - 1, []):
                if det.id not in colors:
                    import random
                    color = (int(random.random() * 256),
                             int(random.random() * 256),
                             int(random.random() * 256))
                    colors[det.id] = color
                cv2.rectangle(current_frame, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), colors[det.id],
                              6)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(cv2.resize(current_frame, (600, 350)))


if __name__ == '__main__':
    task2(sequence='S03', method='histogram', metric='histogram_correl')
