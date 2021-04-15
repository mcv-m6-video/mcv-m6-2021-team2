import os
from pathlib import Path
from tqdm import tqdm

import pickle

from collections import defaultdict

from src.readers.ai_city_reader import parse_annotations, group_by_id, group_by_frame, parse_annotations_from_txt
from sklearn.metrics.pairwise import pairwise_distances
from src.metrics.mot_metrics import IDF1Computation
from src.video import get_frames_from_video
import torchvision.transforms.functional as F
from torch import nn

import numpy as np
import cv2
from PIL import Image
import torch

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


class MetricLearningEncoder(nn.Module):
    def __init__(self, path='./checkpoints/epoch_19__ckpt.pth'):
        super().__init__()
        self.cuda = torch.cuda.is_available()
        self.model = torch.load(path)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()

    @staticmethod
    def transform(img):
        img = Image.fromarray(img)
        img = F.resize(img, (128, 128))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def forward(self, x):
        try:
            return self.model.get_embedding(x)
        except:
            return self.model(x)

    def get_embeddings(self, batch):
        with torch.no_grad():
            batch = torch.stack([self.transform(img) for img in batch])
            if self.cuda:
                batch = batch.cuda()
            return self.forward(batch).squeeze().cpu().numpy()


class HistogramEncoder:

    def __init__(self, num_bins):
        self.histogram_bins = num_bins

    def get_embeddings(self, batch):
        # exctract histogram
        histograms = []
        for image in batch:
            histogram = []
            for channel in [0, 1, 2]:
                histogram_channel = np.transpose(
                    cv2.calcHist([image], [channel], None, [self.histogram_bins], [0, 256]))
                histogram.append(histogram_channel)

            histogram = np.hstack(histogram)
            norm = np.linalg.norm(histogram)
            histogram = histogram / norm
            histograms.append(histogram)

        ret = np.vstack(histograms)
        return ret


class RandomEncoder:
    def __init__(self):
        self.length = 512

    def get_embeddings(self, batch):
        # exctract histogram
        return np.random.random((len(batch), self.length))


def get_encoder(method, num_bins):
    if method == 'histogram':
        return HistogramEncoder(num_bins)
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


def get_correspondances(sequence: str, metric: str = 'euclidean', method: str = 'dummy', num_bins: int = 32,
                        thresh=20.):
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

    encoder = get_encoder(method, num_bins)

    embeddings_file = os.path.join('./embeddings', f'{method}_{num_bins}_{sequence}.pkl')
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


def task2(sequence: str, metric: str = 'euclidean', method: str = 'dummy', thresh=20., num_bins: int = 32):
    # obtain reid results
    seq_path = str(Path(f'train/{sequence}'))
    path_results = f'./results/week5/{sequence}_{metric}_{method}_{thresh}_{num_bins}'
    results = get_correspondances(sequence=sequence, method=method, metric=metric, num_bins=num_bins, thresh=thresh)
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

    return results, idf1_computation.get_computation()


def postprocess(sequence: str, metric: str = 'euclidean', method: str = 'dummy', thresh=20., num_bins: int = 32):
    # obtain reid results
    seq_path = str(Path(f'train/{sequence}'))
    path_results = f'./results/week5/{sequence}_{metric}_{method}_{thresh}_{num_bins}'

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
    compute_video = False
    summary_results = f'./results/week5/summary.txt'
    idfs = []

    param_space = [
        {
            'sequence': 'S03',
            'method': 'histogram',
            'metric': 'euclidean',
            'thresh': 0.33,
            'num_bins': 32,
        },
        {
            'sequence': 'S03',
            'method': 'histogram',
            'metric': 'euclidean',
            'thresh': 0.33,
            'num_bins': 64,
        },
        {
            'sequence': 'S03',
            'method': 'metric',
            'metric': 'euclidean',
            'thresh': 0.33,
            'num_bins': -1,
        },
    ]
    for params in param_space:
        _, summary = task2(**params)
        print(f' IDF1 {summary["idf1"]["metrics"] * 100}')
        idfs.append(summary["idf1"]["metrics"] * 100)
        with open(summary_results, 'a') as f:
            text = ', '.join(f'{key}: {value}' for key, value in params.items())
            text += f'=> {summary["idf1"]["metrics"] * 100}\n'
            f.write(text)
        if compute_video:
            postprocess(**params)

    best_params = param_space[idfs.index(max(idfs))]
    print(f' Best configuration is {best_params}')
    postprocess(**best_params)
