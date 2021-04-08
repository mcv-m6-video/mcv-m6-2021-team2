from copy import deepcopy
from src.track import Track
from src.metrics.iou import iou_single_boxes


class MaxOverlapTracker():
    def __init__(self):
        self.track_number = 0

    def track_by_max_overlap(self, tracks, detections):
        current_detections = deepcopy(detections)
        tracks_on_frame = []

        # Check if current_detections can be matched with detections from current tracks
        for track in tracks:
            if track.finished:
                continue
            best_matched = self.match_detections(track.previous_detection, current_detections)
            if best_matched:
                track.add_detection_to_tracking(best_matched)
                tracks_on_frame.append(track)
                current_detections.remove(best_matched)
            else:
                track.finished = True

        # For the unkown detection create new detections
        for detection in current_detections:
            new_tracking = Track(self.track_number, detection)
            tracks.append(new_tracking)
            tracks_on_frame.append(new_tracking)
            self.track_number += 1
        return tracks, tracks_on_frame

    def match_detections(self, previous_detection, current_detections):
        max_iou = 0
        for detection in current_detections:
            iou = iou_single_boxes(previous_detection.bbox, detection.bbox)
            if iou > max_iou:
                max_iou = iou
                best_match = detection
        if max_iou > 0:
            best_match.id = previous_detection.id
            return best_match
        else:
            return None


class MaxOverlapTrackerEfficient():
    def __init__(self):
        self.track_number = 0
        self.max_coincidence = 2

    def track_by_max_overlap(self, tracks, detections):
        current_detections = deepcopy(detections)
        tracks_on_frame = []

        # Check if current_detections can be matched with detections from current tracks
        for track in tracks:
            if track.finished:
                continue
            best_matched = self.match_detections(track.previous_detection, current_detections)
            if best_matched:
                track.add_detection_to_tracking(best_matched[0])
                tracks_on_frame.append(track)
                for matching in best_matched:
                    current_detections.remove(matching)
            else:
                track.finished = True

        # For the unkown detection create new detections
        for detection in current_detections:
            new_tracking = Track(self.track_number, detection)
            tracks.append(new_tracking)
            tracks_on_frame.append(new_tracking)
            self.track_number += 1
        return tracks, tracks_on_frame

    def match_detections(self, previous_detection, current_detections):
        max_iou = 0
        best_matches = []
        for detection in current_detections:
            iou = iou_single_boxes(previous_detection.bbox, detection.bbox)
            if iou > max_iou:
                max_iou = iou
                best_matches.append((iou, detection))
        if max_iou > 0:
            best_matches = sorted(best_matches, key=lambda tup: tup[0])
            best_matches[0][1].id = previous_detection.id
            match_detections = []
            for match in best_matches:
                match_detections.append(match[1])
            if len(match_detections) > self.max_coincidence:
                return match_detections[:self.max_coincidence]
            else:
                return match_detections
        else:
            return None

class MaxOverlapTrackerWithOpticalFlow():
    def __init__(self):
        self.track_number = 0

    def track_by_max_overlap(self, tracks, detections, optical_flow=None):
        current_detections = deepcopy(detections)
        tracks_on_frame = []

        # Check if current_detections can be matched with detections from current tracks
        for track in tracks:
            if track.finished:
                continue
            best_matched = self.match_detections(track.previous_detection, current_detections, optical_flow)
            if best_matched:
                track.add_detection_to_tracking(best_matched)
                tracks_on_frame.append(track)
                current_detections.remove(best_matched)
            else:
                track.finished = True

        # For the unkown detection create new detections
        for detection in current_detections:
            new_tracking = Track(self.track_number, detection)
            tracks.append(new_tracking)
            tracks_on_frame.append(new_tracking)
            self.track_number += 1
        return tracks, tracks_on_frame

    def match_detections(self, previous_detection, current_detections, optical_flow):
        prev_det = deepcopy(previous_detection)
        
        if optical_flow is not None:
            prev_det.xtl += optical_flow[int(prev_det.ytl), int(prev_det.xtl), 0]
            prev_det.ytl += optical_flow[int(prev_det.ytl), int(prev_det.xtl), 1]
            prev_det.xbr += optical_flow[int(prev_det.ybr), int(prev_det.xbr), 0]
            prev_det.ybr += optical_flow[int(prev_det.ybr), int(prev_det.xbr), 1]

        max_iou = 0
        for detection in current_detections:
            iou = iou_single_boxes(prev_det.bbox, detection.bbox)
            if iou > max_iou:
                max_iou = iou
                best_match = detection
        if max_iou > 0:
            best_match.id = prev_det.id
            return best_match
        else:
            return None