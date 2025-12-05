from track.utils import *
from track.base_tracker import BaseTracker
from track.kalman_box_tracker import KalmanBoxTracker

class ByteTrack(BaseTracker):
    """This is the ByteTrack algorithm for Object Tracking
    """

    def __init__(self, cost_function=ciou, max_age=1, min_hits=3, 
                 high_conf_iou_threshold=0.5, low_conf_iou_threshold=0.4,
                 high_conf_threshold=0.5, low_conf_threshold=0.1, tracker_class=KalmanBoxTracker):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf_iou_threshold = high_conf_iou_threshold
        self.low_conf_iou_threshold = low_conf_iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.cost_function = cost_function
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.tracker_class = tracker_class

    def _associate_detections_to_trackers(self, detections, trackers):
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        high_mask = detections[:, 4] >= self.high_conf_threshold
        low_mask = (detections[:, 4] >= self.low_conf_threshold) & \
            (detections[:, 4] < self.high_conf_threshold)
        high_conf_dets = detections[high_mask]
        low_conf_dets = detections[low_mask]

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        if len(high_conf_dets) > 0:
            high_conf_iou_matrix = self.cost_function(high_conf_dets[:, np.newaxis], trackers[np.newaxis, :])

            if min(high_conf_iou_matrix.shape) > 0:
                a = (high_conf_iou_matrix > self.high_conf_iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    matched_indices = linear_assignment(-high_conf_iou_matrix)
            else:
                matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        matches = []
        unmatched_detections = high_indices.tolist()
        unmatched_trackers = list(range(len(trackers)))

        for m in matched_indices:
            det_idx = high_indices[m[0]]
            tracker_idx = m[1]
            if high_conf_iou_matrix[m[0], m[1]] >= self.high_conf_iou_threshold:
                matches.append([det_idx, tracker_idx])
                unmatched_detections.remove(det_idx)
                if tracker_idx in unmatched_trackers:
                    unmatched_trackers.remove(tracker_idx)

        if len(low_conf_dets) > 0 and len(unmatched_trackers) > 0:
            low_conf_iou_matrix = self.cost_function(low_conf_dets[:, np.newaxis], trackers[unmatched_trackers][np.newaxis, :])

            if min(low_conf_iou_matrix.shape) > 0:
                a = (low_conf_iou_matrix > self.low_conf_iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    low_matched = np.stack(np.where(a), axis=1)
                else:
                    low_matched = linear_assignment(-low_conf_iou_matrix)

                matches_to_remove = []

                for lm in low_matched:
                    det_idx = low_indices[lm[0]]
                    tracker_rel_idx = lm[1]
                    if low_conf_iou_matrix[lm[0], tracker_rel_idx] >= self.low_conf_iou_threshold:
                        tracker_idx = unmatched_trackers[tracker_rel_idx]
                        matches.append([det_idx, tracker_idx])
                        matches_to_remove.append(tracker_idx)

                # Remove AFTER the loop
                for tracker in matches_to_remove:
                    if tracker in unmatched_trackers:
                        unmatched_trackers.remove(tracker)
            
        if len(matches) > 0:
            matches = np.array(matches)
        else:
            matches = np.empty((0, 2), dtype=int)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score,cls],...]
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
            Returns the an array list of trackers.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1
        tracks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for i in range(len(tracks)):
            pos = self.trackers[i].predict()[0]
            tracks[i, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(i)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, tracks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            if dets[i, 4] >= self.high_conf_threshold:
                bbox = dets[i, :4]
                class_id = int(dets[i, 5])
                tracker = self.tracker_class(bbox, class_id=class_id)
                self.trackers.append(tracker)

        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(tracker)
            i -= 1

            if (tracker.time_since_update > self.max_age):
                self.trackers.pop(i)

        return ret