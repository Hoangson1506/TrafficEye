from track.utils import *
from track.base_tracker import BaseTracker

class ByteTrack(BaseTracker):
    """This is the ByteTrack algorithm for Object Tracking
    """

    def __init__(self, cost_function=ciou, max_age=1, min_hits=3, iou_threshold=0.3):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.cost_function = cost_function

    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold = 0.3):
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        high_mask = detections[:, 4] >= 0.5
        low_mask = low_mask = (detections[:, 4] >= 0.1) & (detections[:, 4] < 0.5)
        high_conf_dets = detections[high_mask]
        low_conf_dets = detections[low_mask]

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        if len(high_conf_dets) > 0:
            high_conf_iou_matrix = self.cost_function(high_conf_dets[:, np.newaxis], trackers[np.newaxis, :])

            if min(high_conf_iou_matrix.shape) > 0:
                a = (high_conf_iou_matrix > iou_threshold).astype(np.int32)
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
            if high_conf_iou_matrix[m[0], m[1]] >= iou_threshold:
                matches.append([det_idx, tracker_idx])
                unmatched_detections.remove(det_idx)
                if tracker_idx in unmatched_trackers:
                    unmatched_trackers.remove(tracker_idx)

        if len(low_conf_dets) > 0 and len(unmatched_trackers) > 0:
            low_conf_iou_matrix = self.cost_function(low_conf_dets[:, np.newaxis], trackers[unmatched_trackers][np.newaxis, :])

            if min(low_conf_iou_matrix.shape) > 0:
                a = (low_conf_iou_matrix > iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    low_matched = np.stack(np.where(a), axis=1)
                else:
                    low_matched = linear_assignment(-low_conf_iou_matrix)

                for lm in low_matched:
                    det_idx = low_indices[lm[0]]
                    tracker_idx = unmatched_trackers[lm[1]]
                    if low_conf_iou_matrix[lm[0], lm[1]] >= iou_threshold:
                        matches.append([det_idx, tracker_idx])
                        unmatched_trackers.remove(tracker_idx)
            
        if len(matches) > 0:
            matches = np.array(matches)
        else:
            matches = np.empty((0, 2), dtype=int)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
