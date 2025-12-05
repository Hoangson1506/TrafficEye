from track.utils import *
from track.base_tracker import BaseTracker
from track.kalman_box_tracker import KalmanBoxTracker

class SORT(BaseTracker):
    """This is the SORT (Simple Online and Realtime Tracking) algorithm for Object Tracking
    """

    def __init__(self, cost_function=ciou, max_age=1, min_hits=3, iou_threshold=0.3, tracker_class=KalmanBoxTracker):
        super().__init__(tracker_class=tracker_class)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.cost_function = cost_function
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """Assigns detections to tracked object

        Args:
            detections (ArrayLike): bbox detections
            trackers (ArrayLike): Estimated bbox from trackers
            iou_threshold (float, optional): IoU threshold. Defaults to 0.3.

        Returns:
            matches, unmatched_detections and unmatched_trackers
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        cost_matrix = self.cost_function(detections[:, np.newaxis], trackers[np.newaxis, :])

        if min(cost_matrix.shape) > 0:
            a = (cost_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-cost_matrix)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d, _ in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, _ in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if (cost_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    




