from track.utils import *
from track.kalman_box_tracker import KalmanBoxTracker
from track.utils import iou, ciou, diou

class BaseTracker:
    """This is the base class for Object Tracking algorithms.
    """
    COST_FUNCTION = {
        "iou": iou,
        "ciou": ciou,
        "diou": diou
    }

    def __init__(self, tracker_class=KalmanBoxTracker, cost_function="iou"):
        self.tracker_class = tracker_class
        self.cost_function = self.COST_FUNCTION[cost_function]
        pass

    def _associate_detections_to_trackers(self, detections, trackers):
        """Assigns detections to tracked object

        Args:
            detections (ArrayLike): bbox detections
            trackers (ArrayLike): Estimated bbox from trackers
            iou_threshold (float, optional): IoU threshold. Defaults to 0.3.

        Returns:
            matches, unmatched_detections and unmatched_trackers
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score, cls],...]
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
            Returns an array list of trackers.

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
        matched, unmatched_dets, _ = self._associate_detections_to_trackers(dets, tracks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
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