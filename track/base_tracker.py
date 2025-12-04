from track.utils import *
from track.kalman_box_tracker import KalmanBoxTracker

class BaseTracker:
    """This is the base class for Object Tracking algorithms.
    """

    def __init__(self):
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

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
            Returns the a similar array, where the last column is the object ID.

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
            tracker = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(tracker)

        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            d = tracker.get_state()[0]
            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [tracker.id + 1])).reshape(1, -1))
            i -= 1

            if (tracker.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 5))