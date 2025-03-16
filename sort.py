import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in the format [x1, y1, x2, y2].
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_bb_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_bb_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_bb_test + area_bb_gt - inter
    if union <= 0:
        return 0.0
    return inter / union


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """

    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        bbox: list or array in the format [x1, y1, x2, y2]
        """
        # Define a Kalman Filter with 7 dimensions and 4 measurement dimensions.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        # Adjust measurement uncertainty
        self.kf.R[2:, 2:] *= 10.0
        # Give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize the state vector with the bounding box converted to [x, y, s, r]
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        """
        Converts a bounding box in [x1, y1, x2, y2] format to [x, y, s, r]:
          - x, y: center of the box
          - s: scale (area)
          - r: aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Converts the state vector x to bounding box [x1, y1, x2, y2].
        """
        x_c, y_c, s, r = x[0], x[1], x[2], x[3]
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        bbox = np.array(
            [x_c - w / 2.0, y_c - h / 2.0, x_c + w / 2.0, y_c + h / 2.0]
        ).reshape((1, 4))
        if score is None:
            return bbox
        else:
            return np.concatenate((bbox, np.array([[score]])), axis=1)

    def update(self, bbox):
        """
        Updates the state vector with the observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects.

    Parameters:
      detections: np.array of detections in the format [[x1, y1, x2, y2, score], ...]
      trackers: np.array of predicted trackers in the format [[x1, y1, x2, y2, 0], ...]
      iou_threshold: minimum IOU required for matching

    Returns:
      matches: np.array of matched indices in the format [[detection_index, tracker_index], ...]
      unmatched_detections: list of detection indices that were not matched
      unmatched_trackers: list of tracker indices that were not matched
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0), dtype=int),
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk[:4])

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    Simple Online and Realtime Tracking (SORT).
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT.
          max_age: maximum number of frames to keep alive a track without associated detections.
          min_hits: minimum number of associated detections before track is confirmed.
          iou_threshold: minimum IOU for matching.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Updates the tracker with the current frame's detections.

        Parameters:
          dets: numpy array of detections in the format [[x1, y1, x2, y2, score], ...]

        Returns:
          A numpy array of tracks in the format [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_count += 1

        # Predict new locations for all current trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers.
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers with corresponding detections.
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # Remove trackers that have not been updated for a long time.
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
