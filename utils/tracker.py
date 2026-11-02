import numpy
import scipy.linalg
from scipy.optimize import linear_sum_assignment

TENTATIVE, TRACKED, LOST, REMOVED = range(4)


def box_iou(a, b):
    # (N, 4), (M, 4) x1 y1 x2 y2 -> (N, M)
    lt = numpy.maximum(a[:, None, :2], b[None, :, :2])
    rb = numpy.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


class KalmanFilter:
    """
    Constant velocity on (cx, cy, w, h), noise proportional to the box size (BoT-SORT)
    """

    def __init__(self, std_position=1. / 20, std_velocity=1. / 160):
        self.F = numpy.eye(8)
        self.F[:4, 4:] = numpy.eye(4)
        self.H = numpy.eye(4, 8)
        self.p = std_position
        self.v = std_velocity

    def initiate(self, z):
        wh = numpy.tile(z[2:], 2)
        std = numpy.r_[2 * self.p * wh, 10 * self.v * wh]
        return numpy.r_[z, numpy.zeros(4)], numpy.diag(std ** 2)

    def predict(self, mean, cov):
        # batched: (N, 8), (N, 8, 8)
        wh = numpy.tile(mean[:, 2:4].clip(1), 2)
        q = numpy.concatenate((self.p * wh, self.v * wh), 1) ** 2
        return mean @ self.F.T, self.F @ cov @ self.F.T + q[:, :, None] * numpy.eye(8)

    def update(self, mean, cov, z):
        s = self.H @ cov @ self.H.T + numpy.diag((self.p * numpy.tile(mean[2:4].clip(1), 2)) ** 2)
        k = scipy.linalg.solve(s, self.H @ cov, assume_a='pos').T  # P H^T S^-1
        return mean + (z - mean[:4]) @ k.T, cov - k @ s @ k.T


class CMC:
    """
    Camera motion compensation: similarity transform of sparse optical flow between frames (BoT-SORT)
    """

    def __init__(self, scale=0.5):
        self.scale = scale
        self.prev = None

    def __call__(self, image):
        import cv2

        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=self.scale, fy=self.scale)
        warp = numpy.eye(2, 3)
        if self.prev is not None and self.prev.shape == gray.shape:
            p0 = cv2.goodFeaturesToTrack(self.prev, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)
            if p0 is not None:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None)
                ok = status.ravel() == 1
                if ok.sum() > 10:
                    m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC)
                    if m is not None:
                        warp = m
                        warp[:, 2] /= self.scale
        self.prev = gray
        return warp

    @staticmethod
    def apply(tracks, warp):
        r = numpy.kron(numpy.eye(4), warp[:, :2])
        for t in tracks:
            t.mean = r @ t.mean
            t.mean[:2] += warp[:, 2]
            t.cov = r @ t.cov @ r.T


class Track:
    def __init__(self, box, score, cls):
        self.z = numpy.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]  # cx, cy, w, h
        self.score = score
        self.cls = int(cls)
        self.votes = {self.cls: score}
        self.id = 0
        self.state = TENTATIVE
        self.mean = self.cov = None
        self.start = self.last = 0

    @property
    def xyxy(self):
        cx, cy, w, h = self.z if self.mean is None else self.mean[:4]
        return numpy.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, kf, det, frame):
        self.mean, self.cov = kf.update(self.mean, self.cov, det.z)
        self.score = det.score
        # class voting, DETR may flip between similar classes (car / truck)
        self.votes[det.cls] = self.votes.get(det.cls, 0) + det.score
        self.cls = max(self.votes, key=self.votes.get)
        self.last = frame


class Tracker:
    """
    ByteTrack: high score detections extend and start tracks, low score detections only extend tracks.
    Size-aware Kalman filter and optional camera motion compensation from BoT-SORT,
    class-agnostic association with class voting, IDs are given to confirmed tracks only.
    """

    def __init__(self, params, fps=30):
        self.high = params['track_high']
        self.low = params['track_low']
        self.new = params['track_new']
        self.match = params['track_match']
        self.buffer = max(1, round(params['track_buffer'] * fps / 30))
        self.cmc = CMC() if params['track_cmc'] else None
        self.kf = KalmanFilter()
        self.reset()

    def reset(self):
        self.tracks = []
        self.frame = 0
        self.count = 0
        if self.cmc is not None:
            self.cmc.prev = None

    def update(self, boxes, scores, classes, image=None):
        """
        boxes (N, 4) x1 y1 x2 y2, scores (N,), classes (N,) of one frame, `image` (BGR) is needed for CMC
        returns (M, 7) x1 y1 x2 y2 id score class of the confirmed tracks matched in this frame
        """
        self.frame += 1
        boxes, scores, classes = (numpy.asarray(i, dtype=numpy.float64).reshape(-1, n)
                                  for i, n in ((boxes, 4), (scores, 1), (classes, 1)))
        keep = self.suppress(boxes, scores[:, 0]) & (scores[:, 0] > self.low)
        dets = [Track(b, s, c) for b, s, c in zip(boxes[keep], scores[keep, 0], classes[keep, 0])]
        high = [d for d in dets if d.score >= self.high]
        low = [d for d in dets if d.score < self.high]

        self.predict(self.tracks)
        if self.cmc is not None and image is not None:
            CMC.apply(self.tracks, self.cmc(image))
        pool = [t for t in self.tracks if t.state in (TRACKED, LOST)]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]

        # 1) tracked and lost tracks <-> high score detections
        matches, u_pool, u_high = self.assign(pool, high, self.match, fuse=True)
        for i, j in matches:
            pool[i].update(self.kf, high[j], self.frame)
            pool[i].state = TRACKED

        # 2) remaining tracked tracks <-> low score detections (occluded / blurred objects)
        rest = [pool[i] for i in u_pool if pool[i].state == TRACKED]
        matches, u_rest, _ = self.assign(rest, low, 0.5)
        for i, j in matches:
            rest[i].update(self.kf, low[j], self.frame)
        for i in u_rest:
            rest[i].state = LOST

        # 3) tentative tracks (one frame old) <-> remaining high score detections
        high = [high[j] for j in u_high]
        matches, u_tentative, u_high = self.assign(tentative, high, 0.7, fuse=True)
        for i, j in matches:
            tentative[i].update(self.kf, high[j], self.frame)
            self.confirm(tentative[i])
        for i in u_tentative:
            tentative[i].state = REMOVED

        # 4) new tentative tracks, confirmed by a match in the next frame
        for j in u_high:
            if high[j].score >= self.new:
                t = high[j]
                t.mean, t.cov = self.kf.initiate(t.z)
                t.start = t.last = self.frame
                self.tracks.append(t)

        for t in self.tracks:
            if t.state == LOST and self.frame - t.last > self.buffer:
                t.state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
        self.deduplicate()

        tracks = [t for t in self.tracks if t.state == TRACKED and t.last == self.frame]
        return numpy.array([[*t.xyxy, t.id, t.score, t.cls] for t in tracks]).reshape(-1, 7)

    def confirm(self, track):
        self.count += 1
        track.id = self.count
        track.state = TRACKED

    def predict(self, tracks):
        if not tracks:
            return
        mean = numpy.stack([t.mean for t in tracks])
        cov = numpy.stack([t.cov for t in tracks])
        mean[[t.state != TRACKED for t in tracks], 6:] = 0  # no size change while not observed
        mean, cov = self.kf.predict(mean, cov)
        for t, m, c in zip(tracks, mean, cov):
            t.mean, t.cov = m, c

    @staticmethod
    def suppress(boxes, scores, threshold=0.95):
        # DETR returns one box per (query, class): drop near-identical boxes of lower score
        if len(boxes) < 2:
            return numpy.ones(len(boxes), dtype=bool)
        order = scores.argsort()[::-1]
        iou = numpy.triu(box_iou(boxes[order], boxes[order]), 1)
        keep = numpy.zeros(len(boxes), dtype=bool)
        keep[order] = iou.max(0) <= threshold
        return keep

    @staticmethod
    def assign(tracks, dets, threshold, fuse=False):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = box_iou(numpy.stack([t.xyxy for t in tracks]), numpy.stack([d.xyxy for d in dets]))
        if fuse:
            iou *= numpy.array([d.score for d in dets])[None]
        cost = 1 - iou
        rows, cols = linear_sum_assignment(numpy.where(cost > threshold, 1e6, cost))
        matches = [(i, j) for i, j in zip(rows, cols) if cost[i, j] <= threshold]
        u_tracks = sorted(set(range(len(tracks))) - {i for i, _ in matches})
        u_dets = sorted(set(range(len(dets))) - {j for _, j in matches})
        return matches, u_tracks, u_dets

    def deduplicate(self, threshold=0.85):
        # a lost track re-created as a new one: keep the older track
        tracked = [t for t in self.tracks if t.state == TRACKED]
        lost = [t for t in self.tracks if t.state == LOST]
        if not tracked or not lost:
            return
        iou = box_iou(numpy.stack([t.xyxy for t in tracked]), numpy.stack([t.xyxy for t in lost]))
        for i, j in zip(*numpy.where(iou > threshold)):
            a, b = tracked[i], lost[j]
            (b if a.last - a.start >= b.last - b.start else a).state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
