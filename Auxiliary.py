from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np
import math

def load_class_names(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def preprocess(img: np.ndarray, size: Tuple[int, int], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    h, w = size
    blob = cv2.resize(img, (w, h)).astype(np.float32)
    return ((blob - mean) / scale).transpose(2, 0, 1)[None, ...]


def make_grid_and_strides(in_h: int, in_w: int, strides: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    centers, stride_map = [], []
    for s in strides:
        fh = math.ceil(in_h / s);
        fw = math.ceil(in_w / s)
        yv, xv = np.meshgrid(np.arange(fh), np.arange(fw), indexing='ij')
        cx = (xv + 0.5) * s;
        cy = (yv + 0.5) * s
        pts = np.stack([cx, cy], -1).reshape(-1, 2)
        centers.append(pts)
        stride_map.append(np.full((pts.shape[0],), s, dtype=np.float32))
    return np.concatenate(centers, 0), np.concatenate(stride_map, 0)


def softmax(x: np.ndarray, axis: int = 2) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0];
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1);
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def postprocess(
        pred: np.ndarray,
        orig_sz: Tuple[int, int],
        in_sz: Tuple[int, int],
        strides: List[int],
        conf_thr: float,
        num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig_h, orig_w = orig_sz
    in_h, in_w = in_sz
    cls_logits = pred[:, :num_classes]
    regs = pred[:, num_classes:]
    N, _ = cls_logits.shape

    centers, stride_map = make_grid_and_strides(in_h, in_w, strides)

    scores_all = 1 / (1 + np.exp(-cls_logits))
    class_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(N), class_ids]

    mask = scores > conf_thr
    scores = scores[mask]
    class_ids = class_ids[mask]
    regs = regs[mask]
    centers = centers[mask]
    stride_map = stride_map[mask]

    if scores.size == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, num_classes))

    num_bins = 8
    regs = regs.reshape(-1, 4, num_bins)
    probs = softmax(regs, axis=2)
    bins = np.arange(num_bins, dtype=np.float32)
    dist = (probs * bins).sum(axis=2) * stride_map[:, None]
    l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
    cx, cy = centers[:, 0], centers[:, 1]
    x1, y1 = cx - l, cy - t
    x2, y2 = cx + r, cy + b
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    sx, sy = orig_w / in_w, orig_h / in_h
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep], scores_all[mask][keep]

def draw(
        img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        ids: np.ndarray,
        names: List[str]
) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{names[cid]} {sc:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0