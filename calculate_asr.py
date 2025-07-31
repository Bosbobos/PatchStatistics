import math
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import os
from onnx2torch import convert

# ─────────── settings ───────────
MODEL_PATH       = "nanodet_plus_62_192.onnx"
IMAGE_DIR        = "dataset"
CLASSES_PATH     = None
CONF_THRESHOLD   = 0.3
PATCH_SIZE       = 0.5
PATCH_NAME       = "dpatch5000"
PATCH_PATH       = PATCH_NAME + ".png"
RESULTS_DIR      = "results_" + 'corner_' + PATCH_NAME
# ─────────────────────────────────

def load_class_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def preprocess(img, size, mean, scale):
    h, w = size
    blob = cv2.resize(img, (w, h)).astype(np.float32)
    return ((blob - mean) / scale).transpose(2,0,1)[None,...]

def make_grid_and_strides(in_h, in_w, strides):
    centers, stride_map = [], []
    for s in strides:
        fh = math.ceil(in_h/s); fw = math.ceil(in_w/s)
        yv, xv = np.meshgrid(np.arange(fh), np.arange(fw), indexing='ij')
        cx = (xv + 0.5)*s; cy = (yv + 0.5)*s
        pts = np.stack([cx, cy], -1).reshape(-1,2)
        centers.append(pts)
        stride_map.append(np.full((pts.shape[0],), s, dtype=np.float32))
    return np.concatenate(centers, 0), np.concatenate(stride_map, 0)

def softmax(x, axis=2):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def nms(boxes, scores, iou_thr=0.45):
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

def postprocess(pred, orig_sz, in_sz, strides, conf_thr, num_classes):
    orig_h, orig_w = orig_sz
    in_h, in_w     = in_sz
    cls_logits = pred[:, :num_classes]
    regs       = pred[:, num_classes:]
    N, _       = cls_logits.shape

    centers, stride_map = make_grid_and_strides(in_h, in_w, strides)

    scores_all = 1 / (1 + np.exp(-cls_logits))
    class_ids  = np.argmax(scores_all, axis=1)
    scores     = scores_all[np.arange(N), class_ids]

    mask       = scores > conf_thr
    scores     = scores[mask]
    class_ids  = class_ids[mask]
    regs       = regs[mask]
    centers    = centers[mask]
    stride_map = stride_map[mask]

    if scores.size == 0:
        return np.zeros((0,4)), np.array([]), np.array([]), np.zeros((0, num_classes))

    num_bins = 8
    regs     = regs.reshape(-1, 4, num_bins)
    probs    = softmax(regs, axis=2)
    bins     = np.arange(num_bins, dtype=np.float32)
    dist     = (probs * bins).sum(axis=2) * stride_map[:,None]
    l,t,r,b  = dist[:,0], dist[:,1], dist[:,2], dist[:,3]
    cx,cy    = centers[:,0], centers[:,1]
    x1,y1    = cx - l, cy - t
    x2,y2    = cx + r, cy + b
    boxes    = np.stack([x1,y1,x2,y2], axis=1)

    sx, sy = orig_w / in_w, orig_h / in_h
    boxes[:, [0,2]] *= sx
    boxes[:, [1,3]] *= sy

    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep], scores_all[mask][keep]

def draw(img, boxes, scores, ids, names):
    out = img.copy()
    for (x1,y1,x2,y2), sc, cid in zip(boxes, scores, ids):
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{names[cid]} {sc:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out

def calculate_iou(box1, box2):
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

# ─────────── Основной код ───────────
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
onnx_model = onnx.load(MODEL_PATH)
model = convert(onnx_model)
inp = sess.get_inputs()[0]
name_in, _, _, H, W = inp.name, *inp.shape
outp = sess.get_outputs()[0]
_, N, D = outp.shape

num_offsets = 32
num_classes = D - num_offsets
if CLASSES_PATH:
    class_names = load_class_names(CLASSES_PATH)
    assert len(class_names) == num_classes
else:
    class_names = [f"class_{i}" for i in range(num_classes)]

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

total_class0_objects = 0
successful_attacks = 0
confidence_drops = []

image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
               if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    orig = cv2.imread(image_file)
    if orig is None:
        print(f"Не удалось загрузить {image_file}")
        continue
    orig_h, orig_w = orig.shape[:2]

    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    scale = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    blob = preprocess(orig, (H, W), mean, scale)

    pred = model(torch.from_numpy(blob))[0].detach().numpy()
    strides = [8, 16, 32, 64]
    boxes, scores, cls_ids, scores_all = postprocess(
        pred, (orig_h, orig_w), (H, W), strides, CONF_THRESHOLD, num_classes
    )

    class0_indices = np.where(cls_ids == 0)[0]
    total_class0_objects += len(class0_indices)

    patched = orig.copy()
    for i in class0_indices:
        x1, y1, x2, y2 = map(int, boxes[i])
        width = int(x2 - x1)
        height = int(y2 - y1)
        patch = cv2.imread(PATCH_PATH, cv2.IMREAD_UNCHANGED)
        if patch is None:
            print(f"Не удалось загрузить патч {PATCH_PATH}")
            continue
        patch_resized = cv2.resize(patch, (int(width * PATCH_SIZE), int(height * PATCH_SIZE)))
        try:
            patched[y1:y1 + patch_resized.shape[0], x1:x1 + patch_resized.shape[1]] = patch_resized
        except ValueError as e:
            print(f"Ошибка при наложении патча на {image_file}: {e}")
            continue

    blob = preprocess(patched, (H, W), mean, scale)
    pred = model(torch.from_numpy(blob))[0].detach().numpy()
    boxes_patched, scores_patched, cls_ids_patched, scores_all_patched = postprocess(
        pred, (orig_h, orig_w), (H, W), strides, CONF_THRESHOLD, num_classes
    )

    for idx in class0_indices:
        orig_box = boxes[idx]
        orig_score = scores_all[idx, 0]  # Уверенность в классе 0
        found = False
        for j, patched_box in enumerate(boxes_patched):
            if cls_ids_patched[j] == 0:
                iou = calculate_iou(orig_box, patched_box)
                if iou > 0.5:
                    found = True
                    patched_score = scores_all_patched[j, 0]
                    confidence_drop = orig_score - patched_score
                    confidence_drops.append(confidence_drop)
                    break
        if not found:
            successful_attacks += 1
            confidence_drop = orig_score  # Считаем, что уверенность упала до 0
            confidence_drops.append(confidence_drop)

    vis_clean = draw(orig.copy(), boxes, scores, cls_ids, class_names)
    vis_patched = draw(patched.copy(), boxes_patched, scores_patched, cls_ids_patched, class_names)
    result = np.hstack([vis_clean, vis_patched])
    result_path = os.path.join(RESULTS_DIR, f"result_{os.path.basename(image_file)}")
    cv2.imwrite(result_path, result)
    print(f"Результат сохранен в {result_path}")

print('Название атаки:' + PATCH_NAME)
if total_class0_objects > 0:
    asr = successful_attacks / total_class0_objects
    mean_confidence_drop = np.mean(confidence_drops) if confidence_drops else 0
    print(f"Attack Success Rate (ASR) для нулевого класса: {asr:.4f} "
          f"({successful_attacks}/{total_class0_objects})")
    print(f"Средний Confidence Drop для нулевого класса: {mean_confidence_drop:.4f}")
else:
    print("Объекты нулевого класса не найдены в датасете.")