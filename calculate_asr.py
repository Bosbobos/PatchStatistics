import json
import math
import cv2
import numpy as np
import onnx
import pandas as pd
import torch
import os
from onnx2torch import convert
from typing import Optional, List, Tuple, Dict, Any

from tqdm import tqdm
from ultralytics import YOLO


# ─────────── Вспомогательные функции ───────────
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


def apply_patch(
        img: np.ndarray,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        target_class: int,
        patch_path: str,
        patch_size: float,
        out_of_box: bool = False,
        near_box: bool = False,
) -> np.ndarray:
    """
    Применяет патч к объектам целевого класса на изображении

    :param img: исходное изображение
    :param boxes: координаты объектов
    :param class_ids: классы объектов
    :param target_class: целевой класс для атаки
    :param patch_path: путь к файлу патча
    :param patch_size: относительный размер патча
    :return: изображение с наложенными патчами
    """
    patched = img.copy()
    patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)

    if patch is None:
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    if out_of_box:
        patch_width = max(5, int(patch.shape[0] * patch_size))
        patch_height = max(5, int(patch.shape[1] * patch_size))
        patch_resized = cv2.resize(patch, (patch_width, patch_height))
        patched[:patch_width, :patch_height] = patch_resized
        return patched


    target_indices = np.where(class_ids == target_class)[0]

    for i in target_indices:
        x1, y1, x2, y2 = map(int, boxes[i])
        width = int(x2 - x1)
        height = int(y2 - y1)

        # Пропускаем слишком маленькие объекты
        if width < 5 or height < 5:
            continue

        # Изменяем размер патча
        patch_width = max(5, int(width * patch_size))
        patch_height = max(5, int(height * patch_size))
        patch_resized = cv2.resize(patch, (patch_width, patch_height))

        try:
            # Определяем область для наложения патча
            if near_box:
                y_start = y1 - patch_resized.shape[0]
            else:
                y_start = y1
            y_end = y_start + patch_resized.shape[0]
            x_start = x1
            x_end = x1 + patch_resized.shape[1]

            # Убедимся, что патч не выходит за границы изображения
            # Определяем границы обрезки патча
            patch_y_start = 0
            patch_y_end = patch_resized.shape[0]
            patch_x_start = 0
            patch_x_end = patch_resized.shape[1]

            # Проверяем и корректируем верхнюю границу
            if y_start < 0:
                patch_y_start = -y_start
                y_start = 0

            # Проверяем и корректируем левую границу
            if x_start < 0:
                patch_x_start = -x_start
                x_start = 0

            # Проверяем и корректируем нижнюю границу
            if y_end > patched.shape[0]:
                patch_y_end = patch_resized.shape[0] - (y_end - patched.shape[0])
                y_end = patched.shape[0]

            # Проверяем и корректируем правую границу
            if x_end > patched.shape[1]:
                patch_x_end = patch_resized.shape[1] - (x_end - patched.shape[1])
                x_end = patched.shape[1]

            # Проверяем, остался ли патч после обрезки
            if (patch_y_end <= patch_y_start or patch_x_end <= patch_x_start or
                    y_end <= y_start or x_end <= x_start):
                continue

            # Обрезаем патч и накладываем
            patch_cropped = patch_resized[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            patched[y_start:y_end, x_start:x_end] = patch_cropped
        except Exception as e:
            print(f"Error applying patch: {e}")

    return patched


def load_model(
        model_path: str
) -> Tuple[Any, int, int, int]:
    """
    Загружает модель YOLO или ONNX в зависимости от расширения файла
    """
    if model_path.endswith('.pt'):  # YOLO модель
        # Для YOLO моделей возвращаем специальный кортеж
        model = YOLO(model_path)
        # Получаем информацию о входных размерах из модели
        input_shape = model.model.args.get('imgsz', 640)  # стандартный размер для YOLOv8
        if isinstance(input_shape, (list, tuple)):
            H, W = input_shape
        else:
            H = W = input_shape

        # Получаем количество классов
        num_classes = model.model.nc

        return (model, 'yolo'), H, W, num_classes

    else:  # ONNX модель (оригинальная логика)
        onnx_model = onnx.load(model_path)

        # Получаем размеры входа
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        H = input_shape.dim[2].dim_value
        W = input_shape.dim[3].dim_value

        # Получаем размерность выхода
        output_shape = onnx_model.graph.output[0].type.tensor_type.shape
        D = output_shape.dim[2].dim_value
        num_classes = D - 32  # 32 - количество смещений (offsets)

        model = convert(onnx_model)
        return model, H, W, num_classes


def yolo_detect(model, img: np.ndarray, conf_threshold: float = 0.3) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Детекция с использованием YOLO модели
    """
    results = model(img, conf=conf_threshold, verbose=False)

    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

    # Для совместимости с оригинальным интерфейсом создаем scores_all
    if len(class_ids) > 0:
        scores_all = np.zeros((len(class_ids), model.model.nc))
        for i, class_id in enumerate(class_ids):
            scores_all[i, class_id] = scores[i]
    else:
        scores_all = np.array([])

    return boxes, scores, class_ids, scores_all


def detect_and_compare(
        model: Any,
        img: np.ndarray,
        orig_img: np.ndarray,
        patch_path: str,
        model_params: Dict[str, Any],
        target_class: int = 0,
        patch_size: float = 0.4,
        threshold: float = 0.3,
        save_images: bool = True,
        save_dir: Optional[str] = None,
        img_name: Optional[str] = None,
        out_of_box: bool = False,
        near_box: bool = False,
) -> Tuple[int, int, int, List[float], List[float], Optional[np.ndarray]]:
    """
    Обрабатывает изображение: детектирует объекты, применяет патч и черный квадрат,
    сравнивает результаты
    """
    # Извлекаем параметры модели
    H = model_params['H']
    W = model_params['W']
    mean = model_params['mean']
    scale = model_params['scale']
    strides = model_params['strides']
    conf_threshold = model_params['conf_threshold']
    num_classes = model_params['num_classes']
    class_names = model_params['class_names']
    model_type = model_params.get('model_type', 'onnx')  # 'onnx' или 'yolo'

    orig_h, orig_w = img.shape[:2]

    # Детекция на исходном изображении
    if model_type == 'yolo':
        boxes, scores, class_ids, scores_all = yolo_detect(model, img, conf_threshold)
    else:
        blob = preprocess(img, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes, scores, class_ids, scores_all = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

    # Применяем настоящий патч к целевым объектам
    patched_img = apply_patch(
        img, boxes, class_ids, target_class, patch_path, patch_size, out_of_box, near_box
    )

    # Применяем черный патч к целевым объектам
    black_patch_path = 'black_patch.png'
    black_patched_img = apply_patch(
        img, boxes, class_ids, target_class, black_patch_path, patch_size, out_of_box, near_box
    )

    # Детекция на изображении с настоящим патчем
    if model_type == 'yolo':
        boxes_p, scores_p, class_ids_p, scores_all_p = yolo_detect(model, patched_img, conf_threshold)
        boxes_bp, scores_bp, class_ids_bp, scores_all_bp = yolo_detect(model, black_patched_img, conf_threshold)
    else:
        blob = preprocess(patched_img, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes_p, scores_p, class_ids_p, scores_all_p = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

        blob = preprocess(black_patched_img, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes_bp, scores_bp, class_ids_bp, scores_all_bp = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

    # Собираем статистику по целевым объектам
    num_targets = 0
    num_success_real = 0  # Успехи с настоящим патчем
    num_success_black = 0  # Успехи с черным патчем
    confidence_drops_real = []  # Падение уверенности с настоящим патчем
    confidence_drops_black = []  # Падение уверенности с черным патчем

    target_indices = np.where(class_ids == target_class)[0]
    num_targets = len(target_indices)

    for idx in target_indices:
        orig_box = boxes[idx]
        orig_score = scores_all[idx, target_class] if scores_all.size > 0 else 0

        # Проверяем эффективность настоящего патча
        found_real = False
        for j, patched_box in enumerate(boxes_p):
            if class_ids_p[j] == target_class:
                iou = calculate_iou(orig_box, patched_box)
                if iou > 0.5:
                    found_real = True
                    patched_score = scores_all_p[j, target_class] if scores_all_p.size > 0 else 0
                    confidence_drop = orig_score - patched_score
                    confidence_drops_real.append(confidence_drop)

                    if orig_score > threshold and patched_score < threshold:
                        num_success_real += 1
                    break

        if not found_real:
            confidence_drops_real.append(orig_score)
            num_success_real += 1

        # Проверяем эффективность черного патча
        found_black = False
        for j, black_patched_box in enumerate(boxes_bp):
            if class_ids_bp[j] == target_class:
                iou = calculate_iou(orig_box, black_patched_box)
                if iou > 0.5:
                    found_black = True
                    black_patched_score = scores_all_bp[j, target_class] if scores_all_bp.size > 0 else 0
                    confidence_drop = orig_score - black_patched_score
                    confidence_drops_black.append(confidence_drop)

                    if orig_score > threshold and black_patched_score < threshold:
                        num_success_black += 1
                    break

        if not found_black:
            confidence_drops_black.append(orig_score)
            num_success_black += 1

    # Создаем side-by-side изображение с тремя панелями
    result_img = None
    if save_images or save_dir:
        # Визуализация результатов
        vis_clean = draw(orig_img.copy(), boxes, scores, class_ids, class_names)
        vis_patched = draw(patched_img.copy(), boxes_p, scores_p, class_ids_p, class_names)
        vis_black_patched = draw(black_patched_img.copy(), boxes_bp, scores_bp, class_ids_bp, class_names)

        # Добавляем подписи
        cv2.putText(vis_clean, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_patched, "Real Patch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_black_patched, "Black Patch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        result_img = np.hstack([vis_clean, vis_patched, vis_black_patched])

        if save_images and save_dir:
            result_path = os.path.join(
                save_dir,
                f"result_{os.path.basename(img_name)}"
            )
            cv2.imwrite(result_path, result_img)
#            print(f"Result saved: {result_path}")

    return num_targets, num_success_real, num_success_black, confidence_drops_real, confidence_drops_black, result_img


# ─────────── Основная логика ───────────
def run_experiment(
        model_path: str = "nanodet.onnx",
        image_dir: str = "dataset",
        samples_num: int = 300,
        classes_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        patch_size: float = 0.4,
        patch_name: str = "dpatch5000",
        results_dir: Optional[str] = None,
        target_class: int = 0,
        save_images: bool = True,
        out_of_box: bool = False,
        near_box: bool = False,
) -> Dict[str, Any]:
    # Вычисляем производные пути
    patch_path = f"{patch_name}.png"
    if results_dir is None:
        results_dir = f"patched_{patch_name}" + ('_oob' if out_of_box else "") + ('_near_box' if near_box else '')

    # Загрузка модели
    model_info, H, W, num_classes = load_model(model_path)

    # Определяем тип модели
    if isinstance(model_info, tuple) and model_info[1] == 'yolo':
        model, model_type = model_info
        model_type_str = 'yolo'
    else:
        model = model_info
        model_type_str = 'onnx'

    # Загрузка имен классов
    class_names = load_class_names(classes_path)
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Создание директории для результатов
    os.makedirs(results_dir, exist_ok=True)

    # Параметры для модели
    model_params = {
        'H': H,
        'W': W,
        'mean': np.array([103.53, 116.28, 123.675], dtype=np.float32),
        'scale': np.array([57.375, 57.12, 58.395], dtype=np.float32),
        'strides': [8, 16, 32, 64],
        'conf_threshold': conf_threshold,
        'num_classes': num_classes,
        'class_names': class_names,
        'model_type': model_type_str  # Добавляем тип модели
    }

    # Статистика для обоих типов патчей
    total_targets = 0
    successful_attacks_real = 0
    successful_attacks_black = 0
    confidence_drops_real = []
    confidence_drops_black = []

    # Обработка изображений
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:samples_num]

    for image_file in tqdm(image_files, desc="Processing images"):
        # Загрузка изображения
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load {image_file}")
            continue

        # Обработка изображения
        num_targets, num_success_real, num_success_black, img_drops_real, img_drops_black, _ = detect_and_compare(
            model=model,
            img=img,
            orig_img=img.copy(),
            patch_path=patch_path,
            model_params=model_params,
            target_class=target_class,
            patch_size=patch_size,
            save_images=save_images,
            save_dir=results_dir,
            img_name=os.path.basename(image_file),
            out_of_box=out_of_box,
            near_box=near_box,
        )

        # Обновляем статистику
        total_targets += num_targets
        successful_attacks_real += num_success_real
        successful_attacks_black += num_success_black
        confidence_drops_real.extend(img_drops_real)
        confidence_drops_black.extend(img_drops_black)
        '''
        if num_targets > 0:
            print(f"Processed {os.path.basename(image_file)}: "
                  f"targets={num_targets}, "
                  f"real_success={num_success_real} ({num_success_real / num_targets * 100:.1f}%), "
                  f"black_success={num_success_black} ({num_success_black / num_targets * 100:.1f}%)")
        '''
    # Расчет итоговых метрик
    metrics = {
        'model_path': model_path,
        'patch_name': patch_name,
        'total_targets': total_targets,
        'successful_attacks_real': successful_attacks_real,
        'successful_attacks_black': successful_attacks_black,
        'target_class': target_class,
        'target_class_name': class_names[target_class] if target_class < len(class_names) else 'unknown'
    }

    if total_targets > 0:
        metrics['asr_real'] = successful_attacks_real / total_targets
        metrics['asr_black'] = successful_attacks_black / total_targets
        metrics['mean_confidence_drop_real'] = float(np.mean(confidence_drops_real))
        metrics['mean_confidence_drop_black'] = float(np.mean(confidence_drops_black))
        metrics['relative_effectiveness'] = metrics['asr_real'] - metrics['asr_black']
        metrics['conf_drop'] = metrics['mean_confidence_drop_real'] - metrics['mean_confidence_drop_black']
    else:
        metrics['asr_real'] = 0.0
        metrics['asr_black'] = 0.0
        metrics['mean_confidence_drop_real'] = 0.0
        metrics['mean_confidence_drop_black'] = 0.0
        metrics['relative_effectiveness'] = 0.0
        metrics['conf_drop'] = 0.0

    # Сохранение результатов в файл
    json_results_path = 'results'
    os.makedirs(json_results_path, exist_ok=True)
    results_file = os.path.join(json_results_path, f"{model_path.split('.')[0]}_{patch_name}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Вывод результатов
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT RESULTS: {patch_name}")
    print(f"{'=' * 60}")
    print(f"Target class: {target_class} ({metrics['target_class_name']})")
    print(f"Total targets: {total_targets}")
    print(f"Successful attacks (real patch): {successful_attacks_real}")
    print(f"Successful attacks (black patch): {successful_attacks_black}")

    if total_targets > 0:
        print(f"\nASR (real patch): {metrics['asr_real']:.4f} ({metrics['asr_real'] * 100:.1f}%)")
        print(f"ASR (black patch): {metrics['asr_black']:.4f} ({metrics['asr_black'] * 100:.1f}%)")
        print(
            f"Relative effectiveness: {metrics['relative_effectiveness']:.4f} ({metrics['relative_effectiveness'] * 100:.1f}%)")
        print(f"\nMean confidence drop (real patch): {metrics['mean_confidence_drop_real']:.4f}")
        print(f"Mean confidence drop (black patch): {metrics['mean_confidence_drop_black']:.4f}")
        print(f"Relative conf drop: {(metrics['conf_drop']):.4f}")

        # Дополнительная статистика
        if confidence_drops_real:
            print(f"Max confidence drop (real): {np.max(confidence_drops_real):.4f}")
            print(f"Min confidence drop (real): {np.min(confidence_drops_real):.4f}")
    else:
        print("No target objects found")

    print(f"\nResults saved to: {results_file}")
    print(f"{'=' * 60}")

    return metrics


# ─────────── Точка входа ───────────
if __name__ == "__main__":
    # Вызов с параметрами по умолчанию
    name = '0709_yolo_dpatch_1000'
    out_of_box = True

    patch_size = 1 if out_of_box else 0.447

    res_yolo_oob = run_experiment(patch_name=name,
                             image_dir='inria_test',
                             out_of_box=out_of_box,
                             patch_size=patch_size,
                             model_path='yolo11s.pt',
                   save_images=False)
    '''
    res_ndet_oob = run_experiment(patch_name=name,
                             image_dir='test_dataset',
                             out_of_box=out_of_box,
                             patch_size=patch_size,
                             model_path='nanodet.onnx',
                   save_images=False)
    '''
    res_yolo_ib = run_experiment(patch_name=name,
                             image_dir='inria_test',
                             out_of_box=False,
                             patch_size=0.447,
                             model_path='yolo11s.pt',
                   save_images=True)
    '''
    res_ndet_ib = run_experiment(patch_name=name,
                             image_dir='test_dataset',
                             out_of_box=False,
                             patch_size=0.447,
                             model_path='nanodet.onnx',
                   save_images=False)

    '''
    '''
    # both
    comp = {
        'attack_type': ['ib', 'ib','oob', 'oob'],
        'model': ['yolo', 'ndet', 'yolo', 'ndet'],
        'conf_drop': [res_yolo_ib['conf_drop'], res_ndet_ib['conf_drop'], res_yolo_oob['conf_drop'], res_ndet_oob['conf_drop']],
        'asr': [res_yolo_ib['relative_effectiveness'], res_ndet_ib['relative_effectiveness'], res_yolo_oob['relative_effectiveness'], res_ndet_oob['relative_effectiveness']],
            }
    '''
    '''
    # nanodet
    comp = {
        'attack_type': ['ib','oob'],
        'conf_drop': [res_ndet_ib['conf_drop'], res_ndet_oob['conf_drop']],
        'asr': [res_ndet_ib['relative_effectiveness'], res_ndet_oob['relative_effectiveness']],
            }
    '''
    # yolo
    comp = {
        'attack_type': ['ib','oob'],
        'conf_drop': [res_yolo_ib['conf_drop'], res_yolo_oob['conf_drop']],
        'asr': [res_yolo_ib['relative_effectiveness'], res_yolo_oob['relative_effectiveness']],
            }
    print(pd.DataFrame(comp))