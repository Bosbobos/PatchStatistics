from calculate_asr import calculate_iou, load_model, yolo_detect, load_class_names, preprocess, postprocess
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from torchmetrics.detection import MeanAveragePrecision
import json


def evaluate_model_metrics(
        model_path: str,
        data_dir: str,
        conf_threshold: float = 0.3,
        target_class: int = None,
        classes_path: str = None,
        num_examples=300
) -> Dict[str, float]:
    """
    Оценивает модель детекции по метрикам mAP@0.95 и Recall@0.5

    Args:
        model_path: путь к модели
        data_dir: путь к директории с данными в формате YOLO
        conf_threshold: порог уверенности для детекции
        target_class: целевой класс для оценки (если None, то все классы)
        classes_path: путь к файлу с именами классов

    Returns:
        Словарь с метриками качества
    """
    # Загрузка модели и параметров
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

    # Параметры модели
    model_params = {
        'H': H,
        'W': W,
        'mean': np.array([103.53, 116.28, 123.675], dtype=np.float32),
        'scale': np.array([57.375, 57.12, 58.395], dtype=np.float32),
        'strides': [8, 16, 32, 64],
        'conf_threshold': conf_threshold,
        'num_classes': num_classes,
        'class_names': class_names,
        'model_type': model_type_str
    }

    # Загрузка данных
    images_dir = data_dir
    labels_dir = os.path.join(data_dir, 'labels')

    # Подготовка данных для метрик
    preds = []
    targets = []
    iou_threshold_50_count = 0
    total_objects = 0

    # Обработка изображений
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_examples]

    for img_file in tqdm(image_files, desc="Evaluating model"):
        # Загрузка изображения
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # Детекция объектов
        if model_type_str == 'yolo':
            boxes, scores, class_ids, scores_all = yolo_detect(model, img, conf_threshold)
        else:
            blob = preprocess(img, (H, W), model_params['mean'], model_params['scale'])
            pred = model(torch.from_numpy(blob))[0].detach().numpy()
            boxes, scores, class_ids, scores_all = postprocess(
                pred, (orig_h, orig_w), (H, W), model_params['strides'],
                conf_threshold, model_params['num_classes']
            )

        # Фильтрация по целевому классу
        if target_class is not None:
            target_indices = np.where(class_ids == target_class)[0]
            boxes = boxes[target_indices]
            scores = scores[target_indices]
            class_ids = class_ids[target_indices]
            if scores_all.size > 0:
                scores_all = scores_all[target_indices]

        # Загрузка ground truth
        label_path = os.path.join(labels_dir,
                                  img_file[:-4] + '.txt')

        gt_boxes = []
        gt_labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = [float(x) for x in line.strip().split()]
                    if len(data) < 5:
                        continue

                    cls_id = int(data[0])
                    # Пропускаем если указан целевой класс и он не совпадает
                    if target_class is not None and cls_id != target_class:
                        continue

                    # Конвертация из YOLO формата в абсолютные координаты
                    x_center, y_center, width, height = map(float, data[1:5])
                    x1 = (x_center - width / 2) * orig_w
                    y1 = (y_center - height / 2) * orig_h
                    x2 = (x_center + width / 2) * orig_w
                    y2 = (y_center + height / 2) * orig_h

                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(cls_id)
        else:
            print(f'{label_path} does not exist')
        # Подсчет объектов с IoU > 0.5 и confidence > threshold
        for gt_box in gt_boxes:
            total_objects += 1
            for i, det_box in enumerate(boxes):
                iou = calculate_iou(gt_box, det_box)
                if iou > 0.5 and scores[i] > conf_threshold:
                    iou_threshold_50_count += 1
                    break

        # Форматирование для torchmetrics
        preds.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            'scores': torch.tensor(scores, dtype=torch.float32) if len(scores) > 0 else torch.empty(0),
            'labels': torch.tensor(class_ids, dtype=torch.int32) if len(class_ids) > 0 else torch.empty(0)
        })

        targets.append({
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if len(gt_boxes) > 0 else torch.empty((0, 4)),
            'labels': torch.tensor(gt_labels, dtype=torch.int32) if len(gt_labels) > 0 else torch.empty(0)
        })

    # Вычисление mAP
    metric = MeanAveragePrecision(iou_thresholds=[0.8])
    metric.update(preds, targets)
    map_result = metric.compute()

    # Вычисление Recall@0.5
    recall_at_50 = iou_threshold_50_count / total_objects if total_objects > 0 else 0

    # Сохранение результатов
    results = {
        'mAP@0.8': float(map_result['map'].item()),
        'Recall@0.5': float(recall_at_50),
        'total_objects': int(total_objects),
        'matched_objects': int(iou_threshold_50_count),
        'model_path': model_path,
        'data_dir': data_dir,
        'conf_threshold': conf_threshold,
        'target_class': target_class
    }

    # Сохранение в JSON
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_file = os.path.join(results_dir, f"{model_name}_{data_dir}_metrics.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    yolo_results = evaluate_model_metrics(
        model_path='yolo11s.pt',
        data_dir='dataset',
        conf_threshold=0.3,
        target_class=0
    )
    [print(f"{key}: {value}") for key, value in yolo_results.items()]