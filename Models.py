import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import torch
from ultralytics import YOLO
from pytorchyolo import detect, models
import onnx
from onnx2torch import convert


def load_model(
        model_path: str,
        model_backend: str = 'ultralytics'  # 'ultralytics', 'yolov5_hub', 'pytorchyolo'
) -> Tuple[Any, int, int, int]:
    """
    Загружает модель YOLO или ONNX в зависимости от расширения файла и бэкенда
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

    if model_path.endswith('.pt'):
        if model_backend == 'ultralytics':
            # Загрузка через пакет 'ultralytics' (API YOLOv8)
            model = YOLO(model_path).to(device)
            input_shape = model.model.args.get('imgsz', 640)
            if isinstance(input_shape, (list, tuple)):
                H, W = input_shape
            else:
                H = W = input_shape
            num_classes = model.model.nc
            # Возвращаем кортеж с (model, 'yolo', 'backend_type')
            return (model, 'yolo', 'yolo_ultralytics'), H, W, num_classes

        elif model_backend == 'yolov5_hub':
            # Загрузка через torch.hub (оригинальная библиотека yolov5)
            # 'trust_repo=True' может понадобиться для последних версий torch
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True).to(device)
            H = W = model.imgsz if hasattr(model, 'imgsz') else 640
            num_classes = model.model.nc if hasattr(model.model, 'nc') else 80
            # Возвращаем кортеж с (model, 'yolo', 'backend_type')
            return (model, 'yolo', 'yolo_hub'), H, W, num_classes
        else:
            raise ValueError(f"Unknown model_backend for .pt file: {model_backend}")

    elif model_path.endswith('.weights') and model_backend == 'pytorchyolo':
        # Загрузка через pytorchyolo (для YOLOv3)
        config_path = model_path.rsplit('.', 1)[0] + '.cfg'
        model = models.load_model(config_path, model_path)
        #model.to(device)
        input_size = int(model.hyperparams['height'])
        H = W = input_size
        num_classes = 80  # default
        for module_def in model.module_defs:
            if module_def["type"] == "yolo":
                num_classes = int(module_def["classes"])
                break
            return (model, 'yolo', 'yolo_pytorchyolo'), H, W, num_classes

    elif model_path.endswith('.onnx'):  # ONNX модель (оригинальная логика)
        onnx_model = onnx.load(model_path)
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        H = input_shape.dim[2].dim_value
        W = input_shape.dim[3].dim_value
        output_shape = onnx_model.graph.output[0].type.tensor_type.shape
        D = output_shape.dim[2].dim_value
        num_classes = D - 32
        model = convert(onnx_model).to(device=device)
        model = model
        # ONNX возвращает модель напрямую (не в кортеже)
        return model, H, W, num_classes
    else:
        raise ValueError("Unsupported model format or backend. Use .pt, .weights or .onnx with appropriate backend")


def yolo_detect(model, img: np.ndarray, conf_threshold: float = 0.3) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Детекция с использованием YOLO модели (пакет 'ultralytics')
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


def yolo_hub_detect(
        model, img: np.ndarray, conf_threshold: float, num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Детекция с использованием ОРИГИНАЛЬНОЙ YOLOv5 (torch.hub)
    """
    results = model(img)  # Модель сама применяет нужный размер

    # results.xyxy[0] это тензор [x1, y1, x2, y2, conf, cls]
    preds = results.xyxy[0].cpu().numpy()

    # Фильтруем по conf_threshold
    preds = preds[preds[:, 4] >= conf_threshold]

    if preds.shape[0] == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, num_classes))

    boxes = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Для совместимости с интерфейсом создаем scores_all
    scores_all = np.zeros((len(class_ids), num_classes))
    if len(class_ids) > 0:
        for i, class_id in enumerate(class_ids):
            if class_id < num_classes:
                scores_all[i, class_id] = scores[i]

    return boxes, scores, class_ids, scores_all


def yolo_pytorchyolo_detect(
        model, img: np.ndarray, conf_threshold: float, num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Детекция с использованием YOLOv3 из pytorchyolo
    """
    # Конвертируем в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Выполняем детекцию
    detections = detect.detect_image(model, img_rgb, conf_thres=conf_threshold, nms_thres=0.45)

    if detections is None or len(detections) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, num_classes))

    detections = np.array(detections)
    boxes = detections[:, :4]
    scores = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # Создаем scores_all для совместимости
    scores_all = np.zeros((len(class_ids), num_classes))
    for i, cid in enumerate(class_ids):
        if cid < num_classes:
            scores_all[i, cid] = scores[i]

    return boxes, scores, class_ids, scores_all