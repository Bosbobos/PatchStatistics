import json
import os
from typing import Any

from numpy import ndarray, dtype
from tqdm import tqdm

from Geometry import distance_between_squares
from Models import *
from Auxiliary import *


def apply_patch(
        img: np.ndarray,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        target_class: int,
        patch: np.ndarray,
        patch_size: float,
        out_of_box: bool = False,
        near_box: bool = False,
        location=(0, 0)
) -> tuple[ndarray, list[tuple[int, int] | Any]] | tuple[ndarray, None]:
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

    if out_of_box:
        patch_width = max(5, int(patch.shape[1] * patch_size))
        patch_height = max(5, int(patch.shape[0] * patch_size))
        patch_resized = cv2.resize(patch, (patch_width, patch_height))
        w, h = location
        if h == -1:
            h = img.shape[0] - patch_height
        if w == -1:
            w = img.shape[1] - patch_width
        patched[h:h + patch_height, w:w + patch_width] = patch_resized
        return patched, [w, h, w + patch_width, h + patch_height]

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

    return patched, None


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
        location=(0, 0),
        patch_to_box_min_dist: float = -1.0,
        confidence_threshold: float = -1.0,
        top_k: int = -1,
) -> tuple[
    int, int | Any, int | Any, list[Any], list[Any], ndarray[Any, dtype[Any]] | None, list[Any], list[Any], list[Any]]:
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

    # Определяем бэкенд YOLO (по умолчанию 'ultralytics' для обратной совместимости)
    model_backend_type = model_params.get('model_backend_type', 'yolo_ultralytics')

    orig_h, orig_w = img.shape[:2]

    # Детекция на исходном изображении
    if model_type == 'yolo':
        if model_backend_type == 'yolo_ultralytics':
            boxes, scores, class_ids, scores_all = yolo_detect(model, img, conf_threshold)
        elif model_backend_type == 'yolo_hub':
            boxes, scores, class_ids, scores_all = yolo_hub_detect(model, img, conf_threshold, num_classes)
        elif model_backend_type == 'yolo_pytorchyolo':
            boxes, scores, class_ids, scores_all = yolo_pytorchyolo_detect(model, img, conf_threshold, num_classes)
    else:  # onnx
        blob = preprocess(img, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes, scores, class_ids, scores_all = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

    patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    if patch is None:
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    # Calculate patch_square for filtering (need to know patch location before filtering)
    patch_square = None
    if out_of_box:
        patch_width = max(5, int(patch.shape[1] * patch_size))
        patch_height = max(5, int(patch.shape[0] * patch_size))
        w, h = location
        if h == -1:
            h = img.shape[0] - patch_height
        if w == -1:
            w = img.shape[1] - patch_width
        patch_square = [w, h, w + patch_width, h + patch_height]

    # Filter detections based on patch_to_box_min_dist, confidence_threshold, and top_k
    # This filtering happens BEFORE patch application and affects which boxes are counted
    target_indices = np.where(class_ids == target_class)[0]

    if patch_square is not None and (patch_to_box_min_dist >= 0 or confidence_threshold >= 0 or top_k >= 0):
        # Calculate distances and scores for all target objects
        target_data = []
        for idx in target_indices:
            orig_box = boxes[idx]
            dist = distance_between_squares(orig_box, patch_square)
            orig_score = scores_all[idx, target_class] if scores_all.size > 0 and idx < len(scores_all) else 0
            target_data.append((idx, dist, orig_score))

        # Filter by minimum distance
        if patch_to_box_min_dist >= 0:
            target_data = [(idx, dist, score) for idx, dist, score in target_data if dist > patch_to_box_min_dist]

        # Filter by minimum confidence threshold
        if confidence_threshold >= 0:
            target_data = [(idx, dist, score) for idx, dist, score in target_data if score >= confidence_threshold]

        # Sort by confidence (descending) and take top_k
        if top_k >= 0 and len(target_data) > top_k:
            target_data.sort(key=lambda x: x[2], reverse=True)
            target_data = target_data[:top_k]

        filtered_indices = np.array([idx for idx, _, _ in target_data]) if target_data else np.array([], dtype=int)
    else:
        filtered_indices = target_indices

    # Create filtered versions of detection arrays (only include filtered target objects)
    # For visualization and patch application, we only want to show/process the filtered target objects
    filtered_target_mask = np.zeros(len(class_ids), dtype=bool)
    if len(filtered_indices) > 0:
        filtered_target_mask[filtered_indices] = True

    boxes_filtered = boxes[filtered_target_mask]
    scores_filtered = scores[filtered_target_mask]
    class_ids_filtered = class_ids[filtered_target_mask]
    scores_all_filtered = scores_all[filtered_target_mask] if scores_all.size > 0 else scores_all

    # Применяем настоящий патч к целевым объектам (using filtered detections)
    patched_img, patch_square = apply_patch(
        img, boxes_filtered, class_ids_filtered, target_class, patch, patch_size, out_of_box, near_box, location
    )

    black_patch_path = 'black_patch.png'
    black_patch = cv2.imread(black_patch_path, cv2.IMREAD_UNCHANGED)
    if black_patch is None:
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    # Применяем черный патч к целевым объектам (using filtered detections)
    black_patched_img, black_patch_square = apply_patch(
        img, boxes_filtered, class_ids_filtered, target_class, black_patch, patch_size, out_of_box, near_box, location
    )

    # Детекция на изображении с настоящим патчем
    if model_type == 'yolo':
        if model_backend_type == 'yolo_ultralytics':
            boxes_p, scores_p, class_ids_p, scores_all_p = yolo_detect(model, patched_img, conf_threshold)
            boxes_bp, scores_bp, class_ids_bp, scores_all_bp = yolo_detect(model, black_patched_img, conf_threshold)
        elif model_backend_type == 'yolo_hub':
            boxes_p, scores_p, class_ids_p, scores_all_p = yolo_hub_detect(model, patched_img, conf_threshold,
                                                                           num_classes)
            boxes_bp, scores_bp, class_ids_bp, scores_all_bp = yolo_hub_detect(model, black_patched_img, conf_threshold,
                                                                               num_classes)
        elif model_backend_type == 'yolo_pytorchyolo':
            boxes_p, scores_p, class_ids_p, scores_all_p = yolo_pytorchyolo_detect(model, patched_img, conf_threshold,
                                                                                   num_classes)
            boxes_bp, scores_bp, class_ids_bp, scores_all_bp = yolo_pytorchyolo_detect(model, black_patched_img,
                                                                                       conf_threshold, num_classes)

    else:  # onnx
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
    all_distances = []  # Расстояния от патча до объекта (считается между двумя ближайшими точками)
    successful_real_distances = []
    successful_black_distances = []

    # Use the pre-filtered indices from earlier
    num_targets = len(filtered_indices)

    for idx in filtered_indices:
        orig_box = boxes[idx]
        orig_score = scores_all[idx, target_class] if scores_all.size > 0 and idx < len(scores_all) else 0

        if patch_square is not None:
            dist = distance_between_squares(orig_box, patch_square)
        else:
            dist = 0
        all_distances.append(dist)

        # Проверяем эффективность настоящего патча
        success = False
        found_real = False
        for j, patched_box in enumerate(boxes_p):
            if class_ids_p[j] == target_class:
                iou = calculate_iou(orig_box, patched_box)
                if iou > 0.5:
                    found_real = True
                    patched_score = scores_all_p[j, target_class] if scores_all_p.size > 0 and j < len(
                        scores_all_p) else 0
                    confidence_drop = orig_score - patched_score
                    confidence_drops_real.append(confidence_drop)

                    if orig_score > threshold and patched_score < threshold:
                        success = True
                        num_success_real += 1
                        if patch_square is not None:
                            dist = distance_between_squares(orig_box, patch_square)
                        else:
                            dist = 0
                        successful_real_distances.append(dist)
                    break

        if not found_real:
            confidence_drops_real.append(orig_score)
            if orig_score > threshold:
                success = True
                num_success_real += 1
                if patch_square is not None:
                    dist = distance_between_squares(orig_box, patch_square)
                else:
                    dist = 0
                successful_real_distances.append(dist)

        # Проверяем эффективность черного патча
        found_black = False
        for j, black_patched_box in enumerate(boxes_bp):
            if class_ids_bp[j] == target_class:
                iou = calculate_iou(orig_box, black_patched_box)
                if iou > 0.5:
                    found_black = True
                    black_patched_score = scores_all_bp[j, target_class] if scores_all_bp.size > 0 and j < len(
                        scores_all_bp) else 0
                    confidence_drop = orig_score - black_patched_score
                    confidence_drops_black.append(confidence_drop)

                    if orig_score > threshold and black_patched_score < threshold:
                        num_success_black += 1
                        if black_patch_square is not None:
                            dist = distance_between_squares(orig_box, black_patch_square)
                        else:
                            dist = 0
                        successful_black_distances.append(dist)
                    break

        if not found_black:
            confidence_drops_black.append(orig_score)
            if orig_score > threshold:
                num_success_black += 1
                if black_patch_square is not None:
                    dist = distance_between_squares(orig_box, black_patch_square)
                else:
                    dist = 0
                successful_black_distances.append(dist)

    # Создаем side-by-side изображение с тремя панелями
    result_img = None
    if save_images and num_targets > 0:
        # Создание директории для результатов
        os.makedirs(save_dir, exist_ok=True)

        # Визуализация результатов (use filtered boxes for original image)
        vis_clean = draw(orig_img.copy(), boxes_filtered, scores_filtered, class_ids_filtered, class_names)

        # For patched images, only show boxes that match the original filtered boxes (by IoU > 0.5)
        matched_boxes_p = []
        matched_scores_p = []
        matched_class_ids_p = []
        for idx in filtered_indices:
            orig_box = boxes[idx]
            for j, patched_box in enumerate(boxes_p):
                if class_ids_p[j] == target_class:
                    iou = calculate_iou(orig_box, patched_box)
                    if iou > 0.5:
                        matched_boxes_p.append(patched_box)
                        matched_scores_p.append(scores_p[j])
                        matched_class_ids_p.append(class_ids_p[j])
                        break

        matched_boxes_bp = []
        matched_scores_bp = []
        matched_class_ids_bp = []
        for idx in filtered_indices:
            orig_box = boxes[idx]
            for j, black_patched_box in enumerate(boxes_bp):
                if class_ids_bp[j] == target_class:
                    iou = calculate_iou(orig_box, black_patched_box)
                    if iou > 0.5:
                        matched_boxes_bp.append(black_patched_box)
                        matched_scores_bp.append(scores_bp[j])
                        matched_class_ids_bp.append(class_ids_bp[j])
                        break

        vis_patched = draw(patched_img.copy(),
                           np.array(matched_boxes_p) if matched_boxes_p else np.array([]).reshape(0, 4),
                           np.array(matched_scores_p) if matched_scores_p else np.array([]),
                           np.array(matched_class_ids_p) if matched_class_ids_p else np.array([]),
                           class_names)

        vis_black_patched = draw(black_patched_img.copy(),
                                 np.array(matched_boxes_bp) if matched_boxes_bp else np.array([]).reshape(0, 4),
                                 np.array(matched_scores_bp) if matched_scores_bp else np.array([]),
                                 np.array(matched_class_ids_bp) if matched_class_ids_bp else np.array([]),
                                 class_names)

        # Добавляем подписи
        cv2.putText(vis_clean, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_patched, "Real Patch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_black_patched, "Black Patch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        result_img = np.hstack([vis_clean, vis_patched, vis_black_patched])

        pos = ''
        if out_of_box:
            pos = 'l_' if location[0] == 0 else 'r_' if location[0] == -1 else f'{location[0]}_'
            pos += 'u' if location[1] == 0 else 'd' if location[1] == -1 else f'{location[1]}_'
        
        splitted = os.path.splitext(img_name)
        filename = f"{splitted[0]}_{pos}{splitted[1]}"
        result_path = os.path.join(
            save_dir,
            f"result_{filename}"
        )
        cv2.imwrite(result_path, result_img)

    return (num_targets,
            num_success_real, num_success_black,
            confidence_drops_real, confidence_drops_black,
            result_img,
            all_distances, successful_real_distances, successful_black_distances)


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
        model_backend: str = 'ultralytics',  # 'ultralytics', 'yolov5_hub' или 'pytorchyolo',
        location=(0, 0),
        patch_to_box_min_dist: float = -1.0,
        confidence_threshold: float = -1.0,
        top_k: int = -1,
) -> Dict[str, Any]:
    # Вычисляем производные пути
    patch_path = f"{patch_name}.png"
    if results_dir is None:
        results_dir = f"patched_{patch_name}" + ('_oob' if out_of_box else "") + ('_near_box' if near_box else '')

    # Загрузка модели
    model_info, H, W, num_classes = load_model(model_path, model_backend=model_backend)

    # Определяем тип модели
    model_backend_type = 'onnx'  # По умолчанию
    if isinstance(model_info, tuple):
        # (model, 'yolo', 'yolo_ultralytics'/'yolo_hub'/'yolo_pytorchyolo')
        model, model_type_str, model_backend_type = model_info
    else:
        # model (для ONNX)
        model = model_info
        model_type_str = 'onnx'

    # Загрузка имен классов
    class_names = load_class_names(classes_path)
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]

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
        'model_type': model_type_str,  # 'yolo' или 'onnx'
        'model_backend_type': model_backend_type  # 'yolo_ultralytics', 'yolo_hub', 'yolo_pytorchyolo' или 'onnx'
    }

    # Статистика для обоих типов патчей
    total_targets = 0
    successful_attacks_real = 0
    successful_attacks_black = 0
    confidence_drops_real = []
    confidence_drops_black = []
    all_distances = []
    all_successful_real_distances = []
    all_successful_black_distances = []

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
        (num_targets,
         num_success_real, num_success_black,
         img_drops_real, img_drops_black,
         _,
         distances, successful_real_distances, successful_black_distances) = detect_and_compare(
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
            location=location,
            patch_to_box_min_dist=patch_to_box_min_dist,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )

        # Обновляем статистику
        total_targets += num_targets
        successful_attacks_real += num_success_real
        successful_attacks_black += num_success_black
        confidence_drops_real.extend(img_drops_real)
        confidence_drops_black.extend(img_drops_black)
        all_distances.extend(distances)
        all_successful_real_distances.extend(successful_real_distances)
        all_successful_black_distances.extend(successful_black_distances)
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
        'model_backend': model_backend,
        'patch_name': patch_name,
        'dataset': image_dir,
        'type': 'oob' if out_of_box else 'in box',
        'total_targets': total_targets,
        'successful_attacks_real': successful_attacks_real,
        'successful_attacks_black': successful_attacks_black,
        'target_class': target_class,
        'target_class_name': class_names[target_class] if target_class < len(class_names) else 'unknown',
    }

    if total_targets > 0:
        metrics['asr_real'] = successful_attacks_real / total_targets
        metrics['asr_black'] = successful_attacks_black / total_targets
        metrics['mean_confidence_drop_real'] = float(np.mean(confidence_drops_real)) if confidence_drops_real else 0.0
        metrics['mean_confidence_drop_black'] = float(
            np.mean(confidence_drops_black)) if confidence_drops_black else 0.0
        metrics['relative_effectiveness'] = metrics['asr_real'] - metrics['asr_black']
        metrics['conf_drop'] = metrics['mean_confidence_drop_real'] - metrics['mean_confidence_drop_black']
        metrics['mean_distance'] = np.mean(all_distances)
        metrics['mean_successful_real_dist'] = np.mean(all_successful_real_distances)
        metrics['mean_successful_black_dist'] = np.mean(all_successful_black_distances)
    else:
        metrics['asr_real'] = 0.0
        metrics['asr_black'] = 0.0
        metrics['mean_confidence_drop_real'] = 0.0
        metrics['mean_confidence_drop_black'] = 0.0
        metrics['relative_effectiveness'] = 0.0
        metrics['conf_drop'] = 0.0
        metrics['mean_distance'] = 0.0
        metrics['mean_successful_real_dist'] = 0.0
        metrics['mean_successful_black_dist'] = 0.0

    # Сохранение результатов в файл
    json_results_path = 'results'
    os.makedirs(json_results_path, exist_ok=True)

    results_file = os.path.join(json_results_path,
                                f"{model_path.split('.')[0]}_{model_backend}_{patch_name}_{os.path.basename(image_dir)}_{metrics['type']}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ─────────── Точка входа ───────────
if __name__ == "__main__":
    # Теперь можно выбрать бэкенд для загрузки .pt файла:
    # 'ultralytics' -> использует `from ultralytics import YOLO` (API v8)
    # 'yolov5_hub'  -> использует `torch.hub.load('ultralytics/yolov5', ...)` (API v5)
    import warnings

    warnings.filterwarnings('ignore')

    metrics = run_experiment('yolo11s.pt',
                             'inria_test',
                             model_backend='ultralytics',  # Указываем, что хотим использовать оригинальный yolov5
                             patch_name='0709_yolo_dpatch_1000',
                             save_images=True,
                             out_of_box=True,
                             patch_size=1,
                             location=(0, 0),
                             patch_to_box_min_dist=0.0,
                             confidence_threshold=0.85,
                             top_k=2
                             )
    [print(f"{key}: {value}") for key, value in metrics.items()]

    metrics = run_experiment('yolo11s.pt',
                             'inria_test',
                             model_backend='ultralytics',  # Указываем, что хотим использовать оригинальный yolov5
                             patch_name='0709_yolo_dpatch_1000',
                             save_images=True,
                             out_of_box=True,
                             patch_size=1,
                             location=(0, -1),
                             patch_to_box_min_dist=0.0,
                             confidence_threshold=0.85,
                             top_k=2
                             )
    [print(f"{key}: {value}") for key, value in metrics.items()]

    metrics = run_experiment('yolo11s.pt',
                             'inria_test',
                             model_backend='ultralytics',  # Указываем, что хотим использовать оригинальный yolov5
                             patch_name='0709_yolo_dpatch_1000',
                             save_images=True,
                             out_of_box=True,
                             patch_size=1,
                             location=(-1, 0),
                             patch_to_box_min_dist=0.0,
                             confidence_threshold=0.85,
                             top_k=2
                             )
    [print(f"{key}: {value}") for key, value in metrics.items()]

    metrics = run_experiment('yolo11s.pt',
                             'inria_test',
                             model_backend='ultralytics',  # Указываем, что хотим использовать оригинальный yolov5
                             patch_name='0709_yolo_dpatch_1000',
                             save_images=True,
                             out_of_box=True,
                             patch_size=1,
                             location=(-1, -1),
                             patch_to_box_min_dist=0.0,
                             confidence_threshold=0.85,
                             top_k=2
                             )
    [print(f"{key}: {value}") for key, value in metrics.items()]