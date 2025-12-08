import numpy as np
import math

def do_bboxes_intersect(bbox1, bbox2):
    """Проверить пересечение двух bounding boxes."""
    left1, right1, bottom1, top1 = bbox1[0], bbox1[2], bbox1[1], bbox1[3]
    left2, right2, bottom2, top2 = bbox2[0], bbox2[2], bbox2[1], bbox2[3]
    return (max(left1, left2) <= min(right1, right2)) and (max(bottom1, bottom2) <= min(top1, top2))

def distance_between_bboxes(bbox1, bbox2):
    """Расстояние между bounding boxes (0 если пересекаются)."""
    if do_bboxes_intersect(bbox1, bbox2):
        return 0.0
    left1, right1, bottom1, top1 = bbox1[0], bbox1[2], bbox1[1], bbox1[3]
    left2, right2, bottom2, top2 = bbox2[0], bbox2[2], bbox2[1], bbox2[3]
    dx = max(0, max(left1 - right2, left2 - right1))
    dy = max(0, max(bottom1 - top2, bottom2 - top1))
    return math.sqrt(dx**2 + dy**2)

def distance_between_squares(square1_bbox, square2_bbox):
    """Основная функция: расстояние между двумя квадратами по их bbox [xmin, ymin, xmax, ymax]."""
    # Опционально: проверить, что это квадраты (ширина == высота), но если не обязательно, можно пропустить
    width1 = square1_bbox[2] - square1_bbox[0]
    height1 = square1_bbox[3] - square1_bbox[1]
    width2 = square2_bbox[2] - square2_bbox[0]
    height2 = square2_bbox[3] - square2_bbox[1]
    return distance_between_bboxes(square1_bbox, square2_bbox)
