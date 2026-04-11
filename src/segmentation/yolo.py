"""Segmentação via YOLOv8 Instance Segmentation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from ultralytics import YOLO


class YOLOSegmentation:
    """Wrapper para detecção e segmentação de instâncias com YOLOv8.

    Args:
        model_path: Caminho para o arquivo de pesos do modelo YOLOv8 (.pt).
    """

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """Executa detecção e segmentação em uma imagem.

        Args:
            img: Imagem BGR (OpenCV) de entrada com shape (H, W, C).

        Returns:
            Tupla (bboxes, class_ids, segmentation_contours, scores).
            Retorna quatro tuplas vazias se nenhuma máscara for detectada.
        """
        height, width, _ = img.shape

        result, *_ = self.model.predict(source=img.copy(), save=False, save_txt=False)

        segmentation_contours_idx = []

        try:
            for seg in result.masks.segments:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segmentation_contours_idx.append(np.array(seg, dtype=np.int32))
        except Exception:
            return (), (), (), ()

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores
