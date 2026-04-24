from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as _TFLiteInterpreter
    except Exception:
        _TFLiteInterpreter = None

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_DIR / "models"

MASK_THRESHOLD = 0.82
MASK_MARGIN = 0.18
GLASSES_THRESHOLD = 0.74
GLASSES_MARGIN = 0.16
MASK_SIGMOID_LABEL = "without_mask"
GLASSES_SIGMOID_LABEL = "without_glasses"


@dataclass
class BinaryPrediction:
    positive_confidence: float
    negative_confidence: float
    predicted_label: str
    output_label: str
    source: str


class BinaryImageClassifier:
    def __init__(self, model_name, labels, sigmoid_output_label):
        self.model_path = MODEL_DIR / model_name
        self.labels = labels
        self.sigmoid_output_label = sigmoid_output_label
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_size = (128, 128)
        self._input_channels = 3
        self.load()

    def load(self):
        if not self.model_path.exists():
            print(f"[Accessory] Model not found: {self.model_path}")
            return
        if _TFLiteInterpreter is None:
            print(f"[Accessory] TFLite interpreter unavailable for {self.model_path.name}")
            return

        self.interpreter = _TFLiteInterpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]["shape"]
        try:
            if len(shape) == 4 and int(shape[3]) in (1, 3):
                self.input_size = (int(shape[1]), int(shape[2]))
                self._input_channels = int(shape[3])
            else:
                self.input_size = (128, 128)
                self._input_channels = 3
        except Exception:
            self.input_size = (128, 128)
            self._input_channels = 3

        print(f"[Accessory] Loaded model: {self.model_path.name}")

    def _normalize_scores(self, output):
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return scores

        if scores.size == 1:
            value = float(scores[0])
            if value < 0.0 or value > 1.0:
                value = 1.0 / (1.0 + np.exp(-value))
            return np.array([np.clip(value, 0.0, 1.0)], dtype=np.float32)

        if np.all(scores >= 0.0) and np.all(scores <= 1.0):
            total = float(scores.sum())
            if 0.98 <= total <= 1.02:
                return scores

        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict(self, image):
        if self.interpreter is None or image is None or image.size == 0:
            return BinaryPrediction(0.0, 1.0, "unavailable", self.sigmoid_output_label, self.model_path.name)

        target_h, target_w = self.input_size
        if self._input_channels == 1:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(processed, (target_w, target_h)).astype(np.float32)
            processed = np.expand_dims(processed, axis=-1)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed = cv2.resize(processed, (target_w, target_h)).astype(np.float32)

        processed = np.expand_dims(processed / 255.0, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], processed)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        probs = self._normalize_scores(raw_output)

        if probs.size == 1:
            output_conf = float(probs[0])
            output_is_positive = self.sigmoid_output_label == self.labels[1]
            if output_is_positive:
                positive_conf = output_conf
                negative_conf = 1.0 - output_conf
            else:
                positive_conf = 1.0 - output_conf
                negative_conf = output_conf
        else:
            positive_conf = float(probs[1])
            negative_conf = float(probs[0])

        predicted_label = self.labels[1] if positive_conf >= negative_conf else self.labels[0]
        return BinaryPrediction(
            round(positive_conf, 4),
            round(negative_conf, 4),
            predicted_label,
            self.sigmoid_output_label,
            self.model_path.name,
        )


class AccessoryDetector:
    def __init__(self):
        self.mask_model = BinaryImageClassifier(
            "mask_model.tflite",
            ["without_mask", "with_mask"],
            MASK_SIGMOID_LABEL,
        )
        self.glasses_model = BinaryImageClassifier(
            "glasses_model.tflite",
            ["without_glasses", "with_glasses"],
            GLASSES_SIGMOID_LABEL,
        )

    @staticmethod
    def _crop_face(frame, face_box):
        try:
            top, right, bottom, left = map(int, face_box)
        except Exception:
            return None

        height = max(1, bottom - top)
        width = max(1, right - left)
        h_pad = int(height * 0.18)
        w_pad = int(width * 0.18)

        return frame[
            max(0, top - h_pad): min(frame.shape[0], bottom + h_pad),
            max(0, left - w_pad): min(frame.shape[1], right + w_pad),
        ]

    @staticmethod
    def _crop_mask_region(face):
        if face is None or face.size == 0:
            return None
        face_height = face.shape[0]
        start = int(face_height * 0.42)
        region = face[start:, :]
        return region if region.size else None

    @staticmethod
    def _crop_glasses_region(face):
        if face is None or face.size == 0:
            return None

        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(
            cv2.merge((clahe.apply(l_channel), a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

        height, width = enhanced.shape[:2]
        top = int(height * 0.22)
        bottom = int(height * 0.46)
        left = int(width * 0.16)
        right = int(width * 0.84)
        region = enhanced[top:bottom, left:right]
        return region if region.size else None

    def analyze(self, frame, face_box):
        face = self._crop_face(frame, face_box)
        if face is None or face.size == 0:
            return {
                "mask": "No Mask",
                "glasses": "No Glasses",
                "confidence": "Low",
                "mask_confidence": 0.0,
                "glasses_confidence": 0.0,
                "source": "detector_unavailable",
            }

        mask_result = self.mask_model.predict(self._crop_mask_region(face))
        glasses_result = self.glasses_model.predict(self._crop_glasses_region(face))

        mask_detected = (
            mask_result.positive_confidence >= MASK_THRESHOLD
            and (mask_result.positive_confidence - mask_result.negative_confidence) >= MASK_MARGIN
        )
        glasses_detected = (
            glasses_result.positive_confidence >= GLASSES_THRESHOLD
            and (glasses_result.positive_confidence - glasses_result.negative_confidence) >= GLASSES_MARGIN
        )

        strongest = max(mask_result.positive_confidence, glasses_result.positive_confidence)
        confidence = "High" if strongest >= 0.88 else "Medium" if strongest >= 0.7 else "Low"

        return {
            "mask": "Mask" if mask_detected else "No Mask",
            "glasses": "Glasses" if glasses_detected else "No Glasses",
            "confidence": confidence,
            "mask_confidence": mask_result.positive_confidence,
            "glasses_confidence": glasses_result.positive_confidence,
            "source": f"{mask_result.source} | {glasses_result.source}",
        }

    def status_summary(self):
        ready = []
        if self.mask_model.interpreter is not None:
            ready.append("mask")
        if self.glasses_model.interpreter is not None:
            ready.append("glasses")
        if not ready:
            return "Accessory models unavailable"
        return f"Accessory detector ready: {', '.join(ready)}"
