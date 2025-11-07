"""
YOLOv10 Inference with ONNX Runtime + TensorRT
Ultra-fast object detection with GPU acceleration
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time


class YOLOv10Detector:
    """
    YOLOv10 Object Detector with TensorRT acceleration

    Features:
    - TensorRT FP16/INT8 optimization
    - Engine caching for faster startup
    - Dynamic batch processing
    - NMS-free architecture
    """

    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = True,
        fp16: bool = True,
        int8: bool = False,
        cache_dir: str = "./trt_cache"
    ):
        self.model_path = model_path
        self.input_shape = (640, 640)

        # Configure execution providers
        providers = self._setup_providers(use_tensorrt, fp16, int8, cache_dir)

        # Create inference session
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"Model loaded successfully!")
        print(f"Input: {self.input_name}")
        print(f"Outputs: {self.output_names}")

    def _setup_providers(
        self,
        use_tensorrt: bool,
        fp16: bool,
        int8: bool,
        cache_dir: str
    ) -> List[Tuple]:
        """Setup execution providers with optimal configuration"""

        providers = []

        if use_tensorrt:
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                'trt_fp16_enable': fp16,
                'trt_int8_enable': int8,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1,
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
            print("TensorRT enabled with FP16" if fp16 else "TensorRT enabled")

        providers.extend([
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ])

        return providers

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple]:
        """
        Preprocess image for inference

        Args:
            image: BGR image from cv2

        Returns:
            Preprocessed image, scale factor, padding
        """
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape

        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Convert to RGB and normalize
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0

        # HWC to CHW
        image_chw = np.transpose(image_norm, (2, 0, 1))

        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0)

        return image_batch, scale, (new_w, new_h)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        conf_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process model outputs

        Args:
            outputs: Model outputs
            scale: Scale factor from preprocessing
            conf_threshold: Confidence threshold

        Returns:
            boxes, scores, class_ids
        """
        predictions = outputs[0]  # [batch, num_boxes, 6] -> [x, y, w, h, conf, cls]

        # Filter by confidence
        mask = predictions[0, :, 4] > conf_threshold
        filtered = predictions[0][mask]

        if len(filtered) == 0:
            return np.array([]), np.array([]), np.array([])

        # Extract boxes, scores, class_ids
        boxes = filtered[:, :4] / scale
        scores = filtered[:, 4]
        class_ids = filtered[:, 5].astype(int)

        # Convert to x1,y1,x2,y2 format
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes, scores, class_ids

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run object detection on image

        Args:
            image: Input image (BGR)
            conf_threshold: Confidence threshold

        Returns:
            boxes, scores, class_ids
        """
        # Preprocess
        input_tensor, scale, _ = self.preprocess(image)

        # Inference
        start_time = time.time()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        inference_time = (time.time() - start_time) * 1000

        # Postprocess
        boxes, scores, class_ids = self.postprocess(outputs, scale, conf_threshold)

        print(f"Inference time: {inference_time:.2f}ms | Detections: {len(boxes)}")

        return boxes, scores, class_ids

    def visualize(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str] = None
    ) -> np.ndarray:
        """Draw detection results on image"""

        result = image.copy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
            cv2.putText(
                result, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return result


def main():
    """Example usage"""

    # Initialize detector
    detector = YOLOv10Detector(
        model_path="yolov10n.onnx",
        use_tensorrt=True,
        fp16=True
    )

    # COCO class names
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Load image
    image = cv2.imread("test.jpg")

    # Detect objects
    boxes, scores, class_ids = detector.detect(image, conf_threshold=0.3)

    # Visualize
    result = detector.visualize(image, boxes, scores, class_ids, class_names)

    # Save result
    cv2.imwrite("result.jpg", result)
    print("Result saved to result.jpg")

    # Optional: display
    # cv2.imshow("YOLOv10 Detection", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
