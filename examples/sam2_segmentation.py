"""
SAM 2 (Segment Anything Model 2) with ONNX Runtime + TensorRT
Real-time image and video segmentation
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import time


class SAM2Segmenter:
    """
    SAM 2 Segmentation with TensorRT acceleration

    Features:
    - Image segmentation
    - Video segmentation
    - Point/box prompts
    - Zero-shot capability
    - Real-time performance
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        use_tensorrt: bool = True,
        fp16: bool = True
    ):
        print("Initializing SAM 2...")

        # Setup providers
        providers = self._setup_providers(use_tensorrt, fp16)

        # Load encoder
        print(f"Loading encoder: {encoder_path}")
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)

        # Load decoder
        print(f"Loading decoder: {decoder_path}")
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        self.image_size = 1024
        print("SAM 2 loaded successfully!")

    def _setup_providers(self, use_tensorrt: bool, fp16: bool) -> List:
        """Configure execution providers"""

        providers = []

        if use_tensorrt:
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 6 * 1024 * 1024 * 1024,  # 6GB
                'trt_fp16_enable': fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_sam2_cache',
            }
            providers.append(('TensorrtExecutionProvider', trt_options))

        providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return providers

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for encoder"""

        # Resize to model input size
        target_size = self.image_size
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        padded[:new_h, :new_w] = resized

        # Normalize
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        normalized = (padded - mean) / std

        # HWC to CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(chw, axis=0).astype(np.float32)

        return batched

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to get image embeddings

        Args:
            image: Input image (BGR)

        Returns:
            Image embeddings
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Run encoder
        start_time = time.time()
        embeddings = self.encoder.run(None, {
            self.encoder.get_inputs()[0].name: input_tensor
        })[0]
        encode_time = (time.time() - start_time) * 1000

        print(f"Image encoding time: {encode_time:.2f}ms")
        return embeddings

    def segment_with_points(
        self,
        image_embeddings: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image using point prompts

        Args:
            image_embeddings: Encoded image features
            points: Point coordinates (N, 2) in original image space
            labels: Point labels (N,) - 1 for foreground, 0 for background
            original_size: Original image size (h, w)

        Returns:
            masks, scores
        """
        # Scale points to model input size
        h, w = original_size
        scale = self.image_size / max(h, w)
        scaled_points = points * scale

        # Prepare decoder inputs
        point_coords = np.expand_dims(scaled_points, axis=0).astype(np.float32)
        point_labels = np.expand_dims(labels, axis=0).astype(np.float32)

        # Run decoder
        start_time = time.time()
        outputs = self.decoder.run(None, {
            'image_embeddings': image_embeddings,
            'point_coords': point_coords,
            'point_labels': point_labels,
        })
        decode_time = (time.time() - start_time) * 1000

        masks, scores = outputs[0], outputs[1]

        print(f"Mask decoding time: {decode_time:.2f}ms")
        print(f"Generated {masks.shape[1]} masks with scores: {scores[0]}")

        return masks, scores

    def segment_with_box(
        self,
        image_embeddings: np.ndarray,
        box: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image using box prompt

        Args:
            image_embeddings: Encoded image features
            box: Bounding box [x1, y1, x2, y2] in original image space
            original_size: Original image size (h, w)

        Returns:
            masks, scores
        """
        # Convert box to point prompts
        x1, y1, x2, y2 = box
        points = np.array([
            [x1, y1],  # Top-left
            [x2, y2],  # Bottom-right
        ], dtype=np.float32)
        labels = np.array([2, 3], dtype=np.float32)  # Box corner labels

        return self.segment_with_points(
            image_embeddings, points, labels, original_size
        )

    def visualize_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay mask on image"""

        # Resize mask to image size
        h, w = image.shape[:2]
        mask_resized = cv2.resize(
            mask.squeeze(),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask_resized > 0.5] = color

        # Blend with original image
        result = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        return result


def main():
    """Example usage"""

    # Initialize SAM 2
    segmenter = SAM2Segmenter(
        encoder_path="models/sam2_encoder.onnx",
        decoder_path="models/sam2_decoder.onnx",
        use_tensorrt=True,
        fp16=True
    )

    # Load image
    image = cv2.imread("test.jpg")
    h, w = image.shape[:2]

    # Encode image
    print("\nEncoding image...")
    embeddings = segmenter.encode_image(image)

    # Example 1: Segment with point prompt
    print("\n=== Point Prompt Example ===")
    points = np.array([[w//2, h//2]], dtype=np.float32)  # Center point
    labels = np.array([1], dtype=np.float32)  # Foreground

    masks, scores = segmenter.segment_with_points(
        embeddings, points, labels, (h, w)
    )

    # Use best mask
    best_mask_idx = np.argmax(scores[0])
    best_mask = masks[0, best_mask_idx]

    # Visualize
    result = segmenter.visualize_mask(image, best_mask, color=(0, 255, 0))
    cv2.imwrite("result_point.jpg", result)
    print("Point segmentation saved to result_point.jpg")

    # Example 2: Segment with box prompt
    print("\n=== Box Prompt Example ===")
    box = np.array([w//4, h//4, 3*w//4, 3*h//4], dtype=np.float32)

    masks, scores = segmenter.segment_with_box(embeddings, box, (h, w))

    best_mask_idx = np.argmax(scores[0])
    best_mask = masks[0, best_mask_idx]

    # Visualize with box
    result = segmenter.visualize_mask(image, best_mask, color=(255, 0, 0))
    cv2.rectangle(
        result,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255, 255, 0), 2
    )
    cv2.imwrite("result_box.jpg", result)
    print("Box segmentation saved to result_box.jpg")

    print("\nSegmentation complete!")


if __name__ == "__main__":
    main()
