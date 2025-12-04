import torch
import cv2
import numpy as np
import json
import os
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_calibration_data():
    """Load calibration data for arrow distance correction"""
    calibration_file = 'distance_calibration.json'

    # Use interpolation-based calibration from distance_calibration.json
    # This is camera-only calibration - no additional sensors needed
    use_simple_scale = False  # Set to True only for quick testing with fixed scale factor

    if use_simple_scale:
        print("Arrow detector: Using simple scale factor (0.35) - interpolation disabled")
        return 0.35  # Simple scale factor (use interpolation instead for better accuracy)

    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                measurements = data.get('measurements', [])
                if len(measurements) >= 3:
                    print(f"Arrow detector loaded {len(measurements)} calibration measurements")
                    return measurements
                else:
                    print("Arrow detector: Not enough calibration data, using default")
                    return 0.35
        except Exception as e:
            print(f"Error loading calibration for arrow detector: {e}")
    else:
        print("No calibration file found for arrow detector, using default scale factor of 0.35")
    return 0.35

def apply_distance_calibration(raw_distance, calibration_data):
    """Apply calibration correction to raw distance measurement"""
    if isinstance(calibration_data, (int, float)):
        return raw_distance * calibration_data

    if not calibration_data or len(calibration_data) < 3:
        return raw_distance * 0.131

    # Interpolation-based calibration
    measured_distances = [m['measured_distance'] for m in calibration_data]
    actual_distances = [m['actual_distance'] for m in calibration_data]

    # Sort by measured distance for proper interpolation
    sorted_pairs = sorted(zip(measured_distances, actual_distances))
    measured_distances, actual_distances = zip(*sorted_pairs)

    try:
        corrected_distance = np.interp(raw_distance, measured_distances, actual_distances)
        return corrected_distance
    except:
        # Fallback to simple ratio if interpolation fails
        ratios = [a/m for a, m in zip(actual_distances, measured_distances)]
        avg_ratio = np.mean(ratios) if ratios else 0.131
        return raw_distance * avg_ratio


class ArrowDetector():
    """
    Arrow detection class supporting multiple YOLO models:
    - YOLOv5: Week_9.pt (Right, Left, Bullseye)
    - YOLOv8/v11: FYP/yolo11n.pt (left arrow, right arrow, up arrow)
    Integrates with AVEP's existing distance calculation system
    """
    def __init__(self, model_path='/Users/tanjunhern/Documents/GitHub/AVEP/pthere/Week_9.pt', optimize_for_distance=True):
        self.model_path = model_path

        # Detect if YOLOv5 or YOLOv8 model based on path
        self.is_yolov8 = False

        # Check if model path suggests YOLOv8/v11/ultralytics or arrow_detection repo
        # Try ultralytics (YOLO v8/v11) first for newer models, fallback to YOLOv5
        if ('pyson3' in model_path or 'yolov8' in model_path.lower() or
            'yolo11' in model_path.lower() or 'FYP' in model_path or
            'arrow_detection' in model_path):
            # Load as YOLOv8/v11 (ultralytics)
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                self.is_yolov8 = True
                print(f'Loaded as YOLOv8/v11 model (ultralytics)')
            except Exception as e:
                print(f'Failed to load as YOLOv8/v11: {e}')
                raise
        else:
            # Try YOLOv5 first (for Week_9.pt, pthere models)
            try:
                self.yolo_model = torch.hub.load('./', 'custom', path=model_path, source='local').to(device)
                self.is_yolov8 = False
                print(f'Loaded as YOLOv5 model')
            except Exception as e:
                # If YOLOv5 fails, try ultralytics as fallback
                print(f'YOLOv5 loading failed, trying ultralytics...')
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(model_path)
                    self.is_yolov8 = True
                    print(f'Loaded as YOLOv8/v11 model (ultralytics)')
                except Exception as e2:
                    print(f'Failed to load with both YOLOv5 and ultralytics')
                    print(f'YOLOv5 error: {e}')
                    print(f'Ultralytics error: {e2}')
                    raise

        # Optimized settings for far detection
        if self.is_yolov8:
            # YOLOv8 uses different parameter setting
            if optimize_for_distance:
                self.conf_threshold = 0.3   # Lower confidence for far arrows
                self.iou_threshold = 0.3    # Lower IoU to catch more detections
            else:
                self.conf_threshold = 0.6   # Standard confidence threshold
                self.iou_threshold = 0.45   # Standard NMS IoU threshold
        else:
            # YOLOv5 settings
            if optimize_for_distance:
                self.yolo_model.conf = 0.3   # Lower confidence for far arrows
                self.yolo_model.iou = 0.3    # Lower IoU to catch more detections
            else:
                self.yolo_model.conf = 0.6   # Standard confidence threshold
                self.yolo_model.iou = 0.45   # Standard NMS IoU threshold

        self.calibration_data = load_calibration_data()  # Load from calibration file
        print(f'Arrow Detection model loaded from: {model_path}')

        if self.is_yolov8:
            print(f'Confidence threshold: {self.conf_threshold}')
            print(f'IoU threshold: {self.iou_threshold}')
        else:
            print(f'Confidence threshold: {self.yolo_model.conf}')
            print(f'IoU threshold: {self.yolo_model.iou}')

        if isinstance(self.calibration_data, list):
            print(f'Distance calibration: Using interpolation with {len(self.calibration_data)} points')
        else:
            print(f'Distance calibration factor: {self.calibration_data}')

        # Get actual model classes
        try:
            if self.is_yolov8:
                actual_model_classes = self.yolo_model.names
                print(f'\nModel classes:')
                for idx, name in actual_model_classes.items():
                    print(f'  {idx}: {name}')
            else:
                actual_model_classes = self.yolo_model.names
                print(f'\nModel classes:')
                for idx, name in actual_model_classes.items():
                    print(f'  {idx}: {name}')
        except Exception as e:
            print(f'Could not retrieve model classes: {e}')

        # Arrow classes for Week_9.pt model
        # Model detects: Right, Left, Bullseye
        self.arrow_classes = {
            0: 'Right',
            1: 'Left',
            2: 'Bullseye'
        }

    def detect_arrows(self, img, enhanced=True, detection_size=1280):
        """
        Detect arrows in the image with optimization for far detection
        Returns detection results compatible with AVEP's distance calculation
        """
        processed_img = img

        # Apply image enhancement for better far detection
        if enhanced:
            processed_img = self._enhance_image(img)

        if self.is_yolov8:
            # YOLOv8 inference
            results = self.yolo_model(processed_img, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=detection_size, verbose=False)

            # Convert YOLOv8 results to pandas format for compatibility
            items_list = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls = int(box.cls[0].cpu().item())
                    name = result.names[cls]

                    items_list.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': conf,
                        'class': cls,
                        'name': name,
                        'arrow_type': name,
                        'labels': name
                    })

            items = pd.DataFrame(items_list)
        else:
            # YOLOv5 inference
            results = self.yolo_model(processed_img[...,::-1], size=detection_size)

            # Convert results to pandas format for compatibility
            items = results.pandas().xyxy[0]

            # Add arrow type labels
            if not items.empty:
                items['arrow_type'] = items['name'].copy()
                items['labels'] = items['name'].copy()

        return items

    def _enhance_image(self, image):
        """
        Apply image enhancements to improve far arrow detection
        """
        import cv2
        import numpy as np

        enhanced = image.copy()

        # 1. Sharpen image for better edge detection
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # 2. Increase contrast for better visibility
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

        # 3. Reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def detect_arrows_multiscale(self, img, scales=[1.0, 1.3, 1.6]):
        """
        Detect arrows using multiple scales for better far detection
        """
        import cv2
        import numpy as np

        all_detections = []
        original_height, original_width = img.shape[:2]

        for scale in scales:
            if scale != 1.0:
                # Resize image
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_img = cv2.resize(img, (new_width, new_height))
            else:
                scaled_img = img

            # Run detection with enhancement
            detections = self.detect_arrows(scaled_img, enhanced=True, detection_size=1280)
            arrow_info = self.get_arrow_info(detections)

            # Scale coordinates back to original size
            if scale != 1.0:
                for arrow in arrow_info:
                    arrow['bbox'] = [
                        int(arrow['bbox'][0] / scale),
                        int(arrow['bbox'][1] / scale),
                        int(arrow['bbox'][2] / scale),
                        int(arrow['bbox'][3] / scale)
                    ]

            all_detections.extend(arrow_info)

        # Remove duplicate detections
        return self._remove_duplicates(all_detections)

    def _remove_duplicates(self, detections, iou_threshold=0.5):
        """
        Remove duplicate detections using Non-Maximum Suppression
        """
        import cv2
        import numpy as np

        if len(detections) == 0:
            return []

        # Convert to format needed for NMS
        boxes = []
        scores = []

        for det in detections:
            boxes.append(det['bbox'])
            scores.append(det['confidence'])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.1, iou_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return detections

    def get_arrow_info(self, detection_results, apply_confidence_boost=True, swap_left_right=False):
        """
        Extract arrow information for distance calculation and display
        Applies confidence boost for LEFT arrows to compensate for detection bias
        Returns: list of dictionaries with arrow detection info
        """
        arrows_info = []

        for _, detection in detection_results.iterrows():
            arrow_type = detection['name']
            confidence = detection['confidence']

            # Swap LEFT and RIGHT labels if model has them reversed
            if swap_left_right:
                if self._is_left_arrow(arrow_type):
                    # Change Left to Right
                    if 'left' in arrow_type.lower():
                        arrow_type = arrow_type.replace('Left', 'Right').replace('left', 'right').replace('LEFT', 'RIGHT')
                elif self._is_right_arrow(arrow_type):
                    # Change Right to Left
                    if 'right' in arrow_type.lower():
                        arrow_type = arrow_type.replace('Right', 'Left').replace('right', 'left').replace('RIGHT', 'LEFT')

            arrow_info = {
                'bbox': [int(detection['xmin']), int(detection['ymin']),
                        int(detection['xmax']), int(detection['ymax'])],
                'confidence': confidence,
                'arrow_type': arrow_type,
                'class_id': int(detection['class'])
            }
            arrows_info.append(arrow_info)

        return arrows_info

    def _is_left_arrow(self, arrow_type):
        """Check if arrow type is a left arrow (handles various naming conventions)"""
        arrow_type_lower = str(arrow_type).lower()
        return 'left' in arrow_type_lower

    def _is_right_arrow(self, arrow_type):
        """Check if arrow type is a right arrow (handles various naming conventions)"""
        arrow_type_lower = str(arrow_type).lower()
        return 'right' in arrow_type_lower

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection area
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def resolve_left_right_conflicts(self, arrow_info_list, iou_threshold=0.5):
        """
        Resolve conflicts when LEFT and RIGHT arrows are detected in overlapping regions
        Strategy: Keep the detection with higher confidence after applying left arrow boost
        """
        if len(arrow_info_list) <= 1:
            return arrow_info_list

        # Find pairs of left/right detections that overlap
        to_remove = set()

        for i in range(len(arrow_info_list)):
            if i in to_remove:
                continue

            arrow_i = arrow_info_list[i]
            is_left_i = self._is_left_arrow(arrow_i['arrow_type'])
            is_right_i = self._is_right_arrow(arrow_i['arrow_type'])

            if not (is_left_i or is_right_i):
                continue

            for j in range(i + 1, len(arrow_info_list)):
                if j in to_remove:
                    continue

                arrow_j = arrow_info_list[j]
                is_left_j = self._is_left_arrow(arrow_j['arrow_type'])
                is_right_j = self._is_right_arrow(arrow_j['arrow_type'])

                # Check if one is LEFT and the other is RIGHT
                if (is_left_i and is_right_j) or (is_right_i and is_left_j):
                    # Calculate overlap
                    iou = self._calculate_iou(arrow_i['bbox'], arrow_j['bbox'])

                    if iou > iou_threshold:
                        # Overlapping LEFT and RIGHT detected - keep higher confidence
                        if arrow_i['confidence'] > arrow_j['confidence']:
                            to_remove.add(j)
                            print(f"[Arrow Conflict] Removed {arrow_j['arrow_type']} (conf={arrow_j['confidence']:.2f}), kept {arrow_i['arrow_type']} (conf={arrow_i['confidence']:.2f})")
                        else:
                            to_remove.add(i)
                            print(f"[Arrow Conflict] Removed {arrow_i['arrow_type']} (conf={arrow_i['confidence']:.2f}), kept {arrow_j['arrow_type']} (conf={arrow_j['confidence']:.2f})")
                            break

        # Return filtered list
        return [arrow for idx, arrow in enumerate(arrow_info_list) if idx not in to_remove]

    def filter_arrows_by_confidence(self, items, min_confidence=0.5):
        """
        Filter detections by confidence threshold
        """
        return items[items['confidence'] >= min_confidence]

    def calculate_calibrated_distance(self, depth_values):
        """
        Calculate calibrated distance from depth values using calibration data
        """
        if len(depth_values) > 0:
            raw_distance = np.median(depth_values)
            calibrated_distance = apply_distance_calibration(raw_distance, self.calibration_data)
            return calibrated_distance
        return None

    def get_model_classes(self):
        """
        Get the class names from the loaded model
        """
        try:
            return self.yolo_model.names
        except:
            return self.arrow_classes
