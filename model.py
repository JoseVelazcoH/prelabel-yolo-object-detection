import torch
import numpy as np
from PIL import Image
import io
import requests
import os
import logging
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class NewModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        self.labels = ['label1', 'label2', 'label3']
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'my_model.pt')
        self.api_key = os.getenv('LABEL_STUDIO_API_KEY')

        super(NewModel, self).__init__(**kwargs)
        logger.info("Initializing NewModel for YOLO")
        logger.info(f"Using device: {self.device}")
        self._load_yolo_model()
        logger.info(f"Backend initialized with {len(self.labels)} classes")

    def _load_yolo_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded from: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def setup(self):
        return {
            'model_version': 'v1.0',
            'model_class': self.__class__.__name__,
            'model_type': 'object_detection',
            'labels': self.labels,
            'input_types': ['image'],
            'output_types': ['rectanglelabels'],
        }

    def health(self):
        return {
            'status': 'UP',
            'model_class': self.__class__.__name__,
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'model_file': self.model_path
        }

    def predict(self, tasks, context, **kwargs):
        predictions = []
        logger.info(f"Processing {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            task_id = task.get('id', f'task_{i}')
            logger.debug(f"Processing task {task_id}")

            image_url = task['data'].get('image')
            if not image_url:
                logger.warning(f"No image found in task {task_id}")
                predictions.append(self._empty_prediction())
                continue

            try:
                image = self._load_image(image_url)
                if image is None:
                    predictions.append(self._empty_prediction())
                    continue

                local_path = None
                if hasattr(self, 'get_local_path'):
                    try:
                        local_path = self.get_local_path(image_url)
                        if not os.path.exists(local_path):
                            local_path = None
                    except:
                        local_path = None

                result = self._predict_yolo(image, task_id, local_path)
                predictions.append(result)

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                predictions.append(self._empty_prediction())

        logger.info(f"Completed predictions: {len(predictions)}")
        return ModelResponse(predictions=predictions)

    def _predict_yolo(self, image, task_id, image_path=None):
        if image_path and os.path.exists(image_path):
            logger.info(f"Using direct path: {image_path}")
            results = self.model(image_path, conf=0.7)
        else:
            image_np = np.array(image)
            logger.info(f"Using numpy array - shape: {image_np.shape}")
            results = self.model(image_np, conf=0.7)

        prediction_result = []

        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            logger.info(f"Number of boxes detected: {len(boxes)}")

            if image_path and os.path.exists(image_path):
                image_height, image_width = results[0].orig_shape
            else:
                image_np = np.array(image)
                image_height, image_width = image_np.shape[:2]

            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().tolist()[0]
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])

                logger.info(f"Detection: class={cls}, confidence={conf:.3f}, bbox={xyxy}")

                x = (xyxy[0] / image_width) * 100
                y = (xyxy[1] / image_height) * 100
                width = ((xyxy[2] - xyxy[0]) / image_width) * 100
                height = ((xyxy[3] - xyxy[1]) / image_height) * 100

                if cls < len(self.labels):
                    label = self.labels[cls]
                else:
                    label = f"clase_{cls}"

                prediction_result.append({
                    'from_name': 'label',
                    'to_name': 'image',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': max(0, min(100, x)),
                        'y': max(0, min(100, y)),
                        'width': max(0, min(100, width)),
                        'height': max(0, min(100, height)),
                        'rectanglelabels': [label]
                    },
                    'score': conf
                })

                logger.info(f"Task {task_id}: {label} (confidence: {conf:.3f})")
        else:
            logger.info(f"No objects were detected in task {task_id}")

        return {
            'result': prediction_result,
            'score': len(prediction_result) / 10,
            'model_version': 'yolo_v1.0'
        }

    def _load_image(self, image_url):
        try:
            if image_url.startswith('http'):
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Token {self.api_key}'
                response = requests.get(image_url, headers=headers, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                ls_url = os.getenv('LABEL_STUDIO_URL').rstrip('/')
                full_url = f"{ls_url}{image_url}"
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Token {self.api_key}'
                response = requests.get(full_url, headers=headers, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_url}: {str(e)}")
            return None

    def _empty_prediction(self):
        return {
            'result': [],
            'score': 0.0,
            'model_version': 'v1.0'
        }

    def fit(self, data):
        annotations = data.get('annotations', [])
        logger.info(f"Received {len(annotations)} annotations")

        return {
            'model_path': self.model_path,
            'status': 'training_not_implemented'
        }
