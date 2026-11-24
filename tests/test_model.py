import io
import unittest
import numpy as np
import torch
from PIL import Image
from src.model import ImageModel

class TestImageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ImageModel()

    def test_model_initialization(self):
        """Verify model weights and transforms are properly loaded."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.transform)
        self.assertGreater(len(self.model.categories), 0)
        self.assertIn(self.model.device.type, ('cuda', 'cpu'))

    def test_prepare_image(self):
        """Verify image preprocessing outputs a normalized 3x224x224 float tensor."""
        rng = np.random.default_rng(123)
        raw_arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(raw_arr).save(buf, format='JPEG')
        jpeg_bytes = buf.getvalue()

        tensor = self.model.prepare(jpeg_bytes)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_infer_batch(self):
        """Verify batched inference returns predictions for every input tensor and records execution time."""
        tensors = [torch.zeros((3, 224, 224)) for _ in range(3)]
        preds, gpu_ms = self.model.infer(tensors)

        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn('class_id', p)
            self.assertIn('label', p)
            self.assertIsInstance(p['class_id'], int)
            self.assertIsInstance(p['label'], str)
        self.assertGreaterEqual(gpu_ms, 0.0)

if __name__ == '__main__':
    unittest.main()
