import io
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.app import app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_health_endpoint(self):
        """Verify /health returns 200 and reports server status and batch mode."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("mode", data)
        self.assertIn("device", data)

    def test_stats_endpoint(self):
        """Verify /stats returns 200 and telemetry structure."""
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mode", data)
        self.assertIn("processed", data)
        self.assertIn("batches", data)
        self.assertIn("batch_histogram", data)
        self.assertIn("queued", data)

    def test_predict_valid_image(self):
        """Verify /predict returns 200 with prediction metadata on valid image upload."""
        rng = np.random.default_rng(42)
        raw_arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(raw_arr).save(buf, format='JPEG')
        img_bytes = buf.getvalue()

        resp = self.client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("class_id", body)
        self.assertIn("label", body)
        self.assertIn("batch_size", body)
        self.assertIn("queue_ms", body)
        self.assertIn("gpu_batch_ms", body)

    def test_predict_invalid_image(self):
        """Verify /predict returns 400 Bad Request when non-image bytes are uploaded."""
        resp = self.client.post("/predict", files={"file": ("bad.jpg", b"invalid image bytes", "image/jpeg")})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid image", resp.text)

    def test_predict_oversized_payload(self):
        """Verify /predict returns 413 Payload Too Large when payload exceeds size limit."""
        oversized = b"0" * (10_000_001)
        resp = self.client.post("/predict", files={"file": ("big.jpg", oversized, "image/jpeg")})
        self.assertEqual(resp.status_code, 413)

if __name__ == '__main__':
    unittest.main()
