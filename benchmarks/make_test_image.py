from pathlib import Path
import numpy as np
from PIL import Image

def generate_synthetic_image():
    """Generate a 256x256 random RGB JPEG image for benchmarking latency and throughput."""
    root_dir = Path(__file__).resolve().parent.parent
    target_dir = root_dir / 'test_images'
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / 'synthetic.jpg'

    rng = np.random.default_rng(42)
    img_array = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(out_file, quality=90)
    print(f"Created {out_file.relative_to(root_dir)} (synthetic benchmark image; label has no scientific meaning).")

if __name__ == '__main__':
    generate_synthetic_image()
