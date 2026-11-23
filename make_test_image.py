from pathlib import Path
import numpy as np
from PIL import Image
p = Path('test_images')
p.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
Image.fromarray(rng.integers(0,256,size=(256,256,3),dtype=np.uint8)).save(p/'synthetic.jpg',quality=90)
print('Created test_images/synthetic.jpg (synthetic image; classification label is not meaningful)')
