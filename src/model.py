import io
import time
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

class ImageModel:
    """ResNet-18 Image Classification model for GPU/CPU inference.
    
    Handles image preprocessing and batched inference. When running on CUDA,
    measures precise GPU kernel execution time using torch.cuda.Event.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable. Install CUDA-enabled PyTorch and drivers.')

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights).to(self.device).eval()
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta['categories']

        # Warm-up passes to initialize CUDA context and engine
        with torch.inference_mode():
            for _ in range(5):
                self.model(torch.zeros((8, 3, 224, 224), device=self.device))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def prepare(self, content: bytes) -> torch.Tensor:
        """Decode image bytes and apply canonical ResNet transformations on CPU."""
        with Image.open(io.BytesIO(content)) as im:
            return self.transform(im.convert('RGB'))

    @torch.inference_mode()
    def infer(self, tensors: list) -> tuple:
        """Run batched forward pass on the target device.
        
        Returns:
            tuple: (predictions_list, gpu_kernel_time_ms)
        """
        x = torch.stack(tensors).to(self.device)

        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            logits = self.model(x)
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = round(start_event.elapsed_time(end_event), 3)
        else:
            t0 = time.perf_counter()
            logits = self.model(x)
            gpu_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        idx = logits.argmax(dim=1).cpu().tolist()
        predictions = [{'class_id': int(i), 'label': self.categories[i]} for i in idx]
        return predictions, gpu_time_ms
