import io
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

class ImageModel:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable. Install CUDA-enabled PyTorch and drivers.')
        self.device = torch.device('cuda:0')
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights).to(self.device).eval()
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta['categories']
        with torch.inference_mode():
            for _ in range(10):
                self.model(torch.zeros((8,3,224,224),device=self.device))
        torch.cuda.synchronize()

    def prepare(self, content: bytes):
        with Image.open(io.BytesIO(content)) as im:
            return self.transform(im.convert('RGB'))

    @torch.inference_mode()
    def infer(self, tensors):
        x = torch.stack(tensors).to(self.device)
        logits = self.model(x)
        idx = logits.argmax(dim=1).cpu().tolist()
        return [{'class_id': int(i), 'label': self.categories[i]} for i in idx]
