import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class IrisAuthenticator:
    def __init__(self, model_path, device):
        self.device = device
        import torchvision.models as models
        
        auth_backbone = models.resnet18()
        auth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        auth_backbone.fc = nn.Linear(auth_backbone.fc.in_features, 512) # Giá trị khởi tạo tạm thời
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            raw_weights = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            raw_weights = checkpoint["state_dict"]
        else:
            raw_weights = checkpoint

        # Dịch chuyển cấu trúc tầng lớp tùy biến
        clean_weights = {}
        has_custom_embedding = False
        for k, v in raw_weights.items():
            if k.startswith("backbone."):
                clean_weights[k.replace("backbone.", "")] = v
            elif k.startswith("embedding."):
                clean_weights[k.replace("embedding.", "fc.")] = v
                has_custom_embedding = True
            else:
                clean_weights[k] = v

        if has_custom_embedding and "embedding.0.weight" in raw_weights:
            dim_in = raw_weights["embedding.0.weight"].shape[1]
            dim_mid = raw_weights["embedding.0.weight"].shape[0]
            dim_out = raw_weights["embedding.4.weight"].shape[0]
            auth_backbone.fc = nn.Sequential(
                nn.Linear(dim_in, dim_mid),
                nn.BatchNorm1d(dim_mid),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(dim_mid, dim_out)
            )

        self.model = auth_backbone.to(self.device)
        self.model.load_state_dict(clean_weights, strict=False)
        self.model.eval()
        self.transform = transforms.ToTensor()

    def extract_feature_vector(self, enhanced_gray_strip):
        tensor = self.transform(enhanced_gray_strip).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.model(tensor)
            vector = F.normalize(vector, p=2, dim=1)
        return vector.cpu().numpy().flatten()