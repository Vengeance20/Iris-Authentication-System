import cv2
import torch
import torch.nn as nn
from torchvision import transforms

class AntiSpoofPredictor:
    def __init__(self, model_path, device):
        self.device = device
        import torchvision.models as models
        
        # Tạo khung mạng ResNet18 đầu ra 1 neuron theo đúng kích thước train
        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.fc.in_features, 1)
        )
        
        # Load checkpoint an toàn
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            weights = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            weights = checkpoint["model"]
        else:
            weights = checkpoint
            
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_is_real(self, raw_image):
        img_resized = cv2.resize(raw_image, (224, 224))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
        else:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
            probability = torch.sigmoid(output).item()
            print(f"   [Liveness Log] Xác suất mắt thật: {probability*100:.2f}%")
        return probability > 0.5