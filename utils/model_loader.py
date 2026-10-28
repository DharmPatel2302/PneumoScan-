import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# CLASS LABELS (must match training order exactly)
# ============================================================

DISEASE_CLASSES = [
    "no_finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Consolidation",
    "Edema",
    "Pneumonia",
    "Pneumothorax",
    "Infiltration",
    "Lung_Lesion",
    "Pleural_Thickening",
]

NUM_CLASSES = len(DISEASE_CLASSES)   # 11


# ============================================================
# MODEL ARCHITECTURE
# The new model wraps DenseNet121 inside a container:
#
#   model.densenet.features   ← DenseNet feature extractor
#   model.densenet.classifier ← Sequential(
#       Linear(1024, 512),
#       BatchNorm1d(512),
#       ReLU(),
#       Dropout(0.3),
#       Linear(512, 11)       ← 11-class head
#   )
# ============================================================

class ChestXRayModel(nn.Module):
    """Wrapper that matches the training checkpoint exactly."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        base = models.densenet121(weights=None)
        in_features = base.classifier.in_features  # 1024

        # Replace original classifier with the new head
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.densenet = base

    def forward(self, x):
        return self.densenet(x)


# ============================================================
# LOADER
# ============================================================

def load_model(model_path: str, device):
    """
    Load the multiclass chest X-ray model from a .pt checkpoint.

    The checkpoint is a dict with keys:
        model_state_dict, disease_cols, num_classes, ...
    State-dict keys are prefixed with 'densenet.' which matches
    ChestXRayModel.densenet.*
    """

    model = ChestXRayModel(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        # Prefer explicit keys
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip DataParallel "module." prefix if present
        cleaned = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(cleaned, strict=True)

    else:
        # Entire model object was saved
        model = checkpoint

    # Ensure ReLUs are non-inplace for Grad-CAM hooks
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model = model.to(device)
    model.eval()

    return model
