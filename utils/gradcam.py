import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model, target_layer):

        self.model       = model
        self.target_layer = target_layer

        self.gradients  = None
        self.activations = None
        self._register_hooks()

    # =========================================================
    # HOOKS
    # =========================================================

    def _register_hooks(self):

        def forward_hook(module, inputs, output):
            self.activations = output.detach().clone()
            # Register backward hook on the activation tensor itself
            if output.requires_grad:
                output.register_hook(self._backward_hook)

        def full_backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(full_backward_hook)

    def _backward_hook(self, grad):
        self.gradients = grad.detach().clone()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # =========================================================
    # GENERATE GRAD-CAM
    # target_class_idx:
    #   None  → use the highest-probability class
    #   int   → specific class index (0-10)
    # =========================================================

    def generate(self, input_tensor, target_class_idx=None):

        self.model.eval()
        self.model.zero_grad()
        self.gradients  = None
        self.activations = None

        with torch.enable_grad():

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------
            output = self.model(input_tensor)   # [1, 11]

            # Multiclass: independent sigmoid per class
            probs = torch.sigmoid(output[0])    # [11]

            # Choose which class to explain
            if target_class_idx is None:
                target_class_idx = int(probs.argmax().item())

            score = output[0, target_class_idx]

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------
            score.backward()

            if self.gradients is None or self.activations is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not fire. "
                    "Check that target_layer is inside the model graph."
                )

            # -------------------------------------------------
            # Weighted combination
            # -------------------------------------------------
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, H, W]
            cam = F.relu(cam)

            # -------------------------------------------------
            # Resize to input size
            # -------------------------------------------------
            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze()   # [224, 224]

            # -------------------------------------------------
            # Normalize 0 → 1
            # -------------------------------------------------
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            heatmap = cam.detach().cpu().numpy()

        return heatmap, target_class_idx


# =============================================================
# CREATE OVERLAY
# =============================================================

def create_overlay(image, heatmap, alpha=0.45):
    """Blend original image with JET-coloured Grad-CAM heatmap."""

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
