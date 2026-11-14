import copy
import os
import re
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, swin_t, Swin_T_Weights
from peft import LoraConfig, get_peft_model
import peft.import_utils
import peft.tuners.lora.torchao

peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

BACKBONES = {
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1),
}


class DeepfakeDetectionModel(nn.Module):
    def __init__(self, arch="efficientnet_b4", pretrained=True, dropout=0.4, num_classes=1, freeze_backbone=False, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1, proj_dim=0):
        super().__init__()
        self.arch = arch
        self.use_lora = use_lora
        build, weights = BACKBONES[arch]
        self.backbone = build(weights=weights if pretrained else None)

        # Replace the ImageNet classifier with a single-logit head
        if arch == "efficientnet_b4":
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
            head_name = "classifier"
        elif arch == "convnext_tiny":  # classifier = [LayerNorm2d, Flatten, Linear]
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
            head_name = "classifier"
        else:  # swin_t
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
            head_name = "head"

        if use_lora:
            # EfficientNet: LoRA on the 1x1 convolutions. ConvNeXt / Swin: their pointwise and attention layers are nn.Linear.
            if arch == "efficientnet_b4":
                is_target = lambda m: isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)
            else:
                is_target = lambda m: isinstance(m, nn.Linear)
            target_names = [
                re.escape(f"features.{name}")
                for name, m in self.backbone.features.named_modules()
                if is_target(m)
            ]
            target_modules = f"^({'|'.join(target_names)})$"
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=[head_name],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
        elif freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Projection head for contrastive training only. Not part of the exported / inference model.
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, proj_dim),
            )

    def forward(self, img, return_embedding=False):
        if not return_embedding:
            return self.backbone(img)
        net = self.backbone
        if self.arch == "efficientnet_b4":
            feat = net.avgpool(net.features(img)).flatten(1)
            logits = net.classifier(feat)
        elif self.arch == "convnext_tiny":
            pooled = net.avgpool(net.features(img))
            feat = pooled.flatten(1)
            logits = net.classifier(pooled)
        else:  # swin_t
            feat = net.flatten(net.avgpool(net.permute(net.norm(net.features(img)))))
            logits = net.head(feat)
        return logits, nn.functional.normalize(self.proj(feat), dim=1)

    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

    def merged_state_dict(self):
        # Plain backbone + head weights (LoRA folded into the base weights, projection head dropped)
        if self.use_lora:
            merged_backbone = copy.deepcopy(self.backbone).merge_and_unload()
            return {f"backbone.{k}": v for k, v in merged_backbone.state_dict().items()}
        return {k: v for k, v in self.state_dict().items() if not k.startswith("proj.")}

    def export_merged(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.merged_state_dict(), path)


def load_inference_model(arch, state_dict):
    """Plain (no LoRA, no projection head) model for inference from either kind of saved weights:
    a training checkpoint (*_best.pt, LoRA + projection head) or merged weights (*_weights.pt / export_merged)."""
    if any("lora_" in k for k in state_dict):
        proj_dim = state_dict["proj.2.weight"].shape[0] if "proj.2.weight" in state_dict else 0
        lora_model = DeepfakeDetectionModel(arch=arch, pretrained=False, use_lora=True, proj_dim=proj_dim)
        lora_model.load_state_dict(state_dict)
        state_dict = lora_model.merged_state_dict()
    model = DeepfakeDetectionModel(arch=arch, pretrained=False, use_lora=False)
    model.load_state_dict(state_dict)
    return model.eval()
