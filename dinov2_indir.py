import torch
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.makedirs("modeller", exist_ok=True)

print("DINOv2 ViT-S/14 indiriliyor... (~85 MB)")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
torch.save(model.state_dict(), "modeller/dinov2_vits14.pth")

boyut = os.path.getsize("modeller/dinov2_vits14.pth") / (1024*1024)
print(f"Tamamlandi: modeller/dinov2_vits14.pth ({boyut:.0f} MB)")