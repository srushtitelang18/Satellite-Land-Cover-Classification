import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model (DeepLabV3+)
# -----------------------------
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=8,
    classes=5
)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded")

# -----------------------------
# Load Image Tile
# -----------------------------
img = np.load("tiles/images/img_0.npy")

# Normalize
img = img / 255.0

# Convert to tensor
img = torch.tensor(img).unsqueeze(0).float().to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    pred = model(img)

# Convert to class labels
pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

# -----------------------------
# Debug: Check classes
# -----------------------------
unique, counts = np.unique(pred, return_counts=True)
print("Pixel distribution:", dict(zip(unique, counts)))

# -----------------------------
# Color Mapping
# -----------------------------
def colorize(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 0] = [0, 0, 0]       # background (black)
    color[mask == 1] = [255, 0, 0]     # roads (red)
    color[mask == 2] = [0, 255, 0]     # vegetation (green)
    color[mask == 3] = [0, 0, 255]     # water (blue)
    color[mask == 4] = [255, 255, 255] # buildings (white)

    return color

colored = colorize(pred)

# -----------------------------
# Show Output
# -----------------------------
plt.imshow(colored)
plt.title("Prediction (Land Cover Map)")
plt.axis("off")
plt.show()

# -----------------------------
# Save Output
# -----------------------------
plt.imsave("prediction_output.png", colored)

print("✅ Saved as prediction_output.png")