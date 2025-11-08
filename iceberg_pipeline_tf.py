import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 128
N_CHANNELS = 3  # RGB-like
N_SAMPLES = 1200
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 12
OUTPUT_DIR = "outputs"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Synthetic image generator
# ---------------------------
def make_ocean_background(size):
    \"\"\"Create a noisy blue ocean background (RGB).\"\"\"
    w, h = size, size
    # base blue
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[..., 0] = 10  # R
    base[..., 1] = 70  # G
    base[..., 2] = 120  # B
    # Per-pixel noise
    noise = (np.random.randn(h, w, 1) * 8).astype(np.int16)
    for c in range(3):
        channel = base[..., c].astype(np.int16) + noise[..., 0]
        channel = np.clip(channel, 0, 255)
        base[..., c] = channel.astype(np.uint8)
    # slight blur
    img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=1))
    return img

def add_iceberg(draw, bbox, rotation=0, roughness=8):
    \"\"\"Draw an iceberg shape as an irregular polygon/ellipse into PIL draw.\"\"\"
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    # Create a rough ellipse polygon
    cx = x0 + w/2
    cy = y0 + h/2
    pts = []
    n = 24
    for i in range(n):
        theta = 2*np.pi*i/n
        rx = (w/2) * (0.8 + 0.2*np.sin(3*theta) + np.random.uniform(-0.07,0.07))
        ry = (h/2) * (0.8 + 0.2*np.cos(2*theta) + np.random.uniform(-0.07,0.07))
        x = cx + rx * np.cos(theta)
        y = cy + ry * np.sin(theta)
        pts.append((x, y))
    # Optionally rotate around center
    if rotation != 0:
        sinr = np.sin(np.deg2rad(rotation))
        cosr = np.cos(np.deg2rad(rotation))
        pts_r = []
        for (x, y) in pts:
            dx = x - cx
            dy = y - cy
            xr = cx + dx * cosr - dy * sinr
            yr = cy + dx * sinr + dy * cosr
            pts_r.append((xr, yr))
        pts = pts_r
    # Draw white-to-cyan iceberg with slight shading
    draw.polygon(pts, fill=(245, 250, 255))
    # Add small cracks / shading
    n_cracks = random.randint(0, 3)
    for _ in range(n_cracks):
        a = random.uniform(0, 2*np.pi)
        p1 = (cx + (w/4)*np.cos(a), cy + (h/4)*np.sin(a))
        p2 = (cx + (w/2)*np.cos(a+0.4), cy + (h/2)*np.sin(a+0.4))
        draw.line([p1, p2], fill=(220,230,240), width=random.randint(1,2))

def generate_sample(img_size=IMG_SIZE, min_ice=0, max_ice=4):
    \"\"\"Generate one synthetic satellite tile and corresponding mask.\"\"\"
    bg = make_ocean_background(img_size)
    mask = Image.new(\"L\", (img_size, img_size), 0)  # single-channel mask
    draw_img = ImageDraw.Draw(bg)
    draw_mask = ImageDraw.Draw(mask)

    n_ice = random.randint(min_ice, max_ice)
    boxes = []
    for _ in range(n_ice):
        sw = random.uniform(0.06, 0.5)  # fraction of image
        sh = sw * random.uniform(0.6, 1.4)
        w = int(sw * img_size)
        h = int(sh * img_size)
        x = random.randint(0, img_size - w - 1)
        y = random.randint(0, img_size - h - 1)
        bbox = (x, y, x + w, y + h)
        rot = random.uniform(-30, 30)
        add_iceberg(draw_img, bbox, rotation=rot)
        add_iceberg(draw_mask, bbox, rotation=rot)
        boxes.append(bbox)

    # Add specular highlight / sun glint occasionally
    if random.random() < 0.25:
        gx = random.randint(0, img_size-1)
        gy = random.randint(0, img_size-1)
        rad = random.randint(6, 20)
        g = Image.new(\"RGB\", (img_size, img_size), (0,0,0))
        gd = ImageDraw.Draw(g)
        gd.ellipse([gx-rad, gy-rad, gx+rad, gy+rad], fill=(255,240,200))
        g = g.filter(ImageFilter.GaussianBlur(radius=rad/2))
        bg = Image.blend(bg, g, alpha=0.08)

    # convert to arrays
    arr_img = np.array(bg).astype(np.float32) / 255.0
    arr_mask = (np.array(mask) > 0).astype(np.uint8)
    return arr_img, arr_mask, boxes

# ---------------------------
# Build dataset
# ---------------------------
print(\"Generating synthetic dataset...\")
images = []
masks = []
bboxes = []
for i in range(N_SAMPLES):
    img, mask, boxes = generate_sample()
    images.append(img)
    masks.append(mask[..., np.newaxis])  # add channel dim
    bboxes.append(boxes)

images = np.stack(images, axis=0)
masks = np.stack(masks, axis=0)

print(\"Images shape:\", images.shape, \"Masks shape:\", masks.shape)

# Train/test (stratify by whether iceberg exists)
has_ice = (masks.max(axis=(1,2,3)) > 0).astype(int)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    images, masks, np.arange(len(images)), test_size=TRAIN_TEST_SPLIT,
    stratify=has_ice, random_state=SEED
)

print(\"Train:\", X_train.shape[0], \"Test:\", X_test.shape[0])

# ---------------------------
# Simple U-Net model (Keras)
# ---------------------------
def conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding=\"same\", activation=\"relu\")(x)
    x = layers.Conv2D(n_filters, 3, padding=\"same\", activation=\"relu\")(x)
    return x

def encoder_block(x, n_filters):
    c = conv_block(x, n_filters)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block(x, skip, n_filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, n_filters)
    return x

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS)):
    inputs = layers.Input(shape=input_shape)
    # encoder
    c1, p1 = encoder_block(inputs, 16)
    c2, p2 = encoder_block(p1, 32)
    c3, p3 = encoder_block(p2, 64)
    # bottleneck
    b = conv_block(p3, 128)
    # decoder
    d3 = decoder_block(b, c3, 64)
    d2 = decoder_block(d3, c2, 32)
    d1 = decoder_block(d2, c1, 16)
    outputs = layers.Conv2D(1, 1, activation=\"sigmoid\")(d1)
    model = models.Model(inputs, outputs)
    return model

model = build_unet()
model.compile(optimizer=\"adam\", loss=\"binary_crossentropy\",
              metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

# ---------------------------
# Data generator (simple)
# ---------------------------
def augment_batch(xb, yb):
    \"\"\"Apply random flips and small rotations.\"\"\"
    out_x = []
    out_y = []
    for x, y in zip(xb, yb):
        if random.random() < 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        if random.random() < 0.5:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)
        # small rotation by multiples of 90 degrees for simplicity
        k = random.choice([0, 1, 2, 3])
        x = np.rot90(x, k)
        y = np.rot90(y, k)
        out_x.append(x)
        out_y.append(y)
    return np.stack(out_x), np.stack(out_y)

def batch_generator(X, Y, batch_size=BATCH_SIZE, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            xb = X[batch_idx].copy()
            yb = Y[batch_idx].copy()
            xb, yb = augment_batch(xb, yb)
            yield xb, yb

# ---------------------------
# Train
# ---------------------------
train_gen = batch_generator(X_train, y_train, BATCH_SIZE)
steps_per_epoch = max(1, X_train.shape[0] // BATCH_SIZE)
val_gen = (X_test, y_test)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ---------------------------
# Evaluate / Predict samples
# ---------------------------
print(\"\\nEvaluating on test set...\")
preds = model.predict(X_test, batch_size=32)
pred_masks = (preds[...,0] > 0.5).astype(np.uint8)

# Compute IoU manually for iceberg-present images
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

ious = []
for yt, yp in zip(y_test, pred_masks):
    ious.append(iou_score(yt.squeeze()>0, yp>0))
print(\"Mean IoU on test:\", float(np.mean(ious)))

# Save model
model.save(\"iceberg_unet.h5\")
print(\"Saved model to iceberg_unet.h5\")


# ---------------------------
# Save sample outputs
# ---------------------------
n_save = 12
os.makedirs(os.path.join(OUTPUT_DIR, \"samples\"), exist_ok=True)
idxs = np.linspace(0, X_test.shape[0]-1, n_save, dtype=int)
for i, idx in enumerate(idxs):
    img = (X_test[idx]*255).astype(np.uint8)
    true_mask = (y_test[idx].squeeze()*255).astype(np.uint8)
    pred_mask = (pred_masks[idx]*255).astype(np.uint8)

    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(img)
    axs[0].set_title(\"Image\")
    axs[0].axis(\"off\")
    axs[1].imshow(true_mask, cmap=\"gray\")
    axs[1].set_title(\"Truth\")
    axs[1].axis(\"off\")
    axs[2].imshow(pred_mask, cmap=\"gray\")
    axs[2].set_title(\"Pred\")
    axs[2].axis(\"off\")
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, \"samples\", f\"sample_{i}.png\")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

print(f\"Saved sample predictions to {os.path.join(OUTPUT_DIR, 'samples')}\")


# ---------------------------
# Simple bounding box extraction from predicted mask (for tracking)
# ---------------------------
import scipy.ndimage as ndi

boxes_out = []
for i, pm in enumerate(pred_masks):
    labeled, ncomp = ndi.label(pm)
    comps = []
    for lab in range(1, ncomp+1):
        comp_mask = (labeled == lab)
        coords = np.argwhere(comp_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        comps.append((x_min, y_min, x_max, y_max))
    boxes_out.append(comps)

# Save a CSV-like summary (small)
import json
with open(os.path.join(OUTPUT_DIR, \"predicted_boxes.json\"), \"w\") as f:
    json.dump({\"sample_count\": len(boxes_out), \"boxes_first10\": boxes_out[:10]}, f)

print(\"Saved predicted bounding boxes summary.\")


# Final message
print(\"\\nDone. Review outputs in the 'outputs' folder. For better realism, add multi-spectral channels, more complex noise, clouds, sun glint, and augmentation.\")
