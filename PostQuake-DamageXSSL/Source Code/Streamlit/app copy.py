import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
import math
from collections import Counter
from scipy import ndimage

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import timm
import torch.nn.functional as F

IMG_SIZE = (384, 384)           
NUM_CLASSES = 4
CLASS_NAMES = {0: "no-damage", 1: "minor", 2: "major", 3: "destroyed"}
DEFAULT_CHECKPOINT = r"C:\Users\Asus\Downloads\EARTHQUAKE-TURKEY\EARTHQUAKE-TURKEY\checkpoints_4class\vit_tiny_epoch03_miou0.6246.pth"  

st.set_page_config(page_title="DamageSeg Inference", layout="centered", initial_sidebar_state="expanded")
st.title("📡 Building Damage Segmentation — Inference")
st.markdown("Upload matching pre/post images. Model expects pre+post stacked (6 channels) and images are resized to the training input size.")

# ---------------- Utilities ----------------
def make_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = make_device()
st.sidebar.write(f"Device: **{DEVICE}**")

def preprocess_pil(pil_img, target_size=IMG_SIZE):
    """Return HxWx3 float32 array in [0,1] (same behavior as training TF.to_tensor after resizing)."""
    img = pil_img.convert("RGB")
    img = img.resize(target_size, resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def tensor_from_pair(pre_arr, post_arr, device):
    """
    Return torch tensor 1x6xHxW on device.
    Uses torchvision.transforms.functional.to_tensor semantics (range [0,1]).
    """
    pre_pil = Image.fromarray((pre_arr * 255).astype(np.uint8))
    post_pil = Image.fromarray((post_arr * 255).astype(np.uint8))
    pre_t = TF.to_tensor(pre_pil)   # shape (3,H,W) float in [0,1]
    post_t = TF.to_tensor(post_pil)
    inp = torch.cat([pre_t, post_t], dim=0).unsqueeze(0).to(device)  # 1x6xHxW
    return inp

def overlay_prediction_on_post(post_arr_uint8, pred_mask):
    """Vectorized blending overlay (fast). post_arr_uint8: HxWx3 np.uint8, pred_mask HxW uint8"""
    color_map = {
        1: (255,165,0,140),  # minor orange
        2: (255,69,0,160),   # major
        3: (200,0,0,180)     # destroyed
    }
    H, W = pred_mask.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    for c, col in color_map.items():
        mask = (pred_mask == c)
        if mask.any():
            overlay[mask] = col 
    post_rgba = np.dstack([post_arr_uint8, np.full((H,W),255,dtype=np.uint8)])
    alpha = overlay[...,3:4].astype(np.float32) / 255.0
    out_rgb = (post_rgba[...,:3].astype(np.float32) * (1-alpha) + overlay[...,:3].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out_rgb).convert("RGBA")

def per_building_counts(pred_mask, min_area=8, treat_empty_as_no_damage=True):
    source = (pred_mask > 0).astype(np.uint8)
    labeled, ncomp = ndimage.label(source)
    counts = Counter()
    comps = []
    for comp in range(1, ncomp+1):
        comp_mask = (labeled == comp)
        area = int(comp_mask.sum())
        if area < min_area:
            continue
        vals, cnts = np.unique(pred_mask[comp_mask], return_counts=True)
        mode = int(vals[np.argmax(cnts)])
        counts[mode] += 1
        comps.append((comp, mode, area))
    if sum(counts.values()) == 0 and treat_empty_as_no_damage:
        counts[0] = 1
    return counts, comps

def draw_text_on_image(pil_img, counter, font_size=14, pad=10):
    img = pil_img.convert("RGBA")
    W, H = img.size
    text = " | ".join([f"{CLASS_NAMES[c]}: {counter.get(c,0)}" for c in range(NUM_CLASSES)])
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # wrap
    draw_tmp = ImageDraw.Draw(img)
    max_width = W - 2*pad
    words = text.split(" ")
    wrapped = ""
    line = ""
    for word in words:
        test_line = (line + " " + word).strip()
        try:
            bbox = draw_tmp.textbbox((0,0), test_line, font=font)
            w = bbox[2] - bbox[0]
        except:
            w, _ = draw_tmp.textsize(test_line, font=font)
        if w <= max_width:
            line = test_line
        else:
            wrapped += line + "\n"
            line = word
    wrapped += line
    try:
        bbox = draw_tmp.multiline_textbbox((0,0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]; text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = draw_tmp.textsize(wrapped, font=font)
    banner_h = text_h + 2*pad
    new_img = Image.new("RGBA", (W, H + banner_h), (0,0,0,0))
    new_img.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(new_img)
    draw.rectangle([0, 0, W, banner_h], fill=(0,0,0,150))
    x = (W - text_w) // 2
    y = pad
    draw.multiline_text((x, y), wrapped, font=font, fill=(255,255,255,255), align="center")
    return new_img.convert("RGB")

def resize_vit_pos_embed_for_input(model, img_size):
    """
    Resize model.vit.pos_embed to match the number of tokens for a given input spatial size.
    Assumes model.vit.patch_embed exists and uses square patches (patch_size).
    """
    try:
        vit = model.vit
        if not hasattr(vit, "pos_embed"):
            return
        pos = vit.pos_embed   # shape (1, 1 + old_grid, dim)
        patch_embed = vit.patch_embed
        # try find patch size from patch_embed kernel or from model default
        if hasattr(patch_embed, "kernel_size"):
            if isinstance(patch_embed.kernel_size, tuple):
                patch_size = patch_embed.kernel_size[0]
            else:
                patch_size = patch_embed.kernel_size
        else:
            patch_size = 16
        ph, pw = img_size[0] // patch_size, img_size[1] // patch_size
        new_grid = ph * pw
        old_grid = pos.shape[1] - 1
        if new_grid == old_grid:
            # nothing to do
            return
        # separate cls token and grid
        cls_tok = pos[:, :1, :]
        grid_tok = pos[:, 1:, :].transpose(1,2).reshape(1, pos.shape[2], int(math.sqrt(old_grid)), int(math.sqrt(old_grid)))
        # resize with bilinear
        grid_tok_resized = F.interpolate(grid_tok, size=(ph, pw), mode='bilinear', align_corners=False)
        new_grid_tok = grid_tok_resized.reshape(1, pos.shape[2], ph*pw).transpose(1,2)
        new_pos = torch.cat([cls_tok, new_grid_tok], dim=1).to(pos.device)
        vit.pos_embed = nn.Parameter(new_pos)
        # debug print
        print(f"Resized pos_embed: {int(math.sqrt(old_grid))}x{int(math.sqrt(old_grid))} -> {ph}x{pw} (tokens {old_grid}->{new_grid})")
    except Exception as e:
        print("Warning: could not resize pos_embed:", e)

import torch
import torch.nn.functional as F
import math

def adapt_pos_embed_from_checkpoint(state_dict, model, pos_name_candidates=("pos_embed","vit.pos_embed")):
    """
    Adapt checkpoint pos_embed tensor to match model.vit.pos_embed if sizes differ.
    Modifies state_dict in-place and returns (state_dict, adapted_bool).
    """
    ckpt_key = None
    for k in list(state_dict.keys()):
        for cand in pos_name_candidates:
            if k.endswith(cand):
                ckpt_key = k
                break
        if ckpt_key:
            break

    if ckpt_key is None:
        return state_dict, False

    ckpt_pos = state_dict[ckpt_key] 
    model_pos = None
    try:
        model_pos = model.vit.pos_embed.data
    except Exception:
        if hasattr(model, "pos_embed"):
            model_pos = model.pos_embed.data

    if model_pos is None:
        return state_dict, False

    ckpt_N = ckpt_pos.shape[1]
    model_N = model_pos.shape[1]
    if ckpt_N == model_N:
        return state_dict, False
    ckpt_cls, ckpt_grid = ckpt_pos[:, :1, :], ckpt_pos[:, 1:, :] 
    old_grid = ckpt_grid.shape[1]
    old_size = int(math.sqrt(old_grid))
    new_grid = model_pos.shape[1] - 1
    new_size = int(math.sqrt(new_grid))

    if old_size * old_size != old_grid or new_size * new_size != new_grid:
        print("Warning: non-square pos-embed grids; skipping adaptation.")
        return state_dict, False

    D = ckpt_grid.shape[2]
    ckpt_grid_reshaped = ckpt_grid.reshape(1, old_grid, D).permute(0,2,1).reshape(1, D, old_size, old_size)
    ckpt_grid_resized = F.interpolate(ckpt_grid_reshaped, size=(new_size, new_size), mode='bilinear', align_corners=False)
    ckpt_grid_resized = ckpt_grid_resized.reshape(1, D, new_size*new_size).permute(0,2,1)
    new_pos = torch.cat([ckpt_cls, ckpt_grid_resized], dim=1) 
    state_dict[ckpt_key] = new_pos
    print(f"Adapted checkpoint pos_embed: {old_size}x{old_size} -> {new_size}x{new_size} (tokens {old_grid}->{new_grid})")
    return state_dict, True


@st.cache_resource(show_spinner=False)
def load_model_from_checkpoint(ckpt_path, device, img_size=IMG_SIZE):
    """
    Build ViT-Tiny segmentation model and load checkpoint.
    This loader will attempt to adapt checkpoint pos_embed to the model if sizes mismatch.
    """
    class ViTSegLocal(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
            self.embed_dim = self.vit.embed_dim
            self.vit.patch_embed = nn.Conv2d(in_channels=6, out_channels=self.embed_dim, kernel_size=16, stride=16)
            try:
                old = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).patch_embed.proj
                old_w = old.weight
                new_w = torch.cat([old_w, old_w], dim=1)  # duplicate channels
                self.vit.patch_embed.weight = nn.Parameter(new_w)
                self.vit.patch_embed.bias = nn.Parameter(old.bias.clone())
            except Exception:
                pass

            self.proj = nn.Conv2d(self.embed_dim, 128, kernel_size=1)
            self.decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            )
            self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        def extract_tokens(self, x):
            x = self.vit.patch_embed(x)
            if x.dim() == 4:
                B, C, Hf, Wf = x.shape
                x = x.flatten(2).transpose(1,2)
            B, N, D = x.shape
            cls = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.vit.pos_embed[:, :x.shape[1], :]
            x = self.vit.pos_drop(x)
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)
            return x[:,1:]

        def forward(self, x):
            B, C, H, W = x.shape
            tokens = self.extract_tokens(x)
            N = tokens.shape[1]
            S = int(math.sqrt(N))
            feat = tokens.transpose(1,2).reshape(B, self.embed_dim, S, S)
            feat = self.proj(feat)
            feat = self.decoder(feat)
            logits = self.head(feat)
            logits = nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            return logits

    model = ViTSegLocal(num_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        if isinstance(state, dict) and ('model_state' in state or 'state_dict' in state):
            sd = state.get('model_state', state.get('state_dict', state))
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        elif isinstance(state, dict):
            sd = {k.replace('module.', ''): v for k, v in state.items()}
        else:
            sd = state

        sd, adapted = adapt_pos_embed_from_checkpoint(sd, model)
        if adapted:
            print("Checkpoint pos_embed adapted to model grid.")

        # try strict load first, fall back to non-strict if it fails
        try:
            model.load_state_dict(sd)
            print("Loaded checkpoint (strict).")
        except RuntimeError as e:
            print("Strict load failed:", e)
            model.load_state_dict(sd, strict=False)
            print("Loaded checkpoint with strict=False (some keys ignored).")

    except Exception as e:
        print("Failed loading checkpoint directly:", e)
        # final fallback: attempt non-strict load of raw state (best-effort)
        try:
            model.load_state_dict(state, strict=False)
            print("Fallback: loaded raw checkpoint with strict=False.")
        except Exception as e2:
            print("Final fallback failed:", e2)
            raise

    # finally, ensure model pos_embed also resized to requested img_size (safety)
    try:
        resize_vit_pos_embed_for_input(model, img_size)
    except Exception:
        pass

    model.eval()
    return model


st.sidebar.header("Model")
ckpt_input = st.sidebar.text_input("Checkpoint path (absolute)", value=DEFAULT_CHECKPOINT)
if st.sidebar.button("Load model"):
    if not ckpt_input or not os.path.isfile(ckpt_input):
        st.sidebar.error("Checkpoint not found. Paste absolute .pth path.")
    else:
        try:
            with st.spinner("Loading model..."):
                model = load_model_from_checkpoint(ckpt_input, DEVICE, img_size=IMG_SIZE)
            st.session_state['model'] = model
            st.session_state['ckpt_path'] = ckpt_input
            st.sidebar.success("Model loaded")
        except Exception as e:
            st.sidebar.error("Load failed: " + str(e))

if 'model' not in st.session_state:
    st.info("Load a model .pth in the sidebar to run inference.")

st.markdown("### Upload images")
col_up1, col_up2, col_up3 = st.columns([1,1,1])
with col_up1:
    pre_file = st.file_uploader("Pre-disaster image", type=['png','jpg','jpeg','tif','tiff'])
with col_up2:
    post_file = st.file_uploader("Post-disaster image", type=['png','jpg','jpeg','tif','tiff'])
with col_up3:
    gt_file = st.file_uploader("Optional: GT class mask (png)", type=['png','jpg','jpeg','png'])

use_post_as_pre = st.checkbox("Use post image as pre (if only one available)", value=False)
run_btn = st.button("Run prediction", disabled=('model' not in st.session_state))

if run_btn:
    if 'model' not in st.session_state:
        st.error("Model not loaded.")
    elif (pre_file is None and not use_post_as_pre) or post_file is None:
        st.error("Please upload both pre & post images (or use fallback).")
    else:
        try:
            if pre_file is None and use_post_as_pre:
                st.warning("Using post as pre (fallback).")
                post_pil = Image.open(post_file)
                pre_pil = post_pil.copy()
            else:
                pre_pil = Image.open(pre_file)
                post_pil = Image.open(post_file)

            pre_arr = preprocess_pil(pre_pil, target_size=IMG_SIZE)
            post_arr = preprocess_pil(post_pil, target_size=IMG_SIZE)

            inp = tensor_from_pair(pre_arr, post_arr, device=DEVICE)
            model = st.session_state['model']

            with st.spinner("Running model..."):
                with torch.no_grad():
                    logits = model(inp)      # (1,NUM_CLASSES,H,W)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_mask = np.argmax(probs, axis=0).astype(np.uint8)  # HxW

            post_uint8 = (post_arr * 255).astype(np.uint8)
            overlay = overlay_prediction_on_post(post_uint8, pred_mask)
            counts, comps = per_building_counts(pred_mask)

            annotated = draw_text_on_image(overlay.copy(), counts)

            # layout images
            st.markdown("### Results")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.markdown("**Post (resized)**")
                st.image(post_uint8, width=300)
            with c2:
                st.markdown("**Predicted overlay**")
                st.image(np.array(annotated), width=300)
            with c3:
                st.markdown("**Class mask (colors)**")
                # generate a readable categorical visualization
                cmap = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                palette = {0:(220,220,220), 1:(255,165,0), 2:(255,69,0), 3:(200,0,0)}
                for c, col in palette.items():
                    cmap[pred_mask==c] = col
                st.image(cmap, width=300)

            # show counts text
            st.markdown("#### Per-building predicted counts (connected components)")
            for c in range(NUM_CLASSES):
                st.write(f"- **{CLASS_NAMES[c]}**: {counts.get(c,0)}")

            # download predicted mask (raw)
            buf = io.BytesIO()
            Image.fromarray(pred_mask.astype(np.uint8)).save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download predicted class mask (PNG)", data=buf, file_name="pred_mask.png", mime="image/png")

            # if GT uploaded, show quick per-class IoU for that tile
            if gt_file is not None:
                try:
                    gt_pil = Image.open(gt_file).convert("L")
                    gt_resized = gt_pil.resize(IMG_SIZE, resample=Image.NEAREST)
                    gt_np = np.array(gt_resized).astype(np.uint8)
                    per_class_iou = []
                    eps = 1e-7
                    for c in range(NUM_CLASSES):
                        pred_c = (pred_mask == c)
                        true_c = (gt_np == c)
                        inter = np.logical_and(pred_c, true_c).sum()
                        union = np.logical_or(pred_c, true_c).sum()
                        iou = (inter + eps) / (union + eps)
                        per_class_iou.append(float(iou))
                    miou = float(np.mean(per_class_iou))
                    st.markdown("#### Quick tile metrics (GT provided)")
                    st.write(f"- mIoU: **{miou:.4f}**")
                    for c in range(NUM_CLASSES):
                        st.write(f"  - {CLASS_NAMES[c]} IoU: {per_class_iou[c]:.4f}")
                except Exception as e:
                    st.warning("Could not process GT file: " + str(e))

        except Exception as e:
            st.error("Inference error: " + str(e))

st.markdown("---")
st.markdown(
    "- The app resizes uploaded images to the model's training size and stacks pre/post into 6 channels.\n"
)
