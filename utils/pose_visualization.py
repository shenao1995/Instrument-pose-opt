"""Shared GT/prediction panels for training validation and inference."""
import numpy as np
from PIL import Image, ImageDraw
import torch

# Instrument 1: shaft red, wrist green, jaws blue. Instrument 2: shaft orange,
# wrist cyan, jaws magenta. Further instruments reuse the second palette.
_ARM_PALETTES = ([[1., .15, .15], [.15, 1., .15], [.15, .35, 1.]],
                 [[1., .6, .1], [.1, .95, .95], [1., .2, .95]])


def mask_palette(channels):
    """[C, 3] RGB colours for a semantic mask with 3 channels per instrument."""
    if channels % 3:
        raise ValueError("Semantic masks have three channels per instrument")
    colours = []
    for arm in range(channels // 3):
        colours.extend(_ARM_PALETTES[min(arm, len(_ARM_PALETTES) - 1)])
    return np.asarray(colours, dtype=np.float32)


def validation_images(batch,result):
    """B rows, GT on the left and prediction on the right; same target RGB base.

    Tips are drawn per instrument: GT yellow and prediction pink for the first
    instrument, GT light-yellow and prediction violet for the second.
    """
    rgb = batch["rgb"].detach().float().clamp(0,1)
    palette = rgb.new_tensor(mask_palette(batch["mask"].shape[1]))
    def overlap(mask):
        mask = mask.detach().float()
        alpha = mask.sum(1,keepdim=True).clamp(0,1)
        color = torch.einsum("bchw,cd->bdhw",mask,palette)
        return (rgb*(1-.4*alpha)+.4*color).clamp(0,1)
    def rows(left,right):
        paired = torch.cat((left,right),-1)
        b,c,h,w = paired.shape
        return paired.permute(1,0,2,3).reshape(c,b*h,w).cpu()
    left, right = overlap(batch["mask"]), overlap(result["mask"])
    if "tips" in batch and "tips" in result:
        gt = batch["tips"].detach().cpu().numpy().reshape(len(rgb),-1,2,2)
        pred = result["tips"].detach().cpu().numpy().reshape(len(rgb),-1,2,2)
        conf = batch["tip_confidence"].detach().cpu().numpy().reshape(len(rgb),-1,2)
        gt_colors = ((255,230,0),(255,255,160))
        pred_colors = ((255,60,220),(170,90,255))
        annotated = []
        for panels, prediction in ((left,False),(right,True)):
            images = []
            for i, panel in enumerate(panels):
                image = Image.fromarray((panel.cpu().numpy().transpose(1,2,0)*255).round().astype(np.uint8))
                draw = ImageDraw.Draw(image)
                def tips(points, good, color, prefix):
                    finite = np.isfinite(points).all(-1)
                    # PIL clips segments to the image; cap extreme off-screen projections.
                    safe = np.clip(points, -4*max(image.size), 4*max(image.size))
                    if (good & finite).all():
                        draw.line([tuple(p) for p in safe],fill=color,width=2)
                    for j, (x,y) in enumerate(safe):
                        if not finite[j] or not (0 <= x < image.width and 0 <= y < image.height):
                            continue
                        if good[j]:
                            draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
                            draw.text((x+4,y+2),f"{prefix}{j+1}",fill=color)
                        else:
                            draw.line((x-3,y-3,x+3,y+3),fill=(160,160,160),width=1)
                            draw.line((x-3,y+3,x+3,y-3),fill=(160,160,160),width=1)
                captions = []
                for arm in range(gt.shape[1]):
                    valid = (conf[i,arm] > 0) & np.isfinite(gt[i,arm]).all(-1)
                    label = "" if gt.shape[1] == 1 else f"{'LR'[arm] if arm < 2 else arm+1}"
                    tips(gt[i,arm],valid,gt_colors[min(arm,1)],"G"+label)
                    gap = np.linalg.norm(gt[i,arm,0]-gt[i,arm,1]) if valid.all() else None
                    if prediction:
                        tips(pred[i,arm],np.ones(2,bool),pred_colors[min(arm,1)],"P"+label)
                        error = abs(np.linalg.norm(pred[i,arm,0]-pred[i,arm,1])-gap) if gap is not None else None
                        captions.append(f"{label or 'gap'} err {error:.1f}px" if error is not None else f"{label or 'gap'} err N/A")
                    else:
                        captions.append(f"{label or 'valid'} {valid.sum()}/2 gap {gap:.1f}px" if gap is not None else f"{label or 'valid'} {valid.sum()}/2 gap N/A")
                caption = ("GT yellow / Pred pink; " if prediction else "GT yellow; ")+"; ".join(captions)
                draw.rectangle((0,0,image.width,15),fill=(0,0,0))
                draw.text((3,2),caption,fill=(255,255,255))
                images.append(torch.from_numpy(np.array(image)).permute(2,0,1).float()/255)
            annotated.append(torch.stack(images))
        left, right = annotated
    return {"overlap_GT_left_prediction_right":rows(left,right),
            "RGB_target_left_render_right":rows(rgb,result["rgb"].detach().float().clamp(0,1))}
