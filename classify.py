import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms

# —— 配置区域 —— #

# 输入目录：用户上传的图片和 YOLO JSON
USER_IMAGE_DIR = Path("user_data/image")
USER_JSON_DIR  = Path("user_data/json")

# 输出根目录：分类完后会在这里新建子文件夹
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型权重和类别名（保持与你训练时一致）
CHECKPOINT = "final_model.pth"
CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 模型加载函数 —— #
def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    model = models.resnet101(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, num_classes)
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

# —— 分类函数 —— #
def classify_image_detections(image, detections, model, preprocess, class_names, device):
    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = image.crop((x1, y1, x2, y2))
        inp  = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)[0]
            pid    = int(probs.argmax().item())
            pconf  = float(probs[pid].item())
        results.append({
            "bbox":      [x1, y1, x2, y2],
            "pred_id":    pid,
            "pred_label": class_names[pid],
            "confidence": pconf
        })
    return results

# —— 可视化函数 —— #
def annotate_image(image, results, outline="red", font=None):
    draw = ImageDraw.Draw(image)
    if font is None:
        font = ImageFont.load_default()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        label = f"{r['pred_label']} {r['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
        text_pos = (x1, max(0, y1 - 12))
        draw.text(text_pos, label, fill=outline, font=font)
    return image

# —— 主程序 —— #
if __name__ == "__main__":
    # 加载模型与预处理
    model = load_model(CHECKPOINT, num_classes=len(CLASS_NAMES), device=DEVICE)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    font = ImageFont.load_default()

    # 遍历用户图片目录，只处理尚未移动的图片
    for img_path in sorted(USER_IMAGE_DIR.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # 寻找对应 YOLO JSON
        json_path = USER_JSON_DIR / (img_path.stem + ".json")
        if not json_path.is_file():
            print(f"[跳过] 未找到 JSON for {img_path.name}")
            continue

        # 创建子目录和 original/ annotated/
        subdir = OUTPUT_DIR / img_path.stem
        orig_dir = subdir / "original"
        ann_dir  = subdir / "annotated"
        orig_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        # 打开原图和原始 JSON
        img  = Image.open(img_path).convert("RGB")
        dets = json.load(open(json_path, "r"))

        # 分类检测框
        results = classify_image_detections(
            image=img,
            detections=dets,
            model=model,
            preprocess=preprocess,
            class_names=CLASS_NAMES,
            device=DEVICE
        )

        # 保存带标注的图片到 annotated/
        annotated = annotate_image(img.copy(), results, outline="red", font=font)
        annotated.save(ann_dir / img_path.name)

        # 保存新的 JSON 到 annotated/
        out_json = ann_dir / (img_path.stem + ".json")
        json.dump(results, open(out_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        # 移动原始图片和 JSON 到 original/
        shutil.move(str(img_path), str(orig_dir / img_path.name))
        shutil.move(str(json_path), str(orig_dir / json_path.name))

        print(f"[已处理] {img_path.name}")

    print("分类任务完成。")
