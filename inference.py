"""
Inference pipeline for Visual Lego Counter.
Runs trained YOLOv8 model on project images, outputs counts and annotated results.
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json


PROJECT_IMAGES = Path('project_image_preprocess')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class names (must match training order from dataset.yaml)
CLASS_NAMES = [
    "Brick 2x4",
    "Plate 2x3",
    "Wheel Rim 20x30 with 6 Dual Spokes and External Ribs",
    "Technic Brick 1x10 with holes",
    "Panel 1x2x3 with Hollow Studs",
    "Tile 2x4 With Groove",
    "Brick 2x2 round with pin hole",
    "Plate 2x2 with 2 Wheel Pins",
    "Cone 3x3x2",
    "Plate 2x2 with 2 Studs on the side",
    "Slope brick 6x2 curved",
    "Brick 2x3 with Curved Top",
    "Tyre 18-56x17 Off-Road with Offset Centre",
    "Tyre 26-49x30 Tractor",
    "Window 1x2x3 Pane with Thick Corner Tabs",
    "Technic Sprocket Wheel 25.4",
    "Wheel Rim 14x18 with Holes on Both Sides",
    "Brick 1x4 with Studs on Side",
]

# Color map for drawing (BGR)
COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 128), (0, 255, 0),
    (128, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0),
    (255, 0, 128), (255, 0, 255), (128, 0, 255), (0, 0, 128),
    (0, 128, 128), (128, 128, 0), (128, 0, 128), (0, 128, 0),
    (128, 0, 0), (0, 0, 0),
]


def load_model(weights_path):
    """Load trained YOLOv8 model."""
    model = YOLO(str(weights_path))
    return model


def run_inference(model, image_path, conf=0.25, iou=0.5):
    """Run inference on a single image."""
    img = cv2.imread(str(image_path))
    results = model(img, conf=conf, iou=iou, verbose=False)[0]
    return img, results


def draw_results(img, results, conf=0.25):
    """Draw bounding boxes and labels on image."""
    annotated = img.copy()
    detections = []

    if results.boxes is None:
        return annotated, detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf_val, cls_id in zip(boxes, confs, clss):
        if conf_val < conf:
            continue
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[cls_id]
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{cls_name} ({conf_val:.2f})'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        detections.append({
            'class_id': int(cls_id),
            'class_name': cls_name,
            'confidence': float(conf_val),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
        })

    return annotated, detections


def count_by_class(detections):
    """Aggregate detection count per class."""
    counts = {}
    for d in detections:
        name = d['class_name']
        counts[name] = counts.get(name, 0) + 1
    return counts


def process_all_images(model_path, conf=0.25):
    """Run inference on all 10 project images and generate reports."""
    model = load_model(model_path)
    image_files = sorted(PROJECT_IMAGES.glob('*.jpg'))

    all_results = {}

    for img_path in image_files:
        img_name = img_path.stem
        print(f'Processing {img_name}...')

        img, results = run_inference(model, img_path, conf=conf)
        annotated, detections = draw_results(img, results, conf=conf)
        counts = count_by_class(detections)

        # Save annotated image
        out_img_path = OUTPUT_DIR / f'{img_name}_detected.jpg'
        cv2.imwrite(str(out_img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        all_results[img_name] = {
            'detections': detections,
            'counts': counts,
            'total_pieces': len(detections),
        }

        print(f'  Found {len(detections)} pieces:')
        for name, count in sorted(counts.items()):
            print(f'    {name}: {count}')

    # Save JSON report
    report_path = OUTPUT_DIR / 'detection_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Generate summary table
    generate_summary_table(all_results)

    return all_results


def generate_summary_table(all_results):
    """Generate a markdown summary table of counts per image."""
    table_path = OUTPUT_DIR / 'count_summary.md'
    lines = ['# Lego Piece Count Summary', '']
    lines.append('| Image | ' + ' | '.join(CLASS_NAMES) + ' | Total |')
    lines.append('|-------|' + '|'.join(['-------'] * (len(CLASS_NAMES) + 1)) + '|')

    for img_name in sorted(all_results.keys(), key=lambda x: int(x)):
        counts = all_results[img_name]['counts']
        row = [img_name]
        for cls_name in CLASS_NAMES:
            row.append(str(counts.get(cls_name, 0)))
        row.append(str(all_results[img_name]['total_pieces']))
        lines.append('| ' + ' | '.join(row) + ' |')

    # Add difficulty-level sections
    lines.append('')
    lines.append('## By Difficulty Level')
    levels = {'Simple (1-4)': [1, 2, 3, 4], 'Medium (5-7)': [5, 6, 7], 'Hard (8-10)': [8, 9, 10]}
    for level_name, indices in levels.items():
        lines.append(f'\n### {level_name}')
        for i in indices:
            img_name = str(i)
            if img_name in all_results:
                counts = all_results[img_name]['counts']
                pieces_str = ', '.join(f'{n}: {c}' for n, c in sorted(counts.items()) if c > 0)
                lines.append(f'- **{img_name}.jpg**: {all_results[img_name]["total_pieces"]} pieces — {pieces_str}')

    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nSummary table: {table_path}')


if __name__ == '__main__':
    import sys

    # Find the best trained model (synthetic data trained)
    weights_dir = Path('runs/detect/training_data/yolo_synthetic/weights')
    best_pt = weights_dir / 'best.pt'
    last_pt = weights_dir / 'last.pt'
    model_path = best_pt if best_pt.exists() else last_pt

    if not model_path.exists():
        print(f'ERROR: No trained model found at {weights_dir}')
        print('Falling back to pretrained yolov8n.pt')
        model_path = Path('yolov8n.pt')

    print(f'Using model: {model_path}')
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    process_all_images(model_path, conf=conf)
