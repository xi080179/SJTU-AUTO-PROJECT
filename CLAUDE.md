# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual Lego Counter — a course project for "Fundamentals of Digital Image Processing" (AU3304, Fall 2023-2024). The goal is to detect, classify, and count Lego pieces from top-view photos of Lego brick piles using computer vision.

## Tech Stack

- Python 3, virtual environment at `.venv`
- **YOLOv8** (Ultralytics 8.4.53) for object detection and instance segmentation
- **OpenCV** (opencv-python 4.13) for image processing
- **PyTorch** 2.12.0 (CPU only — no CUDA)
- matplotlib, numpy, scipy for visualization and computation

## Commands

```bash
# Activate the virtual environment
.venv/Scripts/activate

# Run a Python script
.venv/Scripts/python your_script.py

# Install packages (if needed)
.venv/Scripts/pip install <package>

# YOLOv8 training (from within the project)
.venv/Scripts/yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# YOLOv8 validation
.venv/Scripts/yolo detect val data=dataset.yaml model=path/to/best.pt
```

## Data Pipeline

```
project_image/          (raw 1.jpg–10.jpg)
    ↓ preprocess.py     (shadow removal, CLAHE, edge enhancement)
project_image_preprocess/  (preprocessed input images)
    ↓ inference.py      (YOLOv8 detection + drawing)
results/                (annotated images, JSON report, markdown summary)
```

## Key Files

| File | Purpose |
|---|---|
| `preprocess.py` | Shadow removal + edge enhancement preprocessing pipeline. Reads from `project_image/`, writes to `project_image_preprocess/` |
| `inference.py` | End-to-end inference: loads trained YOLO model, runs detection on preprocessed images, outputs annotated images + `detection_report.json` + `count_summary.md` to `results/` |
| `extract_piece_images.py` | Extracts 18 individual piece images from the green-background cells in `piece_catalog_table.jpg`, saves RGBA PNGs + `labels.txt` to `training_data/piece_images/` |
| `__extract_pdf_text.py` | One-off script to extract and print text from `course_project.pdf` |
| `yolov8n.pt` | YOLOv8 nano detection model weights |
| `yolov8n-seg.pt` | YOLOv8 nano instance segmentation model weights |
| `course_project.pdf` | Assignment description — problem statement, piece catalog, submission requirements |

## Key Directories

| Directory | Contents |
|---|---|
| `project_image/` | 10 raw test images (3 difficulty levels: 1-4 simple, 5-7 medium, 8-10 hard with occlusions) |
| `project_image_preprocess/` | Preprocessed versions of test images (output of `preprocess.py`) |
| `training_data/piece_images/` | 18 extracted catalog piece images + `labels.txt` (output of `extract_piece_images.py`) |
| `training_data/multiview_templates/` | Multi-angle template images for alternative template-matching approaches |
| `runs/detect/` | YOLOv8 training run outputs (weights, metrics, etc.) |
| `results/` | Inference output (created by `inference.py`) |

## Lego Piece Types

18 distinct piece types in the catalog. Pieces of the same shape but different colors are counted together. Pieces of different sizes are counted separately. "Bricks" and "Plates" look similar from top view and may be confused. Each piece follows the naming convention: "Piece type + size + optional modifiers" (e.g., "Brick 2x4", "Window 1x2x3 Pane with Thick Corner Tabs").

Reference resources for 3D models and photos:
- [LDraw.org](https://ldraw.org/) — open-source Lego part models, searchable by official ID/name
- [Bricklink](https://bricklink.com/) — 3D models + real photos of Lego pieces

## Architecture Guidance

The standard approach for this project involves:

1. **Preprocessing**: Remove shadows (multi-scale Gaussian blur subtraction in LAB colorspace), enhance edges (unsharp masking + Laplacian), improve contrast (CLAHE). See `preprocess.py`.
2. **Detection**: Use YOLOv8 to detect individual Lego pieces in the image. The segmentation model (`yolov8n-seg.pt`) can help separate touching pieces.
3. **Classification**: Match each detected piece to one of the 18 catalog types. Top-view images mean the primary distinguishing features are 2D shape outline, stud pattern, and relative dimensions.
4. **Counting**: Aggregate counts per piece type and mark piece locations on the output image.

Training a custom YOLOv8 model requires creating a labeled dataset with annotations for each of the 18 classes. The `project_image/` → `project_image_preprocess/` → model inference pipeline is designed to work with a model trained on the preprocessed image style.
