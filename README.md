# Digital Gauge Orientation Normalizer & Reading Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laneandfeed/digi-gauge-orientation-flipper/blob/main/demo.ipynb)

An intelligent computer vision pipeline designed to detect digital tire tread depth gauges, estimate device orientation, and automatically correct/rotate the LCD screen to a standardized landscape orientation for downstream OCR and automated measurement reading.


---

## Overview

Industrial measurement photos taken in the field often have arbitrary rotations, varied lighting conditions, reflections, and inverted display colors (dark font on light background vs. light font on dark OLED/LCD).

This pipeline provides an end-to-end automated solution:
1. **Object Detection**: Identifies gauge outer casing (`out_digital`) and inner digital LCD screen (`in_digital`) using a custom trained deep learning model (`best.pt`).
2. **Axis & Angle Estimation**: Employs GrabCut segmentation and PCA contour analysis to determine the physical angle of the gauge.
3. **180° Flip Disambiguation**: Uses a dual-box spatial check and screen ink-density distribution to distinguish upright displays from upside-down displays.
4. **Tight Crop & Landscape Normalization**: Delivers an upright, tightly cropped image of the LCD display ready for OCR.

---

## Repository Structure

```
├── best.pt                       # Custom trained gauge detection model weights (via Git LFS)
├── quick_test_v6.py              # Main test pipeline with adaptive edge-case handling
├── gauge_identification_two.py   # Multi-class detection and claim-level processing
├── testingcode/                  # Versioned development scripts & orientation experiments
├── testing/                      # Test sample images and failure test cases
├── models/                       # Model backbone definitions
├── utils/                        # Preprocessing, letterboxing, and NMS utilities
└── requirements.txt              # Environment dependencies
```

---

## Quick Start

### 1. Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- OpenCV
- Git LFS (required to clone `best.pt`)

```bash
git lfs install
pip install -r requirements.txt
```

### 2. Run Inference & Orientation Correction

To test orientation correction on a single image or test folder:

```bash
python quick_test_v6.py
```

Outputs will be saved with `_FINAL.jpg` showing the corrected, landscape-oriented screen ready for OCR.
---
<img width="200" height="200" alt="tester" src="https://github.com/user-attachments/assets/df8f5eca-8ff8-41d5-b3de-78423501566a" />
<img width="157" height="226" alt="tester_raw_in_crop" src="https://github.com/user-attachments/assets/4d14b79d-fc2a-4d55-8c54-06b070af1967" />
<img width="420" height="210" alt="tester_brightness_check" src="https://github.com/user-attachments/assets/faa553ac-8181-4361-bb03-bdc7cfa547cd" />
<img width="187" height="85" alt="tester_FINAL" src="https://github.com/user-attachments/assets/94fc0132-1f49-4ca5-a7ab-6ce95af74101" />

---

## Authors & Maintainers

- **Maintained by**: [laneandfeed](https://github.com/laneandfeed)
- **Model & Pipeline**: Custom industrial tire depth gauge detection and orientation normalization system.
