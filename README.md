# Real-Time Facial Emotion Recognition

A small webcam app that detects a face, classifies its emotion with an ONNX model, and shows live probability bars next to the video feed. Built with OpenCV's DNN module only (no PyTorch/TensorFlow needed at runtime).

## Features

- Real-time face detection with **YuNet** (`cv2.FaceDetectorYN`)
- Face alignment and cropping with **SFace** (`cv2.FaceRecognizerSF.alignCrop`)
- Emotion classification with an **ONNX** model (`cv2.dnn.readNetFromONNX`)
- 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Temporal smoothing: the last 5 predictions are averaged to reduce flicker
- Side panel with a color-coded probability bar per emotion
- Mirrored webcam view, face bounding box and light denoising on the face region

## How It Works

1. Capture a frame from the webcam and flip it horizontally.
2. Detect the face with YuNet (first detected face is used).
3. Align and crop the face with SFace, resize to 48x48, convert to grayscale.
4. Build a blob (`scale = 1/95`) and run the ONNX emotion model.
5. Apply softmax, push into a 5-frame buffer, and average.
6. Draw the bar chart (300 px wide) and stack it beside the frame with `np.hstack`.
7. Press **Esc** to quit.

## Requirements

- Python 3.9+ (tested on 3.12)
- `opencv-python` (4.8+ recommended, for `FaceDetectorYN` / `FaceRecognizerSF`)
- `numpy`
- `matplotlib` (imported, currently unused)

```bash
pip install opencv-python numpy matplotlib
```

## Models

Download and place these files, then update the paths in the notebook:

| Model | Purpose | Source |
|-------|---------|--------|
| `face_detection_yunet_2023mar.onnx` | Face detection | OpenCV Zoo |
| `face_recognition_sface_2021dec.onnx` | Face alignment | OpenCV Zoo |
| `onnx_model.onnx` | Emotion classifier (48x48 grayscale, 7 classes) | Facial-Emotion-Recognition-PyTorch-ONNX |

Paths currently hard-coded in the notebook:

```python
model_path = r'C:\Users\fatima\Downloads\Facial-Emotion-Recognition-PyTorch-ONNX-master\ONNX\models\onnx_model.onnx'
detector   = cv2.FaceDetectorYN_create(r'I:\v12\face_detection_yunet_2023mar.onnx', "", (320, 320), 0.3, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF_create(r'I:\v12\face_recognition_sface_2021dec.onnx', "")
```

## Usage

1. Open the notebook in Jupyter.
2. Fix the three model paths above.
3. Run all cells. A window named `h` opens with the webcam feed and the emotion chart.
4. Press **Esc** to stop.

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Detection score threshold | `FaceDetectorYN_create(..., 0.3, ...)` | 0.3 |
| NMS threshold | `FaceDetectorYN_create(..., 0.3, ...)` | 0.3 |
| Smoothing window | `deque(maxlen=5)` | 5 frames |
| Bar colors | `emotion_colors` | BGR per emotion |
| Camera index | `cv2.VideoCapture(0)` | 0 |

## Project Structure

```
.
├── Untitled178.ipynb   # Main notebook (rename as you like)
└── README.md
```

## Credits

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for YuNet and SFace
- Facial-Emotion-Recognition-PyTorch-ONNX for the emotion model
