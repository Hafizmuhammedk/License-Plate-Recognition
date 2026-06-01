# License-Plate-Recognition

License-Plate-Recognition is a Python computer vision project for detecting vehicle license plates in video using a YOLO/Ultralytics model. The main script processes an input video frame by frame, draws license plate bounding boxes, assigns simple plate IDs across frames, writes an annotated output video, logs detections to CSV, and saves the best crop for each tracked plate.

## Detection Preview

![License plate detection preview](assets/detection.gif)

## Features

- Detects license plates in video files with a YOLO model.
- Supports common video formats: `mp4`, `avi`, `mov`, `mkv`, and `wmv`.
- Draws bounding boxes, confidence scores, and plate IDs on each detected plate.
- Adds a side panel showing current detections and saved plate count.
- Tracks plates across frames using IoU and center-distance matching.
- Exports an annotated video.
- Writes a CSV detection log with frame number, bounding box coordinates, confidence, and class ID.
- Saves the best cropped image for each tracked plate.
- Uses CUDA automatically when a compatible GPU is available, otherwise falls back to CPU.

## Project Structure

```text
License-Plate-Recognition/
|-- detect_video.py                  # Main video detection script
|-- req.txt                          # Basic project dependencies
|-- license-plate-recognition.ipynb  # Notebook experiments/training workflow
|-- test.ipynb                       # Test notebook
|-- best/                            # Local trained model weights
|-- videos/                          # Local input videos
|-- outputs/                         # Generated videos, CSV logs, and crops
|-- runs/                            # Ultralytics training/inference outputs
`-- License-Plate-Recognition-1/     # Local dataset export
```

Large local assets such as datasets, videos, model weights, and run outputs are ignored by Git through `.gitignore`.

## Requirements

- Python 3.9 or newer
- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy

The repository includes `req.txt` with:

```text
roboflow
ultralytics
```

If your environment does not already have OpenCV, PyTorch, or NumPy installed, install them as well.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/License-Plate-Recognition.git
cd License-Plate-Recognition
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r req.txt
pip install opencv-python torch numpy
```

## Model and Video Setup

Place your trained YOLO model weights in a local folder, for example:

```text
best/best.pt
```

Place your input video in:

```text
videos/input.mp4
```

The default paths in `detect_video.py` are currently configured for:

```text
C:\License-Plate-Recognition\best\best.pt
C:\License-Plate-Recognition\videos\input.mp4
```

You can either keep that structure locally or pass custom paths from the command line.

## Usage

Run detection with the default paths:

```bash
python detect_video.py
```

Run detection with custom model, input, and output paths:

```bash
python detect_video.py ^
  --model best/best.pt ^
  --video videos/input.mp4 ^
  --output outputs/input_annotated.mp4 ^
  --csv outputs/detections.csv ^
  --crops-dir outputs/crops
```

On macOS/Linux, use backslashes only for line continuation:

```bash
python detect_video.py \
  --model best/best.pt \
  --video videos/input.mp4 \
  --output outputs/input_annotated.mp4 \
  --csv outputs/detections.csv \
  --crops-dir outputs/crops
```

Run without the live OpenCV preview window:

```bash
python detect_video.py --no-show
```

Process every alternate frame for faster inference:

```bash
python detect_video.py --frame-skip 2
```

Test only the first 300 frames:

```bash
python detect_video.py --max-frames 300
```

Disable cropped plate saving:

```bash
python detect_video.py --no-crops
```

## Command Line Options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `C:\License-Plate-Recognition\best\best.pt` | Path to YOLO weights file. |
| `--video` | `C:\License-Plate-Recognition\videos\input.mp4` | Path to input video. |
| `--output` | `C:\License-Plate-Recognition\outputs\input_annotated.mp4` | Path for annotated output video. |
| `--csv` | `C:\License-Plate-Recognition\outputs\detections.csv` | Path for CSV detection log. |
| `--crops-dir` | `C:\License-Plate-Recognition\outputs\crops` | Directory for saved best plate crops. |
| `--conf` | `0.50` | Detection confidence threshold. |
| `--iou` | `0.45` | YOLO non-max suppression IoU threshold. |
| `--track-iou` | `0.15` | IoU threshold used to keep the same plate ID across frames. |
| `--track-distance` | `4.0` | Center-distance multiplier used for plate tracking. |
| `--imgsz` | `640` | Inference image size. |
| `--frame-skip` | `1` | Process every Nth frame. |
| `--max-frames` | `0` | Stop after this many frames. `0` means full video. |
| `--no-crops` | Off | Disable cropped plate image saving. |
| `--no-show` | Off | Disable the live preview window. |

## Outputs

After a successful run, the script creates:

```text
outputs/input_annotated.mp4
outputs/detections.csv
outputs/crops/plate_001.jpg
outputs/crops/plate_002.jpg
...
```

The CSV contains:

| Column | Description |
| --- | --- |
| `detection_number` | Sequential detection number. |
| `plate_id` | Tracked plate ID assigned by the script. |
| `frame_number` | Frame where the detection occurred. |
| `frame_detection_count` | Number of detections in that frame. |
| `x1`, `y1`, `x2`, `y2` | Bounding box coordinates. |
| `confidence` | Model confidence score. |
| `class` | YOLO class ID. |

## Notes

- Press `q` in the preview window to stop processing early.
- If the preview window does not open on a server or notebook environment, run with `--no-show`.
- If video writing fails, check that OpenCV can write MP4 files on your system.
- For better performance, use a CUDA-enabled PyTorch install and an NVIDIA GPU.

## License

Add your preferred license for this project before publishing the repository.
