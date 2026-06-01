from ultralytics import YOLO
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
PANEL_WIDTH = 320
MAX_MISSED_FRAMES = 90


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 license plate detection on a video file."
    )
    parser.add_argument(
        "--model",
        default=r"C:\License-Plate-Recognition\best\best.pt",
        help="Path to YOLOv8 weights file.",
    )
    parser.add_argument(
        "--video",
        default=r"C:\License-Plate-Recognition\videos\input.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\License-Plate-Recognition\outputs\input_annotated.mp4",
        help="Path for annotated output video.",
    )
    parser.add_argument(
        "--csv",
        default=r"C:\License-Plate-Recognition\outputs\detections.csv",
        help="Path for CSV detection log.",
    )
    parser.add_argument(
        "--crops-dir",
        default=r"C:\License-Plate-Recognition\outputs\crops",
        help="Directory for the best cropped license plate image per tracked plate.",
    )
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--track-iou",
        type=float,
        default=0.15,
        help="IoU threshold used to keep the same plate ID across frames.",
    )
    parser.add_argument(
        "--track-distance",
        type=float,
        default=4.0,
        help="Center-distance multiplier used to keep a moving plate as the same ID.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame. Use 2 for every alternate frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process the full video.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Disable saving cropped license plate detections.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live preview window while processing.",
    )
    return parser.parse_args()


def validate_paths(model_path, video_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union else 0.0


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def center_distance(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def assign_plate_id(bbox, tracks, frame_idx, track_iou, track_distance, used_track_ids):
    best_track_id = None
    best_score = -1.0
    bbox_w, bbox_h = bbox_size(bbox)

    for track_id, track in tracks.items():
        if track_id in used_track_ids:
            continue

        if frame_idx - track["last_seen"] > MAX_MISSED_FRAMES:
            continue

        iou = bbox_iou(bbox, track["bbox"])
        distance = center_distance(bbox, track["bbox"])
        track_w, track_h = bbox_size(track["bbox"])
        distance_limit = max(bbox_w, bbox_h, track_w, track_h) * track_distance

        if iou >= track_iou:
            score = iou + 1.0
        elif distance <= distance_limit:
            score = 1.0 - (distance / distance_limit)
        else:
            continue

        if score > best_score:
            best_score = score
            best_track_id = track_id

    if best_track_id is not None:
        tracks[best_track_id]["bbox"] = bbox
        tracks[best_track_id]["last_seen"] = frame_idx
        used_track_ids.add(best_track_id)
        return best_track_id

    new_track_id = len(tracks) + 1
    tracks[new_track_id] = {
        "bbox": bbox,
        "last_seen": frame_idx,
        "best_conf": 0.0,
        "best_crop": None,
    }
    used_track_ids.add(new_track_id)
    return new_track_id


def clear_previous_plate_crops(crops_dir):
    for crop_path in crops_dir.glob("plate_*.jpg"):
        if crop_path.is_file():
            crop_path.unlink()


def draw_plate_list_panel(frame, detections, total_tracks):
    panel = frame.copy()
    panel_height = frame.shape[0]
    list_panel = np.full((panel_height, PANEL_WIDTH, 3), (80, 80, 80), dtype=np.uint8)

    cv2.putText(
        list_panel,
        "Number Plate",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        list_panel,
        f"Current: {len(detections)}",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        list_panel,
        f"Saved plates: {total_tracks}",
        (24, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    y = 142
    if not detections:
        cv2.putText(
            list_panel,
            "No plate detected",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        for det in detections[:12]:
            label = f"Plate {det['plate_id']:03d}  {det['confidence']:.2f}"
            cv2.rectangle(list_panel, (18, y - 24), (PANEL_WIDTH - 18, y + 10), (105, 105, 105), -1)
            cv2.putText(
                list_panel,
                label,
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 42
            if y > panel_height - 22:
                break

    return cv2.hconcat([panel, list_panel])


def draw_detections(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"Plate {det['plate_id']:03d} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_w, text_h = text_size
        label_y1 = max(0, y1 - text_h - 8)
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (min(x1 + text_w + 8, annotated.shape[1] - 1), label_y1 + text_h + 8),
            (0, 160, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y1 + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main():
    args = parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    crops_dir = Path(args.crops_dir)

    validate_paths(model_path, video_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
        clear_previous_plate_crops(crops_dir)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    print(f"Using device: {device}")

    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video metadata. Check codec compatibility.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width + PANEL_WIDTH, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    total_detections = 0
    confidence_sum = 0.0
    frames_with_plates = 0
    frame_idx = 0
    stop_requested = False
    window_name = "License Plate Detection"
    tracks = {}

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "detection_number",
                "plate_id",
                "frame_number",
                "frame_detection_count",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "class",
            ]
        )

        if not args.no_show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while cap.isOpened() and not stop_requested:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"Stopped after --max-frames {args.max_frames}")
                break

            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame
            frame_detection_list = []
            used_track_ids = set()

            if frame_idx % args.frame_skip == 0:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes
                frame_detections = len(boxes) if boxes is not None else 0

                print(f"Frame {frame_idx}: {frame_detections} detection(s)")

                if frame_detections > 0:
                    frames_with_plates += 1

                    sorted_boxes = sorted(
                        boxes,
                        key=lambda current_box: float(current_box.conf[0]),
                        reverse=True,
                    )

                    for box in sorted_boxes:
                        raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in box.xyxy[0].tolist()]
                        x1 = max(0, min(raw_x1, width - 1))
                        y1 = max(0, min(raw_y1, height - 1))
                        x2 = max(0, min(raw_x2, width - 1))
                        y2 = max(0, min(raw_y2, height - 1))
                        bbox = (x1, y1, x2, y2)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        plate_id = assign_plate_id(
                            bbox,
                            tracks,
                            frame_idx,
                            args.track_iou,
                            args.track_distance,
                            used_track_ids,
                        )

                        frame_detection_list.append(
                            {
                                "plate_id": plate_id,
                                "bbox": bbox,
                                "confidence": confidence,
                                "class_id": class_id,
                            }
                        )

                        detection_number = total_detections + 1
                        writer.writerow(
                            [
                                detection_number,
                                plate_id,
                                frame_idx,
                                frame_detections,
                                x1,
                                y1,
                                x2,
                                y2,
                                f"{confidence:.4f}",
                                class_id,
                            ]
                        )
                        total_detections += 1
                        confidence_sum += confidence
                        print(
                            "  "
                            f"#{detection_number}: "
                            f"plate_id={plate_id:03d}, "
                            f"bbox=({x1}, {y1}, {x2}, {y2}), "
                            f"conf={confidence:.4f}, class={class_id}"
                        )

                annotated = draw_detections(frame, frame_detection_list)

                if not args.no_crops:
                    for det in frame_detection_list:
                        plate_id = det["plate_id"]
                        confidence = det["confidence"]
                        if confidence <= tracks[plate_id]["best_conf"]:
                            continue

                        x1, y1, x2, y2 = det["bbox"]
                        plate_crop = frame[y1:y2, x1:x2]

                        if plate_crop.size:
                            tracks[plate_id]["best_conf"] = confidence
                            tracks[plate_id]["best_crop"] = plate_crop.copy()
                            print(
                                "  "
                                f"Updated best plate crop in memory: plate_{plate_id:03d}.jpg "
                                f"(conf={confidence:.4f})"
                            )

                output_frame = draw_plate_list_panel(
                    annotated, frame_detection_list, len(tracks)
                )
                display_frame = output_frame
                out.write(output_frame)
            else:
                output_frame = draw_plate_list_panel(frame, [], len(tracks))
                display_frame = output_frame
                out.write(output_frame)

            if not args.no_show:
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user. Pressed q in preview window.")
                    stop_requested = True

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    saved_crop_count = 0
    if not args.no_crops:
        for plate_id, track in tracks.items():
            if track["best_crop"] is None:
                continue

            crop_path = crops_dir / f"plate_{plate_id:03d}.jpg"
            cv2.imwrite(str(crop_path), track["best_crop"])
            saved_crop_count += 1

    average_confidence = confidence_sum / total_detections if total_detections else 0.0
    print("Done.")
    print(f"Output video: {output_path}")
    print(f"CSV log: {csv_path}")
    if not args.no_crops:
        print(f"Crops directory: {crops_dir}")
        print(f"Best plate JPG files saved: {saved_crop_count}")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames with plates: {frames_with_plates}")
    print(f"Total detections: {total_detections}")
    print(f"Unique plate IDs saved: {len(tracks)}")
    print(f"Average confidence: {average_confidence:.4f}")


if __name__ == "__main__":
    main()
