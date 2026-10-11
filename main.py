
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "phone_finetune-2"
    / "weights"
    / "best.pt"
)

IMAGE_SIZE = 416
WINDOW_NAME = "Telefon Tespiti - Q ile cikis"
PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telefon tespit demosu"
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Kullanilacak Windows kamera indeksi (varsayilan: 0)",
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Kamera yerine bu fotograf dosyasinda tahmin yap",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Isaretlenmis fotografi kaydet (yalnizca --image ile)",
    )

    return parser.parse_args()


def predict(model: YOLO, frame):
    return model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=0.30,
        iou=0.45,
        max_det=10,
        classes=[1],
        device="cpu",
        verbose=False,
    )[0]


def resize_for_preview(frame):
    height, width = frame.shape[:2]

    scale = min(
        PREVIEW_WIDTH / width,
        PREVIEW_HEIGHT / height,
        1.0,
    )

    if scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def run_image(
    model: YOLO,
    image_path: Path,
    output_path: Path | None,
) -> int:
    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Fotograf okunamadi: {image_path}")
        return 1

    result = predict(model, frame)
    annotated_frame = result.plot()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), annotated_frame):
            print(f"Isaretlenmis fotograf kaydedilemedi: {output_path}")
            return 1

        print(f"Isaretlenmis fotograf kaydedildi: {output_path}")

    print(f"Telefon tespit sayisi: {len(result.boxes)}")

    try:
        preview = resize_for_preview(annotated_frame)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()

    return 0


def run_camera(model: YOLO, camera_index: int) -> int:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        print(
            f"Kamera acilamadi (indeks {camera_index}). "
            "Kamera yerine fotograf test etmek icin --image kullanin."
        )
        return 1

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Kameradan goruntu alinamadi; "
                    "uygulama guvenli sekilde kapatiliyor."
                )
                return 1

            result = predict(model, frame)
            annotated_frame = result.plot()
            preview = resize_for_preview(annotated_frame)

            cv2.imshow(WINDOW_NAME, preview)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()

    if args.output is not None and args.image is None:
        print("--output secenegi yalnizca --image ile kullanilabilir.")
        return 2

    if not MODEL_PATH.is_file():
        print(f"Model agirligi bulunamadi: {MODEL_PATH}")
        return 1

    model = YOLO(str(MODEL_PATH))

    if args.image is not None:
        return run_image(model, args.image, args.output)

    return run_camera(model, args.camera_index)


if __name__ == "__main__":
    raise SystemExit(main())