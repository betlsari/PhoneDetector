# PhoneDetector

A computer vision project for detecting mobile phones in images using Python, OpenCV, and YOLO.

## Features

* Mobile phone detection in images
* YOLO-based object detection
* Annotated image output
* CPU inference support

## Technologies

* Python
* Ultralytics YOLO
* OpenCV
* NumPy

## Requirements

* Python 3.11 or newer

## Installation

Clone the repository and navigate to the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run detection on an image:

```bash
python main.py --image "path/to/image.jpg" --output "result.jpg"
```

The trained model weights and datasets are not included in this repository. The model must be placed at the path configured in `main.py`.

## Notes

The default mode uses the computer's camera. Use the `--image` argument to run detection on a saved image instead.

## License

No license has been specified yet.
