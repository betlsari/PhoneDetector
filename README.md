# PhoneDetector

A computer vision project that detects mobile phones in images using Python, OpenCV, and a fine-tuned YOLO model.

## Features

* Mobile phone detection in images
* YOLO-based object detection
* Bounding boxes and confidence scores
* Option to save detection results
* CPU inference support

## Technologies

* Python
* Ultralytics YOLO
* OpenCV
* NumPy

## Requirements

* Python 3.11
* A trained YOLO model (`best.pt`)
* Dependencies listed in `requirements.txt`

## Installation

Clone the repository:

```bash
git clone https://github.com/betlsari/PhoneDetector.git
cd PhoneDetector
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights

The trained model weights are not included in this repository.

To run the project, obtain the trained `best.pt` file and place it at the model path configured in `main.py`.

The model weights can be shared separately in a future release.

## Usage

Run the detector on an image:

```bash
python main.py --image "path/to/image.jpg" --output "result.jpg"
```

Replace the example image path with the path to your own image.

## Limitations

* A compatible trained model is required.
* Webcam detection depends on camera availability.
* Detection accuracy depends on the training dataset and model quality.

## License

No license has been specified yet.
