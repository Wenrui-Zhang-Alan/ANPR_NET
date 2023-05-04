# ANPR_NET
By leveraging the latest features in deep learning, this project involves approaches to recognize standard UK(or Europe)license plates from still image & video input of any given size.
## Getting Started
 In this implementation, a complete
pipeline has been built to extract hierarchical different levels of image, each
image(frame) input will be first passed into a pre-trained Mask R-CNN to
locate the cars, then segment individual sub-images of the cars and pass the
one selected car into another network namely ANPR_NET, which is trained
from scratch. ANPR_NET finds plate locations within the sub-image and
passes to the final part of the pipeline to perform OCR to extract plate numbers subsequently
### Prerequisites
Before using the project, you need to have the following dependencies installed:

- Python 3.8
- External libraries:
  - PyTorch
  - OpenCV
  - torchvision
  - NumPy
  - os
  - imutils
  - pytesseract
  - easyocr
  - torchvision
  - random
  - matplotlib
  - re
  - warnings
- Local modules:
  - ANPR_NET
  - GPS

You can install them using pip:

```bash
pip install torch opencv-python torchvision numpy imutils pytesseract easyocr torchvision matplotlib
```
