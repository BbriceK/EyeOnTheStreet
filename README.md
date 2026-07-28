# EyeOnTheStreet

Prior work shows that many Canadian cities lack accurate and complete records of traffic calming measures (TCMs). TCMs are physical modifications to roads that help reduce traffic collisions, yet incomplete information limits the evaluation of real-world road safety impacts and hinders equitable urban planning. Manually identifying them across all locations and years at the city scale is impractical. 

To address this, this work presents a **baseline multi-label classification model** that identifies four TCMs—**curb extensions, cycle tracks, median islands, and speed humps**—in default-angle Google Street View (GSV) images from Toronto and Montréal. Applying this to historical imagery **enables the construction of a geospatial database that tracks when and where TCMs were implemented**, supporting future safety analysis and policy evaluation.

For details, check the associated paper; this work was presented in the UbiComp4VRU Workshop at UbiComp ’25.

#### Modeling challenges in real-world scenarios
Identifying these four TCMs in **default-angle GSV images** involves several non-trivial challenges:

- High variability within the same category: TCMs are defined by function rather than appearance, so even within one category, visual forms can vary widely. For example, cycle tracks may be separated by bollards, concrete barriers, or raised pavement. Furthermore, variation is further increased across cities due to differences in local policies.

- Temporal environment factors at longitudinal scale: Images of the same location can differ substantially over time due to changes in weather, lighting conditions, seasons, and infrastructure updates.

- Occlusion and visual clutter: In default-angle GSV images, TCMs often appear small, partially occluded, or embedded within cluttered urban scenes.

## Overview
This repository provides:

- `data/` – training, validation, and test images, along with the corresponding label file.
- `src/` – source code for embedding generation, model training, and the inference pipeline, along with sample shell scripts.
- `weights/` – the pretrained DINOv2 weight and the best-performing classifier checkpoint used for inference.

## Installation
To set up the project, first create and activate a virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Clone this repository:
```bash
git clone https://github.com/BbriceK/EyeOnTheStreet.git
cd EyeOnTheStreet
```

This project relies on frozen DINOv2 weights as a backbone. Clone the DINOv2 repository in the src folder and install the dependencies:
```bash
cd src
git clone https://github.com/facebookresearch/dinov2.git

cd dinov2
pip install -r requirements.txt
```

Install the remaining dependencies for this project:
```bash
cd ../..
pip install -r requirements.txt
```

## Usage
This repository supports two workflows:

1. **Training pipeline** – generate image embeddings, train the classifier, and evaluate model performance.
2. **Inference** – generate predictions on new images using the provided classifier checkpoint without retraining.

### Training pipeline
#### Step 1: Generate image embeddings
A sample script (`src/embeddings.sh`) is provided to compute image embeddings. Before running the script, configure all the paths according to the comments. 

Run the script with:
```bash
bash src/embeddings.sh
```
This step produces embeddings for all images, which are stored in the designated output folder for use in classifier training.

#### Step 2: Train classifier and evaluate
The second step trains the multi-label classifier and evaluates performance on the test set. Again, all the paths should be set appropriately in the provided script (`src/model.sh`)

Run the script with:
```bash
bash src/model.sh
```
All outputs — including the best model weight, prediction results, and evaluation metrics — are saved in the directory specified in the script, enabling further analysis.

### Inference
The trained classifier checkpoint (`weights/best.pth`) and a sample script (`src/inference.sh`) are provided for users who want to generate predictions on new images without retraining the model. Before running inference, place the images in the desired input folder and configure the relevant paths in the shell script according to the comments.

Run the script with:
```bash
bash src/inference.sh
```
The inference pipeline extracts DINOv2 embeddings using the frozen backbone and applies the trained classifier checkpoint to generate multi-label predictions. The prediction results are saved in the output directory specified in the script.

The classifier outputs four class probabilities, corresponding to the likelihood of each TCM being present:
- curb extensions
- cycle tracks
- median islands
- speed humps
