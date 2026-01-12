# EyeOnTheStreet

Prior work has shown that many Canadian cities lack accurate and complete records of traffic calming measures (TCMs). TCMs are physical modifications to roads that help reduce traffic collisions, yet incomplete information limits the evaluation of their real-world road safety impacts and hinders equitable, data-driven urban planning. Manually identifying TCMs across all locations and years at the city scale is impractical. To address this, this work presents a **baseline multi-label classification model** for identifying four TCMs in default-angle Google Street View (GSV) images from Toronto and Montréal: **curb extensions, cycle tracks, median islands, and speed humps**. Applying this approach to historical imagery **enables the construction of a geospatial database that tracks when and where TCMs were implemented**, supporting longitudinal safety analysis and policy evaluation. For details, see the associated paper; this work was presented in the UbiComp4VRU Workshop at UbiComp ’25.

## Challenges
Detecting these four TCMs in GSV images presents several non-trivial challenges, particularly because we use **default-angle GSV images**: the locations of TCMs are not known in advance, so capturing targeted camera angles or zoom levels for every site would be exhaustive and impractical.

- Functionally defined target objects: TCMs are defined by their function rather than a standardized visual appearance. Therefore, their identification often depends on contextual cues within the street environment rather than distinctive visual features.

- High variability within the same category: even within the same TCM category, visual appearance can vary substantially. For example, a cycle track is physically separated from the roadway, and the separation may be bollards, concrete barriers, or a slightly raised pavement. Furthermore, the appearance of a given category may differ across cities due to local policies.

- Temporal environment factors at longitudinal scale: Images of the same location can differ substantially over time due to changes in weather, lighting conditions, seasons, and infrastructure updates.

- Occlusion and visual clutter: as a result of relying on default-angle GSV images, TCMs often appear small, partially occluded, or embedded within cluttered urban scenes rather than captured as focal objects.

- Class imbalance: the uneven geographic distribution of TCMs means that some measures are common while others are rare or absent in many locations, resulting in highly imbalanced label frequencies.

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

## Project Structure

- `data/` – training, validation, and test images, and the label file.
- `src/` – source code, including scripts for generating embeddings, training the classifier, and sample shell script.
- `weights/` – the pretrained dinov2 weight.

## Usage

### Step 1: Generate image embeddings
A sample script (`src/embeddings.sh`) is provided to compute image embeddings. Before running the script, configure all the paths according to the comments. 

Run the script with:
```bash
bash src/embeddings.sh
```
This step produces embeddings for all images, which are stored in the designated output folder for use in classifier training.

### Step 2: Train classifier and evaluate
The second step trains the multi-label classifier and evaluates performance on the test set. Again, all the paths should be set appropriately in the provided script (`src/model.sh`)

Run the script with:
```bash
bash src/model.sh
```
All outputs — including the best model weight, prediction results, and evaluation metrics — are saved in the directory specified in the script, enabling further analysis.
