# EyeOnTheStreet

This work provides a **baseline multi-label classification model** for identifying traffic calming measures from Google Street View images in Toronto and Montreal, as presented in the UbiComp4VRU Workshop at UbiComp ’25. **The associated paper has now been published and is available for access**.

Traffic calming measures are physical modifications to roads that help reduce traffic collisions. This baseline focuses on four measures:

- **Curb extensions**  
- **Cycle tracks**  
- **Median islands**  
- **Speed humps**

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
