# EyeOnTheStreet

This work provides a **baseline multi-label classification model** for identifying traffic calming measures from Google Street View images in Toronto and Montreal, as presented in the UbiComp4VRU Workshop at UbiComp ’25 (DOI pending).

Traffic calming measures are physical modifications to roads that help reduce traffic collisions. This baseline focuses on four measures:

- **Curb extensions**  
- **Cycle tracks**  
- **Median islands**  
- **Speed humps**

## Installation
To set up the project, first clone this repository and then install dependencies:
```bash
# create a virtual environment
python -m venv venv
source venv/bin/activate

# install required Python packages
pip install -r requirements.txt
```

This project relies on frozen DINOv2 weights as a backbone. Clone the DINOv2 repository in the same directory:
```bash
git clone https://github.com/facebookresearch/dinov2.git
```
Follow the installation instructions in the official DINOv2 repository: https://github.com/facebookresearch/dinov2
