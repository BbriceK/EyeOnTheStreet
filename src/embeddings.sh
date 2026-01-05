source # activate the virtual environment

INPUT_PATH_1="..." # path to the image data folder
INPUT_PATH_2="..." # path to the label JSON file
INPUT_PATH_3="..." # path to the folder that saves output embeddings
INPUT_PATH_4="..." # path to the pretrained DINOV2 weight

export PYTHONPATH=$PWD/dinov2:$PYTHONPATH
python -m models.train $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3