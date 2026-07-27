source # activate the virtual environment

INPUT_PATH_1="..." # path to the folder that stores the target images
INPUT_PATH_2="..." # path to the folder that stores the output embedding files
INPUT_PATH_3="..." # path to the pretrained dinov2 weight
INPUT_PATH_4="..." # path to the folder where the best model weight was saved
INPUT_PATH_5="..." # path to the folder where the inference results will be saved

export PYTHONPATH=$PWD/dinov2:$PYTHONPATH
python inference.py $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3 $INPUT_PATH_4 $INPUT_PATH_5

