source # activate the virtual environment

INPUT_PATH_1="..." # path to the existing folder containing image data
INPUT_PATH_2="..." # path to the existing JSON file with labels for the images
INPUT_PATH_3="..." # path to the folder where generated embeddings will be saved
INPUT_PATH_4="..." # path to the existing folder containing the pretrained DINOv2 model weights

export PYTHONPATH=$PWD/dinov2:$PYTHONPATH
python -m embeddings.main $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3 $INPUT_PATH_4