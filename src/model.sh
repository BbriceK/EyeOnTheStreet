source # activate the virtual environment


INPUT_PATH_1="..." # path to the folder that stores embedding files
INPUT_PATH_2="..." # path to save the best model weight
INPUT_PATH_3="..." # path to save the test set predictions

python -m models.train $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3

