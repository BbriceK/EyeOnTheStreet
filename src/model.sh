source # activate the virtual environment


INPUT_PATH_1="..." # path to the existing folder that stores the output embedding files
INPUT_PATH_2="..." # path to the folder where the best model weight will be saved
INPUT_PATH_3="..." # path to the folder where the test set predictions and performance metrics will be saved

python -m models.train $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3

