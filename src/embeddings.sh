source # activate the virtual environment

INPUT_PATH_1="..."
INPUT_PATH_2="..."
INPUT_PATH_3="..."
INPUT_PATH_4="..."
WORLD_SIZE=
DIST_URL="..."

export PYTHONPATH=$PWD/dinov2:$PYTHONPATH
python -m embeddings.main $INPUT_PATH_1 $INPUT_PATH_2 $INPUT_PATH_3 $INPUT_PATH_4 --world_size $WORLD_SIZE --dist_url $DIST_URL
