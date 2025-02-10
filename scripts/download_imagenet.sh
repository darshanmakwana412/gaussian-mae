#!/bin/bash

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipeline fails
set -x  # Print commands being executed

# Define dataset directory
DATASET_DIR="datasets/imagenet"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# 1. Download the data
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

# 2. Extract the training data
mkdir -p train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# 3. Extract the validation data and move images to subfolders
mkdir -p val && mv ILSVRC2012_img_val.tar val/ && cd val
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..

# 4. Extract the test data
mkdir -p test && mv ILSVRC2012_img_test_v10102019.tar test/ && cd test
tar -xvf ILSVRC2012_img_test_v10102019.tar && rm -f ILSVRC2012_img_test_v10102019.tar
cd ..

# 5. Remove a known corrupted image (optional, but recommended)
CORRUPTED_IMAGE="train/n04266014/n04266014_10835.JPEG"
if [ -f "$CORRUPTED_IMAGE" ]; then
    rm "$CORRUPTED_IMAGE"
    echo "Removed corrupted image: $CORRUPTED_IMAGE"
fi

echo "ImageNet dataset setup completed in $DATASET_DIR"