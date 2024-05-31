#!/usr/bin/env bash
# procedure.sh
# Usage: bash procedure.sh

# environment activate
pyenv activate py3_8


datarootpath="./dcms"

# preprocess
echo "Image Data Preprocessing"
python3 preprocess/img_pre.py -r $datarootpath

# contour finding
echo "Image Data Contour Finding"
python3 preprocess/img_con.py -r $datarootpath
