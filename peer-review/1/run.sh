#!/usr/bin/env bash

if [ -f "/Users/user/miniforge3/etc/profile.d/conda.sh" ]; then
    source /Users/user/miniforge3/etc/profile.d/conda.sh
else
    exit 1
fi

conda activate stat243 || exit 1

cd "$(dirname "$0")"

mkdir -p output

python clean.py
[ $? -ne 0 ] && conda deactivate && exit 1

python models.py
[ $? -ne 0 ] && conda deactivate && exit 1

conda deactivate

echo "Done."