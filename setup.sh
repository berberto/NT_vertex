#!/bin/zsh

python3 setup_cent.py build_ext --inplace
python3 setup_fe.py build_ext --inplace
python3 setup_fe_omp.py build_ext --inplace