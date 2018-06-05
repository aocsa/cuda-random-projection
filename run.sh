#!/bin/bash
# use predefined variables to access passed arguments
#echo arguments to the shell

DB_NAME=synth16d
DB_TRAIN=home/aocsa/datasets/$DB_NAME/$DB_NAME.TRAIN
DB_SIZE=10000
WORD_SIZE=8
CARDINALITY=8
MASK_SIZE=7
ITERATIONS=7
THREADS=256
ARCH=sm_61
MAIN=RandomProjection.cu
EXE=rp-$DB_NAME.exe


echo "Arguments to Random Projection"
echo "DB_NAME = $DB_NAME"
echo "DB_TRAIN = $DB_TRAIN"
echo "DB_SIZE = $DB_SIZE"
echo "EXE = $EXE"

nvcc $MAIN -O3 -w -o $EXE -arch=$ARCH -DDATASET=$DB_TRAIN -DNUM_WORDS=$DB_SIZE -DWORD_SIZE=$WORD_SIZE -DCARDINALITY=$CARDINALITY -DMASK_SIZE=$MASK_SIZE -DITERATIONS=$ITERATIONS -DTHREADS=$THREADS

echo  "Execution"
nvcc -run ./$EXE