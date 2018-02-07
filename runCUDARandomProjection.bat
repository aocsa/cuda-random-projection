:: CUDA Random Projection Motif Discovery

:: 1. Compilation
@ECHO OFF
set SOURCE=src
set BIN=bin
IF NOT EXIST %BIN% MKDIR %BIN%
@ECHO ON

@echo ===========================================================
set DB_NAME=audio
set DB_TRAIN=\"E:/datasets/%DB_NAME%/%DB_NAME%.TRAIN\"
set DB_SIZE=10000
set WORD_SIZE=8
set CARDINALITY=8
set MASK_SIZE=7
set ITERATIONS=7
set THREADS=256
set ARCH=sm_12
set MACHINE=64
set MAIN=RandomProjection.cu
set EXE=%BIN%/rp-%DB_NAME%.exe
@echo ===========================================================


nvcc %SOURCE%/%MAIN% -O3 -w -o %EXE% -arch=%ARCH% -m%MACHINE% -DDATASET=%DB_TRAIN% -DNUM_WORDS=%DB_SIZE% -DWORD_SIZE=%WORD_SIZE% -DCARDINALITY=%CARDINALITY% -DMASK_SIZE=%MASK_SIZE% -DITERATIONS=%ITERATIONS% -DTHREADS=%THREADS%

:: 2. Execution
nvcc -run %EXE%