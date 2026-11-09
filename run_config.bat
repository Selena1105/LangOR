@echo off
rem Shared configuration for training and legacy inference.
set "ROOT_DIR=%~dp0data"
set "DATASET=4dor"
set "SCENE_NAME=004_010951"
set "SCENE_ITERATIONS=100"
set "FEATURE_ITERATIONS=300"
rem Current feature trainer uses one level and 32 channels.
set "NUM_LVL=1"
set "FEATURE_DIM=32"
set "SCENE_DIR=%ROOT_DIR%\%DATASET%\%SCENE_NAME%"
set "MODEL_DIR=%SCENE_DIR%\output"
set "INFER_OUTPUT=%~dp0output\%DATASET%\%SCENE_NAME%"
