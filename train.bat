@echo off
setlocal EnableExtensions
pushd "%~dp0" || exit /b 1
call "%~dp0run_config.bat"
python train_scene.py -s "%SCENE_DIR%" --model_path "%MODEL_DIR%" ^
    --iterations %SCENE_ITERATIONS% --eval ^
    --test_iterations %SCENE_ITERATIONS% --save_iterations %SCENE_ITERATIONS%
if errorlevel 1 goto failed
python train_affinity_features.py -m "%MODEL_DIR%" ^
    --iterations %FEATURE_ITERATIONS% --feature_dim %FEATURE_DIM% --eval ^
    --test_iterations %FEATURE_ITERATIONS% --save_iterations %FEATURE_ITERATIONS%
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1
