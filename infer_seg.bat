@echo off
setlocal EnableExtensions
pushd "%~dp0" || exit /b 1
call "%~dp0run_config.bat"
rem Legacy inference imports both project modules and sibling infer modules.
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
if not exist "%MODEL_DIR%\cfg_args" (
    echo Missing trained model. Run train_ff_qa.bat first.
    goto failed
)
python infer\inference.py ^
    --model_path "%MODEL_DIR%" ^
    --mask_folder "%SCENE_DIR%\segmentations" ^
    --output_path "%INFER_OUTPUT%" ^
    --feature_gaussian_iteration %FEATURE_ITERATIONS% ^
    --num_lvl %NUM_LVL% --descriptor_mode rule --show_results
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1
