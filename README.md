# LangOR
[MICCAI 2026] LangOR: 3D Language Field Reconstruction for Operating Room

![image](https://github.com/Selena1105/LangOR/blob/main/teaser.png)


## Data format


```text
data/<dataset>/<scene>/
|-- images/
|   `-- <image_name>.png
|-- sparse/0/
|   |-- cameras.bin
|   |-- images.bin
|   `-- points3D.bin
|-- test.txt
|-- language_features/
|   |-- <image_name>_s.npy
|   `-- <image_name>_f.npy
`-- segmentations/
    `-- <image_name>/
        `-- <object_name>.png
```

- `images/`: RGB images registered in the COLMAP reconstruction. Use undistorted images with supported pinhole camera intrinsics. Matching COLMAP text files are also supported; `points3D.ply` can be used for the initial point cloud.
- `test.txt`: one test image name per line, without its extension, such as `camera01_colorimage-010951`. Other registered images form the training set when evaluation is enabled.
- `language_features/<image_name>_s.npy`: integer segment-ID maps with shape `(L, H, W)`, where `L` is the stored level count. Nonnegative IDs index rows of the corresponding feature array; `-1` denotes unassigned pixels. Keep the level order and ID mapping consistent with the prepared data. The existing scene uses shape `(4, 1080, 1440)` and dtype `int32`.
- `language_features/<image_name>_f.npy`: segment embeddings with shape `(N, 384)` and dtype `float32`, where `N` is the number of segment embeddings. The current text encoder is `sentence-transformers/all-MiniLM-L6-v2`; stored embeddings must use the same encoder space.
- `segmentations/<image_name>/<object_name>.png`: binary ground-truth masks with background `0` and foreground `255`, aligned with the corresponding image. Directory names must match image names without extensions. Mask filenames supply the text queries, for example `operating_table.png`. These annotations are required by the current inference and evaluation pipeline.

Set the dataset root and scene name in `run_config.bat` to match this layout.

This version of the code is only intended for reconstructing the language field of an OR using GT masks. The complete code is still being finalized.

## Environment

The validated runtime uses Python 3.10.19, PyTorch 2.2.0 with CUDA 11.8, and PyTorch3D 0.7.7. Install Conda, Git, an NVIDIA driver supporting CUDA 11.8, CUDA Toolkit 11.8, and Visual Studio C++ Build Tools with a CUDA-compatible x64 MSVC toolset and Windows SDK.

### 1. Create the environment

```bat
conda create -n langor python=3.10.19 pip=25.3 -y
conda activate langor
```


### 2. Install Python dependencies

```bat
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

### 3. Build CUDA extensions

```bat
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "MAX_JOBS=4"
where cl
nvcc --version

python -m pip install --no-build-isolation --no-deps --no-cache-dir ./submodules/simple-knn
python -m pip install --no-build-isolation --no-deps --no-cache-dir ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps --no-cache-dir ./submodules/diff-gaussian-rasterization_contrastive_f
python -m pip install --no-build-isolation --no-deps --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Citation

If you find langOR useful in your research or applications, please cite it using the following BibTeX:

```bibtex
@inproceedings{li2026langor,
  title={LangOR: 3D Language Field Reconstruction for Operating Room},
  author={Li, Wei and Li, Ruiyang and Pei, Jialun and Sun, Haowei and Qiu, Shi and Heng, Pheng-Ann},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={241--251},
  year={2026},
  organization={Springer}
}
```

