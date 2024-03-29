# pleural_effusion_segmentation

A repository used to train [pleural effusion](https://en.wikipedia.org/wiki/Pleural_effusion) segmentation model.

![example.jpg](media/example.jpg)

## Installation

If you want to train a model, run in terminal code below:
```
git clone https://github.com/JI411/pleural_effusion_segmentation.git
cd pleural_effusion_segmentation
pip install -r requirements.txt
python download_dataset.py
python run.py --accelerator="gpu" --batch=10 --max_epochs=2000 --log_every_n_steps=20
```

You can specify run with pytorch-lightning params, see examples [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-in-python-scripts)

## Examples
1. [Example](https://app.supervise.ly/share-links/CLaWf6xh1Fkwrqqj1WOw1b5pzr2q4gYnIiVnEl7mIooZzGq9PNesu01o431Sp16b) of object in Supervisely project
2. [Augmentations](https://colab.research.google.com/drive/1z8OTP7m3l1p8lYFqQNl81qWg8k4o0yHb?usp=sharing) visualisation
3. [Experiments](https://wandb.ai/lekomtsev/pleural_effusion_segmentation)
4. [Run training in colab](https://colab.research.google.com/drive/1vPlel5uoezxDbhfv8CaPPtB7iRFEF_Jf?usp=sharing)


## Contributing

Install libraries from `requirements-dev.txt`, run pytest and pylint 
(disabled checks can be found in [pylint.yml](https://github.com/JI411/pleural_effusion_segmentation/blob/main/.github/workflows/pylint.yml))
before push. You can see previous experiments in [wandb](https://wandb.ai/lekomtsev/pleural_effusion_segmentation?workspace=user-lekomtsev).


## TODO

### MVP
- [x] Project structure  
- [x] Dataset downloading script
- [x] Dice loss
- [x] Pylint & actions
- [x] Logging
- [x] 2D dataset for pleural effusion
- [x] Unet model from smp
- [x] Visualisation with [Supervisely](https://supervise.ly/) (from DICOM to nrrd format)
- [x] Preprocessing & augmentations
- [x] Use Supervisely dataset format
- [x] Train script
- [x] Dataset caching
- [x] Add 3D dataset
- [x] Add models from [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)

## Next
- [ ] More augmentations
- [x] FP16 training
- [x] ONNX export
- [ ] ONNX export tests 
- [ ] Add losses and metrics
- [ ] Add [SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) or another 3D model from [MONAI](https://github.com/Project-MONAI)
- [ ] Run from config
- [ ] Different channels length
- [ ] Wrap to Docker
- [ ] Accelerate with [voltaML](https://github.com/VoltaML/voltaML)