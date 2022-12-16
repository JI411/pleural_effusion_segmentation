# pleural_effusion_segmentation

A repository used to train [pleural effusion](https://en.wikipedia.org/wiki/Pleural_effusion) segmentation model. 

## Installation

If you want to train model, run in terminal code below:
```
git clone https://github.com/JI411/pleural_effusion_segmentation.git
cd pleural_effusion_segmentation
pip install -r requirements.txt
python download_dataset.py
python run.py --accelerator="gpu" --max_epochs=100
```

You can specify run with pytorch-lightning params, see examples [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-in-python-scripts)

## Contributing

Install libraries from `requirements-dev.txt`, run pytest and pylint 
(disabled checks can be found in [pylint.yml](https://github.com/JI411/pleural_effusion_segmentation/blob/main/.github/workflows/pylint.yml))
before push.



## TODO

### MVP
- [x] Project structure  
- [x] Dataset downloading script
- [x] Torch dataset for pleural effusion
- [x] Unet model from smp
- [x] Dice loss
- [x] Pylint & actions
- [x] Unet3D model from [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
- [ ] Visualisation 
- [ ] Preprocessing & augmentations  
- [x] Train script
- [x] Logging
- [x] Create a dataloader with variable-size input with collate_fn
- [x] Dataloader for 3D slices with, for UnetSMP3D 

## Next
- [ ] Dataset caching
- [ ] Move concatenating dataloader, model, transform etc. to pytorch-lightning module, 
create different child classes for each combination 
- [ ] Use [NestedTensors](https://pytorch.org/tutorials/prototype/nestedtensor.html) instead of padding
- [ ] Add more augmentations
- [ ] Add models from [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)

### Tests
  - [x] test_dataset
  - [x] test_dice
  - [x] test_wrappers