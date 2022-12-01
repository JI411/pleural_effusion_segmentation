# pleural_effusion_segmentation

A repository used to train [pleural effusion](https://en.wikipedia.org/wiki/Pleural_effusion) segmentation model. 

## Installation

If you want train model, run in terminal code below:
```
git clone https://github.com/JI411/pleural_effusion_segmentation.git
cd pleural_effusion_segmentation
pip install -r requirements.txt
python download_dataset.py
python run.py config/best.yml
```

You can specify run with your own config, see examples [here](https://github.com/JI411/pleural_effusion_segmentation/tree/main/config)

## Contributing

Install libraries from `requirements-dev.txt` and run pytest and pylint 
(disabled checks out can find in [pylint.yml](https://github.com/JI411/pleural_effusion_segmentation/blob/main/.github/workflows/pylint.yml))
before push.



## TODO

### MVP
- [x] Project structure  
- [x] Dataset downloading script
- [x] Torch dataset for pleural effusion
- [x] Unet model from smp
- [x] Dice loss
- [x] Pylint & actions
- [ ] Visualisation 
- [ ] Preprocessing & augmentations  
- [ ] Train script
- [ ] Logging

### Tests
  - [x] test_dataset
  - [x] test_dice
  - [ ] test_train
  - [ ] test_preprocessing