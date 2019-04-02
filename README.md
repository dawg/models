# ML Models for Vusic
[![Build Status](https://travis-ci.org/dawg/models.svg?branch=master)](https://travis-ci.org/dawg/models)

## Introduction
This repo is used to develop the machine learning models used for automated track Separation and music Transcription in [`Vusic`](https://github.com/dawg/models). The track separation modell will use deep learning and blind source separation techniques to automate the process of separating vocals from their accompaniment. The second machine learning model will be used to automate the process of converting piano track to MIDI files that users can apply effects to or convert to sheet music.

## Tasks
Here is an incomplete list of the major tasks that need to be completed. :runner: indicates active development.
- [x] Load and preprocess MUSDB18 dataset.
- [x] Implement an STFT algorithm thats optimized for PyTorch.
- [x] Create spectograms from the STFTs.
- [ ] :runner: Load and preprocess MAPS dataset
- [ ] :runner: Implement RNN Encoder/Decoder for Track Seperation
- [ ] :runner: Implement Denoiser for Track Seperation
- [ ] Implement the BiLTSM for Onset and frame detection (Music Transcription)
- [ ] Implement Constant-Q transform for music transcription
- [ ] Integrate the machine learning models in Vusic 

## Contributing
### Prerequisites
1. Install the dependencies using the following command:

   ```
   pipenv install
   ```

2. Run the following setup script in the project root dirctory to fetch the packages:
   ```
   python setup.py install
   ```

### Environment
1. Setup your environment using the following script:
   ```
   ./environment.sh
   ```

2. Install the requirements using the following script:
   ```
   ./requirements.sh
   ```
### Jupyter Notebook instructions

1. Install all dependencies and enter the pipenv shell
   ```
   pipenv install
   pipenv shell
   ```

2. Create a kernel for your virtual environment
   ```
   python -m ipykernel install --user --name=my-virtualenv-name
   ```

3. Launch the jupyter notebook from your pipenv shell
   ```
   jupyter notebook
   ```

4. Change the notebook kernel to your virtual environment's kernel
   ![](https://i.stack.imgur.com/htimC.png)

### Formatting
Follow [these instructions](https://github.com/ambv/black#pycharm) to set up black in PyCharm.

