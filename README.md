# MA_DDPM

## Work Journal

| Date       | Content              | Problems             |
| :---       | :---                 | :---                 |
| >21.02.2025 | Create a General Structure, with a Preprocessing, Train, Eval and Util files. Important Data Can be set in the Conf.py file. The processing creats n second splits and the converts those to STFT spectograms.  | Using Linear Layers in the VAE will lead to too many Parameters when the VAE does not have a high compression rate. Therefore I implementet a convolutional bottleneck which drastically lowers parameter counts. |
|>23.02.2025 | It seems that even though the results of the VAE got better, due too the Convolutional Architecture the Bottleneck has a very high dimensionality due to larg number of filters. And further Compression is not an option so due to the small size of my spectograms Latent Diffusion seems to be impractical. Training in Paperspace of conv_VAE_v2 with batch size lr = 1-e4, batch = 32. >1000 epochs on dataset 640 and 1280, weight of reprod loss = 10'000. Reprod loss (*weight) = 1.9e5, KL = 4e4 | VAE does not seem to be the way to go, it induces too much noise and the error does not seem to get lower with lots of epochs. |
| >03.03.2025 | I implemented the Diffusion and the corresponding UNET the first try was very unsuccesfull(Totally noisy output no characteristics whatsoever). Looking closer at the data it seems that UNET was implemented incorrectly with scaling the noise to 0-1 as last layer (Sigmoid) which prevented it from precisely predicting noise, furthermore i had to switch the training data interval to [-1 1] rather than [0, 1] as the noise added is of form N(0, I). Furthermore i added gradient scaling and mixed precision training with cuda and also the possibility to use gradient accumulation, as i can only train with batch size 8 due to the model having ~450k params. The second run started better, but the loss spiked at epoch 50 and didnt go down any furhter from that point on. Second training run also ends in no senseful data, noise too high but training seemed fine. | Optimize the model further to make it more memory efficient or upgrade and use bigger GPU's. Inference or sampling is also an issue atm as it takes 47 minutes to create a single Sample on my i7 CPU. |
| >04.03.2025 | Refactoring the code to be based on the version by Outlier. Due to the increase in Model size i want to now generate bad quality audio and then later use a second neural net to upsample the audio. THis seems to be the state-of-the-art solution for big resolution outputs. | |
| >05.03.2025 | Played around with preprocessing to create smaller data while still having the best possible quality. Switch sr to 32k(so 0-16k Hz). Tried mel spectograms but due to the bad reconstruction and no significant increase in compression stayed with stft diagrams but modified fft_len=480 and hop_len=288. for 4 sec audio this gives me 93184 values rather than 688k for 8sec with the previous settings.| |
| >06.03.2025 | Fixed a lot of small bugs and started traing more models, Switched the Attention layer due to complexity and param count for an SE block. THis did not seem to produce meaningful data therefore i switched to Double conv blocks instead of SE blocks. Also switched to a linear noise schedule as the cosine is not working somehow. | |
| >07.03.2025 | Finally some output is generated it does not containg high contrast and is very noisy but a pattern can be seen. This was achieved by training with Conv_UNET, 1280 samples, 300 epochs (100 epochs/h). Also when training my ~17M param NN takes 10mins per epoch with batch 16, accum. of 2, when training with the full 7.8k samples| |
| >09.03.2025 | The outputs seemd to have gotten better, but still with a bad output range. Which led me to taking a closer look at my data which got me to the realisation that my normalization might be bad and signle sample might differ a lot due to the normalization taking the min and max value of a whole set rather than the individual sample.||
| >15.03.2025 | Tried another training run with other schedulers a higher Lr seemd to be beneficial getting down to ~0.08. Also switch to Batchnorm but this seemed to have lead to numerical instability during inference so i switched back to groupnorm with 8 groups. After run 1 switched from GELU to SiLU just to see if it would make a difference.  |  |
| >17.03.2025 | I finally got to a loss >0.08 using manipulation in my unet and a custom lr scheduler. The model now has 72M params. Trained with 4000 samples on another smaller model 18M params | |
| >20.03.2025 | Removed modulation and trying to get back to model v3 which produced some outputs.| |
| >24.03.2025 | Added extra  Skip connection. Matching the implementation of Luke Ditra. This is now giving me some results again, seemingly training is not completely stable as after some epochs all highs seem to get removed but with low epoch counts the outputs are very noisy  | |
| >23.04.2025 | Added more blocks and structures to the unet file, also created a new unet more based on the one proposed by Flavio schneider, but having size and memory issues there. ||
| >28.04.2025 | Training on the cifar10 dataset now to test the architecture and also to compare the results to the unetV0 by Flavio schneider for Archisound. | |

## General Info
### Directory Details

- **Data**
  - `datasets.npy` - Preprocessed dataset file (e.g., spectrograms).
  - `music_file.wav` - Example audio file for processing.

- **Libraries**
  - `VAE.py` - Variational Autoencoder implementation.
  - `Diffusion.py` - Diffusion model implementation.
  - `Utils.py` - General utility functions.
  - `U_Net.py` - U-Net architecture definition.

- **MainScripts**
  - `Conf.py` - Configuration settings.
  - `Preprocessing.ipynb` - Notebook for data preprocessing.
  - `Train Diffusion.ipynb` - Notebook for training the diffusion model.
  - `Eval Diffusion.ipynb` - Notebook for evaluating the diffusion model.
  - `Train VAE.ipynb` - Notebook for training the VAE.
  - `Eval VAE.ipynb` - Notebook for evaluating the VAE.

- **Models**
  - Directory for saved model weights (e.g., `.pth` files).

- **Results**
  - Directory for experiment results and outputs (e.g., generated spectrograms, loss plots).

### Prerequisits

- **Python**
  - This was written with Python `3.13.2` but older versions should work aswell.

- **External libraries**
  - `Numpy`: 2.1.3
  - `Torch`: 2.6.0
  - `Librosa`: 0.10.2 (Depending on the python version might require `standard-sunau`, `standard-aifc` and `standard-chunk` which have been removed from the pre-installed libraries in newer python versions.)
  - `Matplotlib`: 3.10.0
  - `Soundfile`: 0.13.1

- **Pre-Installed Libraries**
  - `os`
  - `sys`
  - `logging`
  - `time`
  - `typing`

### Neural Nets

Inorder to view the Neural Nets Netron can be used with .pt files the code for the generation of this file can be found in the eval file.

### Logging

This codebase is based on the logging module. For the minimal output set logging level to `logging.INFO`. Due to the immense output of some libraries in `logging.DEBUG` mode i added a custom mode between `DEBUG` and `INFO`. Inorder to use this level which prints a lot of info in the custom implemented functions set debug level to `LIGHT_DEBUG`. This is defined in the `conf.py` file.

### Data

The dataset is created in the preprocessing file. The Model is trained on 4s mono samples with a sample rate of 32Khz. The STFT spectograms then are created with an FFT length of 480 and a HOP length of 288. The spectrograms then are resized to the closest multiple of 32 which makes divisions in the UNET easy. This results in spectograms of shape 1x224x416.

## Sources

### Papers

- [Original DDPM Paper(Sohl-Dickstein et. al 2015)](http://arxiv.org/pdf/1503.03585)
- [DDPM(Ho et. al 2020)](https://arxiv.org/pdf/2006.11239) (This is the most useful one and the one most of my implementation is based on).
- [Improved DDPM(Nichol et. al 2021)](https://arxiv.org/pdf/2102.09672) (Explanation of cosine schedule).
- [DDIM(Song et. al 2022)](https://arxiv.org/pdf/2010.02502) (The faster generation method).
- [Diffusion for music gen. ETH (Flavio Schneider 2023)](https://arxiv.org/pdf/2301.13267) (A good overview over methods).

### Youtube Videos

- [Paper explanation by Outlier](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [Implementation by Outlier](https://www.youtube.com/watch?v=TBCRlnwJtZU)
- [Explanation by ExplainingAI](https://www.youtube.com/watch?v=H45lF4sUgiE)
- [Implemtation by ExplainingAI](https://www.youtube.com/watch?v=vu6eKteJWew)
- [Explanation UNET by rupert ai](https://www.youtube.com/watch?v=NhdzGfB1q74)

