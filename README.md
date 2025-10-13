# MA Diffusion

**How can I use generative AI to create music based on DJ sets as training data?**

## General Info

### Usage

To reproduce results or experiment with the models yourself:

1. **Clone the repository**
   - **Option 1: Clone with Git**

     ``` bash
     git clone https://github.com/FinianLandes/MA_Diffusion.git
     cd MA_Diffusion
     ```

   - **Option 2: Download as ZIP**
     - Go to the repository page: [MA_Diffusion](https://github.com/FinianLandes/MA_Diffusion)
     - Click the green **Code** button → **Download ZIP**
     - Extract the ZIP file on your computer and open the folder in your IDE (e.g., VS Code or Jupyter).

2. **Install dependencies**
  Install the required dependencies as listed below.

3. **Preprocess your dataset**
    In all files set ```remote_kernel = False``` if you are running on your local machine.
    Open and run `Preprocessing.ipynb`.
    Adjust input/output directories in the first cell to match your audio files.
    (You can skip this step if you only want to run the pretrained model.)

4. **Train a model**
    Open `Wave Diffusion.ipynb`.
    Set parameters (epochs, batch size, etc.) in the configuration cell.
    Run all cells to start training, you can also continue to train a pretrained model with your data.
    (Model checkpoints are automatically saved, with the current epoch and deleted if training successfully finishes.)
    (You can skip this step if you only want to run the pretrained model.)

5. **Generate samples**
    Use Wave Diffusion `Inference.ipynb`.
    Load the trained weights (e.g. WaveDiffusion_v6.pth) or your own model.
    Run the generation cells to produce and listen to new 4-second audio samples.

### Directory Details

- **Files And Results**
  - **Test Data** - Selection of Samples from the test set.
  - **Training Data** - Selection of Files from the training set.
  - **Model/Type Name** - Results from different methods/models.

- **Libraries**
  - `Utils.py` - General utility functions and the Trainer class for the newer models.
  - `U_Net.py` - U-Net architecture definition.
  - `Diffusion.py` - Diffusion model implementation.

- **MainScripts**
  - `Preprocessing.ipynb` - Notebook for data preprocessing.
  - `Wave Diffusion.ipynb` - Notebook for traing the diffusion models.
  - `Wave Diffusion Inference.ipynb` - Notebook for generating samples using diffusion and also to save the model architecture.

The model weights for the final model can be found at: [Wave Diffusion v6](https://huggingface.co/Chaerne/DirectWaveDiffusion_v6/blob/main/WaveDiffusion_v6.pth)

### Prerequisits

- **Python**
  - This was written with Python `3.13.2` but older versions should work aswell.

- **External libraries**
  - `Numpy`: 2.1.3
  - `Torch`: 2.6.0
  - `Librosa`: 0.10.2 (Depending on the python version might require `standard-sunau`, `standard-aifc` and `standard-chunk` which have been removed from the pre-installed libraries in newer python versions.)
  - `Matplotlib`: 3.10.0
  - `Soundfile`: 0.13.1
  - **Optional**
    - `tensorboard`: 2.19.0
    - `torchviz`: 0.0.3
    - `torchinfo`: 1.8.0
    - `optuna`: 4.3.0
    - `plotly`: 6.1.2

- **Pre-Installed Libraries**
  - `os`
  - `sys`
  - `logging`
  - `time`
  - `typing`

### Logging

This codebase is based on the logging module. For the minimal output set logging level to `logging.INFO`. Due to the immense output of some libraries in the `logging.DEBUG` mode I added a custom mode between `DEBUG` and `INFO`. Inorder to use this level which debugs the custom impelemented functions, set debug level to `LIGHT_DEBUG`.

## Sources

Some Additional Sources and learning resources not mentioned in the main Paper.

### Youtube Videos

- [Diffusion paper explanation by Outlier](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [Diffusion implementation by Outlier](https://www.youtube.com/watch?v=TBCRlnwJtZU)
- [Diffusion explanation by ExplainingAI](https://www.youtube.com/watch?v=H45lF4sUgiE)
- [Diffusion implemtation by ExplainingAI](https://www.youtube.com/watch?v=vu6eKteJWew)
- [Explanation UNET by rupert ai](https://www.youtube.com/watch?v=NhdzGfB1q74)

## Work Journal

| Date       | Content              | Problems             | Next Steps           | Time Spent (Training and MA paper writing times not included) |
| :---       | :---                 | :---                 | :---                 | :---       |
| >21.02.2025 | Created a general structure with Preprocessing, Train, Eval and Util files. Configurations set in `conf.py`. Processing creates *n*-second splits and converts them to STFT spectrograms. | Using linear layers in the VAE led to too many parameters when compression rate was low. Implemented a convolutional bottleneck, which drastically lowered parameter count. | Test if bottleneck still captures enough structure for music data. | 7h |
| >23.02.2025 | Convolutional bottleneck improved results, but latent dimensionality too high due to large number of filters. Further compression not feasible with small spectrograms. Training conv_VAE_v2 on Paperspace (lr=1e-4, batch=32, >1000 epochs, dataset 640/1280). Reprod loss (*weight)=1.9e5, KL=4e4. | VAE induces too much noise. Loss plateaued even after many epochs. Latent diffusion seems impractical at this resolution. | Consider abandoning VAE path for audio and testing alternatives. | 5h |
| >03.03.2025 | Implemented diffusion + UNet. First try failed (pure noise). Found issues: UNet used Sigmoid output, wrong for noise prediction; dataset scaled to [0,1] instead of [-1,1]. Fixed both. Added gradient scaling, mixed precision training, gradient accumulation (batch=8, ~450k params). Second run started better, but loss spiked at epoch 50 and didn’t recover. | Output still unusable. Sampling extremely slow (47 min/sample on CPU). | Optimize UNet for efficiency, test smaller models, or switch to bigger GPU. | 6h15 |
| >04.03.2025 | Refactored code based on Outlier’s implementation. Decided to generate lower-quality audio first, then upsample with a second model (SOTA approach for high resolution). | Scaling up increases memory usage. | Plan secondary neural net for upsampling. | 4h15 |
| >05.03.2025 | Adjusted preprocessing: switched sr=32k (0–16kHz). Tested mel spectrograms but reconstruction poor → stayed with STFT. Changed fft_len=480, hop_len=288. 4s audio now ~93k values instead of 688k for 8s. | Mel failed for music reconstruction. | Continue tuning STFT params for balance between quality and size. | 4h30 |
| >06.03.2025 | Fixed bugs, trained more models. Replaced attention with SE blocks (too heavy), then switched to double conv blocks. Changed cosine noise schedule to linear (cosine unstable). | SE blocks underperformed. Cosine schedule broke training. | Keep testing noise schedules + block designs. | 3h |
| >07.03.2025 | First noisy but structured outputs using Conv_UNet. Trained on 1280 samples, 300 epochs (~100 epochs/h). ~17M param NN takes ~10min/epoch (batch=16, accum=2). | Outputs low-contrast and noisy. | Explore normalization + loss tweaks. | 1h30 |
| >09.03.2025 | Output improved slightly but still poor range. Found normalization bug: used dataset-level min/max instead of per-sample, causing big inconsistencies. | Wrong normalization distorted training signal. | Fix normalization to per-sample scaling. | 2h30 |
| >15.03.2025 | Tried schedulers with higher lr (~0.08) → beneficial. Switched to BatchNorm, caused instability → reverted to GroupNorm(8). Tested GELU vs SiLU. | Inference unstable with BatchNorm. | Stick with GroupNorm, keep SiLU. | 2h45 |
| >17.03.2025 | Reached loss ~0.08 with UNet tweaks + custom lr scheduler. 72M param model + smaller 18M variant trained on 4k samples. | Large model is memory-heavy. | Optimize for GPU memory (checkpointing, mixed precision). | 5h30 |
| >20.03.2025 | Removed modulation layers, returned to model v3 (previously worked better). | Performance degraded after removing features. | Re-assess which features are critical. | 4h |
| >24.03.2025 | Added extra skip connections (Luke Ditra style). Training unstable: after some epochs highs vanish; with few epochs outputs noisy. | Instability across epochs. | Add early stopping, test other loss weights. | 3h |
| >23.04.2025 | Expanded UNet with more blocks. Created new version based on Flavio Schneider’s design. Hit size and memory issues. | Out-of-memory with big UNet. | Reduce block depth or add efficient layers. | 11h |
| >28.04.2025 | Training on CIFAR-10 to benchmark against UNetV0 (Flavio Schneider, Archisound). | Image results ≠ audio, risk of mismatch. | Compare architectures on both audio + CIFAR. | 1h |
| >30.05.2025 | Added Trainer class in utils → cleaner training loop. Added gradient clipping, partially stabilized training. | Still some instability, clipping not a full fix. | Add lr warmup/cosine restarts. | 13h |
| >01.06.2025 | Moving closer to Archisound/Mousai. Switched to mel spectrograms with fewer bins, treated frequency bins as channels (1D net) → faster. Started vocoder implementation for mel→waveform. | Vocoder not yet functional. | Implement/test HiFiGAN. | 5h |
| >02.06.2025 | Better results with plain diffusion (loss=9.4e-2) but temporal features missing. Switched to diffusion autoencoder (Mousai). Slower: 10min/epoch @1k samples. Encoder setup unclear (paper underspecified). | Temporal coherence lacking. Encoder params unknown. | Reproduce encoder setup from Mousai or experiment. | 2h45 |
| >03.06.2025 | Switched to magnitude encoder (like Mousai). Contacted Flavio Schneider about encoder mismatch → no response. | Encoder setup still unclear. | Keep testing variants. | 5h |
| >07.06.2025 | Implemented MelGAN vocoder. Produced waveform but very noisy. | MelGAN quality poor. | Replace with HiFiGAN. | 3h |
| >10.06.2025 | Getting results. Hyperparam tuning with Optuna. Training upsampler for diffusion outputs. | Noisy results. | Continue Optuna search. | 5h |
| >17.06.2025 | Implemented HiFiGAN. Better than MelGAN. Training diffusion to output better spectrograms. Also training HiFiGAN enhancer to reduce aliasing. | Alias artifacts remain. | Tune HiFiGAN enhancer further. | 5h |
| >06.07.2025 | Tested models on Maestro dataset to rule out dataset issues → results worse. Started writing matura paper. | Dataset dependency unclear. | Refactor utils, create git tag for reproducibility. | 5h |
| >10.08.2025 | Tested AudioDiffusion PyTorch with Optuna. No good results. | Parameters not optimal. | Deeper tuning. | 10h |
| >20.08.2025 | Implemented VQ-VAE. Training + inference reconstructions ok. Built simple transformer, only ~10% accuracy, unusable outputs. | Likely bad token representation or dataset→token conversion bug. | Debug tokenizer + try continuous latent spaces. | 5h|
| >25.08.2025 | VQ-VAE + Transformer (fixed conversion bug). Accuracy ~15%, outputs still poor. Switched to continuous VAE. VAE samples collapse, KL loss balancing hard. Latent diffusion not viable if latent space bad. | Latent space collapse. Poor recon quality. | Tune KL weight, try β-VAE or VQ fallback. | 3h30 |
| >07.09.2025 | Reimplemented Wave Diffusion with v-objective. Fixed earlier formula errors. Increased receptive field, improved UNet (res blocks). Promising results (low freq captured, no highs). | High frequency info missing. | Add multi-band diffusion or frequency loss. | 16h15 |
| >08.09.2025 | Tried band-split diffusion. No success. Added band-weighted loss, which improved outputs. | Band diffusion ineffective. | Keep band-weighted loss. | 2h |
| >18.09.2025 | Created architecture diagram for Diffusion v6 (draw.io).  | | | 4h |
| >23.09.2025 | Wrote algorithm overview/pseudocode in LaTeX for all diffusion models. Made graphics for matura paper. Cleared up code files and added samples to github. | | | 5h15 |

The practical part of the project (excluding model training) took roughly 130 hours. Writing the thesis probably took 60–80 hours, though this is just an estimate since I often wrote in short sessions, so exact timing is hard to track.
