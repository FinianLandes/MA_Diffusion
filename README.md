# MA_DDPM

| Date       | Content              | Problems             |
| :---       | :---                 | :---                 |
| >21.02.2025 | Create a General Structure, with a Preprocessing, Train, Eval and Util files. Important Data Can be set in the Conf.py file. The processing creats n second splits and the converts those to STFT spectograms.  | Using Linear Layers in the VAE will lead to too many Parameters when the VAE does not have a high compression rate. Therefore I implementet a convolutional bottleneck which drastically lowers parameter counts. |
| | | |