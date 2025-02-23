# MA_DDPM

| Date       | Content              | Problems             |
| :---       | :---                 | :---                 |
| >21.02.2025 | Create a General Structure, with a Preprocessing, Train, Eval and Util files. Important Data Can be set in the Conf.py file. The processing creats n second splits and the converts those to STFT spectograms.  | Using Linear Layers in the VAE will lead to too many Parameters when the VAE does not have a high compression rate. Therefore I implementet a convolutional bottleneck which drastically lowers parameter counts. |
|>23.02.2025 | It seems that even though the results of the VAE got better, due too the Convolutional Architecture the Bottleneck has a very high dimensionality due to larg number of filters. And further Compression is not an option so due to the small size of my spectograms Latent Diffusion seems to be impractical. | VAE does not seem to be the way to go, it induces too much noise and the error does not seem to get lower with lots of epochs. |
|  |  |  |