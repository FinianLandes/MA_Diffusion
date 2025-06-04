from archisound import ArchiSound

import torch
import soundfile
import numpy as np

autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")
with open('ae_config.txt', 'w') as f:
    print(autoencoder, file=f)


"""
audio = load_audio_file("Data/DA2407_ADO.wav",48000, False)[:, :2**18]
x = torch.tensor(audio).reshape(1, 2, 2**18)
z = autoencoder.encode(x)
y = autoencoder.decode(z, num_steps=20) 
y = normalize(y[0].cpu().numpy(), -0.99999, 0.99999)[0]
soundfile.write("test.wav", y, 48000)
"""