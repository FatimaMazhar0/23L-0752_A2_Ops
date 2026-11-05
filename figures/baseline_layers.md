| Layer | Output shape | Params |
|---|---|---|
| image (InputLayer) | (None, 160, 160, 3) | 0 |
| rescale (Rescaling) | (None, 160, 160, 3) | 0 |
| conv1 (Conv2D) | (None, 160, 160, 32) | 864 |
| bn1 (BatchNormalization) | (None, 160, 160, 32) | 128 |
| relu1 (Activation) | (None, 160, 160, 32) | 0 |
| pool1 (MaxPooling2D) | (None, 80, 80, 32) | 0 |
| conv2 (Conv2D) | (None, 80, 80, 64) | 18,432 |
| bn2 (BatchNormalization) | (None, 80, 80, 64) | 256 |
| relu2 (Activation) | (None, 80, 80, 64) | 0 |
| pool2 (MaxPooling2D) | (None, 40, 40, 64) | 0 |
| conv3 (Conv2D) | (None, 40, 40, 128) | 73,728 |
| bn3 (BatchNormalization) | (None, 40, 40, 128) | 512 |
| relu3 (Activation) | (None, 40, 40, 128) | 0 |
| pool3 (MaxPooling2D) | (None, 20, 20, 128) | 0 |
| conv4 (Conv2D) | (None, 20, 20, 256) | 294,912 |
| bn4 (BatchNormalization) | (None, 20, 20, 256) | 1,024 |
| relu4 (Activation) | (None, 20, 20, 256) | 0 |
| pool4 (MaxPooling2D) | (None, 10, 10, 256) | 0 |
| gap (GlobalAveragePooling2D) | (None, 256) | 0 |
| dropout (Dropout) | (None, 256) | 0 |
| prob_ai (Dense) | (None, 1) | 257 |
| **Total** | | **390,113** |