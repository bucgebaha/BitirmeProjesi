import os, shutil
    
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()

model = models.load_model('deepfake_detect.h5')

import tensorflowjs as tfjs
