"""lungct: training-stability and architecture comparison for 2D lung CT segmentation.

Modules
-------
paths        repository folders (data, results)
data         slice loading, intensity mapping, augmentation, data splits
models       U-Net, Attention U-Net, ResU-Net, ResUNet++
metrics      Dice, IoU, accuracy, sensitivity, specificity, calibration
training     configuration and training/evaluation of one run (Algorithm 1)
experiments  the experiments of the article (seeded, resumable)
"""

__version__ = "1.0.0"
