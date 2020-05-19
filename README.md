CS231N Project - COVID-19 CXR Neural Network Extensions

[XRays](Chest XRays.png)

Significant works
=================
1. [COVID-Net](https://arxiv.org/pdf/2003.09871.pdf)
   * [Model](https://github.com/lindawangg/COVID-Net)
2. [COVID-19 Image collection Paper](https://arxiv.org/pdf/2003.11597.pdf)
   * [Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
3. [COVID CT Database](https://arxiv.org/pdf/2003.13865.pdf)
   * [Dataset](https://github.com/UCSD-AI4H/COVID-CT)
4. Additional dataset of CXR:
   * [TorchXRayvision](https://github.com/mlmed/torchxrayvision)
5. Pytorch version of COVID-Net https://github.com/IliasPap/COVIDNet
6. Datasets:
   * [Phase 2 RSNA Challenge]] https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

Major work Items
================
1. Port model to Pytorch (from Keras) - Found Pytorch port done.
2. Establish a baseline using existing model and other prediction techniques
3. Data Augmentation
   * Rotation
   * Scale
   * Contrast
   * Noise injection

4. Train Models
    * Given the small amount of COVID-19 data, we will likely need to use transfer learning
5. Evaluate results
6. Write up literasture survey
7. Preare report and Video

Target Dates
============
1. Port by 5/15 - collab in github
2. Data Augmentation infrastructure and first - pull data and add to private gitghub
3. KNN for basesline, random Forest, logistic regression.
