**cannab&#39;s SpaceNet5 Solution Description**

**1. Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Victor Durnov
- Handle: cannab
- Placement you achieved in the MM: 2nd
- About you:  I&#39;m independent Software Developer/Data Scientist interested in hard algorithmic challenges and machine learning
- Why you participated in the MM: I liked previous SpaceNet challenges a lot

**2. Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- My approach is heavily based on my solutions from SpaceNet3 ([https://github.com/SpaceNetChallenge/RoadDetector/tree/master/cannab-solution](https://github.com/SpaceNetChallenge/RoadDetector/tree/master/cannab-solution)) and SpaceNet4 ([https://github.com/SpaceNetChallenge/SpaceNet\_Off\_Nadir\_Solutions/tree/master/cannab](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/cannab))
- The main part is Neural Networks (NN) for segmentation with encoder-decoder architecture similar to UNet  ( https://arxiv.org/pdf/1505.04597.pdf ). Used different pretrained encoder for ensembling (Transfer Learning). Models taken from SpaceNet4 solution and use 9 channels as input (PAN + PS-MS)
- Postprocessing to build road graph taken from SpaceNet3 + added functions for speed and distance processing from baseline solution
-  ([https://github.com/CosmiQ/cresi](https://github.com/CosmiQ/cresi))

**3. Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- Finally, I&#39;ve used 3 Neural Network architectures for ensemble with same UNet-like decoder and different pretrained encoders:
  - Pretrained SE-ResNeXt50 encoder, trained with 2 different random 5% of data used as validation. Then best checkpoints was tuned further on images resized to 960\*960. So, 4 checkpoints in total.
  - Pretrained DPN92 encoder, trained on 960\*960 images with 2 different random 5% of data used as validation. Then best checkpoints was tuned further on original images. 4 checkpoints in total.
  - Pretrained ResNet34 encoder, trained on 8 folds. 2 models trained on original images + 2 models trained from scratch on  960\*960 images.
- Neural Network input is an image with 9 channels: 8 channels from PS-MS +  1 PAN channel (expected this to be consistent according to training dataset and previous competitions)
- Neural Network input is an image with 12 channels:
  - 1 – Main road mask
  - 2 – junctions (for more attention and to not broke the roads)
  - 3-11 – road masks by speed bins (15, 18.75, 20, 25, 30, 35, 45, 55, 65)
  - 12 – continuous speed (used  as auxiliary output only)
-  5 points taken between two road&#39;s endpoints to calculate the speed

**4. Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- Anaconda as base Python 3 environment, [www.anaconda.com](http://www.anaconda.com/)
- Pytorch, [https://pytorch.org](https://pytorch.org/)
- Pretrained models, [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- OpenCV, [https://opencv.org](https://opencv.org/) BSD License

**5. Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- Bigger ensemble, more training data, train more time with more augmentations.

**6. Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- Works bad on corrupted data if some channels are missing.

**7. Deployment Guide**

Please provide the exact steps required to build and deploy the code:

- Dockerized version prepared as requested.

**8. Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

- train.sh and test.sh scripts meet required specification.
