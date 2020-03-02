**ikibardin&#39;s SpaceNet5 Solution Description**

**1. Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Ilya Kibardin
- Handle: ikibardin
- Placement you achieved in the MM: 4
- About you: Computer Vision Engineer at X5 Retail Group
- Why you participated in the MM: The match seemed to me as a great opportunity to sharpen my skills in image segmentation on satellite data. Besides that, I wanted to learn to extract graph data from segmentation maps.

**2. Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- I started developing my solution from the baseline provided by the organizers. After training the original baseline, I replaced the resnet34 + unet model with resnet50 + fpn, because FPN decoder usually works better of multiclass semantic segmentation tasks. This change significantly improved baseline performance.
- After looking at baseline predictions I was not satisfied by the quality of binary road masks. So I decided to use baseline masks to estimate the speed while training separate models for binary segmentation of the roads.
- I tried training models with topology-aware segmentation loss because BCE/IoU/Dice and other classical loss functions do not really represent the quality of segmentation road map. After visual analysis, it seemed to me that the results were better with it, but the validation score was lower than with ordinary loss functions, so I dropped the models from the final ensemble.
- I also experimented with predicting speed using an FPN network with softmax activation to predict smoothened speed bin for every pixel. The network was trained using KL divergence as the loss function. I did not manage to get it to perform better than the baseline speed predictions so I did not use it in my final solution.
- I noticed that baseline postprocessing can connect adjacent nodes of the graph. The models seemed to fails often on three-way crossroads, so I thought that it might be a good idea to connect not only adjacent nodes but also a node to an edge if they are closer than some threshold. Implementing such postprocessing boosted my score on local validation so I decided to keep for the final solution.

**3. Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- Training images and masks are prepared using baseline code provided by the organizers.
- The baseline is trained replacing resnet34 unet with resnet50 fpn. The model is then used for speed estimation.
- An ensemble of models trained for road binary segmentation. The models include resnext101 unet, se\_resnext50 unet and resnet50 fpn. Unet models are trained in 4 stages:

	1. Encoder frozen, 224x224 crops
	2. Full model training, 224x224 crops
	3. Encoder frozen, 512x512 crops
	4. Full model training, 512x512 crops.

- FPN model is trained using only two last stages.
- Every model is trained to optimizer 1\*Dice + 3\*FocalLoss loss function. Adam optimizer with an initial learning rate of 0.0001 with a reduction of LR on plateau policy is used for every stage.
- A minimalistic set of augmentations including random crops, scales, flips and rotates was used. Besides, a random HSV shift improved the models&#39; performance a bit.
- The inference is executed on full images by padding them to 1344x1344. Test time augmentation is used for a minor score boost.

**4. Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- numpy ( https://pypi.python.org/pypi/numpy), BSD
- pandas ( https://pypi.python.org/pypi/pandas), BSD
- pencv-python ( https://pypi.python.org/pypi/opencv-python), MIT
- pytorch ( http://pytorch.org/), BSD
- torchvision ( https://pypi.python.org/pypi/torchvision), BSD
- GDAL ( https://anaconda.org/conda-forge/gdal), MIT
- sknw ( https://github.com/yxdragon/sknw) , BSD
- APLS ( https://github.com/CosmiQ/apls) and it&#39;s requirements
- scipy ( https://pypi.python.org/pypi/scipy), BSD
- scikit-image ( https://pypi.python.org/pypi/scikit-image), Modified BSD
- tqdm ( https://pypi.python.org/pypi/tqdm), MPLv2, MIT
- albumentations ( [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)), MIT
- tensorboardX ( https://pypi.python.org/pypi/tensorboardX), MIT
- pretrainedmodels ( [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)), BSD
- PyYAML ( [https://github.com/yaml/pyyaml](https://github.com/yaml/pyyaml)), MIT
- Docker ( [https://www.docker.com/](https://www.docker.com/)), Apache
- nvidia-docker ( [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)), BSD

**5. Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- Same as for any other deep learning algorithm, the solution should strongly benefit from obtaining more annotated data for training.
- It seems that there should exist a better way of estimating road speed.

**6. Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- The algorithm seems to struggle when executed on images with certain weather conditions.
- Good performance is not guaranteed on new locations due to possible lack of generalization,
- However only single models could be used instead of the whole ensemble, the algorithm is still quite computationally expensive.

**7. Deployment Guide**

Please provide the exact steps required to build and deploy the code:

- The solution should be deployed using docker. Follow the same flow as for any other dockerized system.

**8. Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

1. Build the docker image of the solution and run a container with it.
2. Use train.sh and test to train and execute models, as required by the competition rules.