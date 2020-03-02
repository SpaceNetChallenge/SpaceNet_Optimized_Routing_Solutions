**XD\_XD&#39;s SpaceNet5 Solution Description**

**1. Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Kohei Ozaki
- Handle: XD\_XD
- Placement you achieved in the MM: 1st place
- About you: Software Engineer at Preferred Networks, inc.
- Why you participated in the MM: To study the effectiveness of deep learning algorithms in satellite imagery and in particular road network prediction problems.

**2. Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- Similar to CRESIv2, I solved the problem as a multi-channel segmentation problem.
- Since the unknown city had a large weight in the final evaluation, I used all given cities for validation to prevent overfitting.
- What things don&#39;t work out:
  - Mixed precision training. I tried to speed up the learning step with NVIDIA apex, but I didn&#39;t use it in the final submission. For training a SE-ResNeXt model, the dynamic range of the average pooling part in the SE-module was large. So, I implemented the pooling part of the SE-module with fp32. NVIDIA apex reduced training time of SE-ResNeXt50 from 38 minutes to 28 minutes for each epoch. However, the loss did not always converge.
  - I tried to use Joint learning of orientation, Deep layer aggregation, Lovasz loss, Spatial Squeeze and Channel Excitation (cSE) Block and Attention layer. In my experiment, SE-ResNeXt50 was the best choice of model in this competition. Deeper encoders like SE-ResNeXt101 improved the validation loss but took significantly longer time to train.
  - I also tried to use CLAHE for pre-processing and data augmentation. It didn&#39;t improve the validation loss significantly.
  - PS-RGB, non-RGB channels in PS-MUL and PAN didn&#39;t help to improve the validation score significantly.

**3. Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- Modeling
  - I used the baseline code provided by the competition host and solution codes in SpaceNet3.
  - To avoid overfitting, all six cities were used for the validation set.
  - Ensemble 8 models: 4 \* ResNet50 + 4 \* SE-ResNeXt50. Optimizer=Adam, Loss=dice+focal.
  - For model selection, I used the mean dice score of 4 folds as the metric.
- Post-processing
  - Speed conversion bins were the same as the bins defined in the baseline.
  - In addition to the baseline post-processing, my solution removed small connected components that were disconnected from large connected components. This idea was based on the hypothesis that roads that are not connected to public roads are private roads or noise.

**4. Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- \* click (https://github.com/pallets/click), BSD
- \* easydict (https://github.com/makinacorpus/easydict), Lessor GNU Public
- \* networkx (https://networkx.github.io/), BSD
- \* osmnx (https://osmnx.readthedocs.io/en/stable/), MIT
- \* geopandas (https://github.com/geopandas/geopandas), BSD
- \* imageio (https://github.com/imageio/imageio), BSD-2-Clause
- \* statsmodels (https://github.com/statsmodels/statsmodelsJ), Modified BSD
- \* tqdm (https://pypi.python.org/pypi/tqdm), MPLv2, MIT
- \* numpy (https://pypi.python.org/pypi/numpy), BSD
- \* opencv-python (https://pypi.python.org/pypi/opencv-python), MIT
- \* matplotlib (https://pypi.python.org/pypi/matplotlib), BSD
- \* scipy (https://pypi.python.org/pypi/scipy), BSD
- \* scikit-image (https://pypi.python.org/pypi/scikit-image), Modified BSD
- \* scikit-learn (https://pypi.python.org/pypi/scikit-learn), BSD
- \* tensorboardX (https://pypi.python.org/pypi/tensorboardX), MIT
- \* pytorch (http://pytorch.org/), BSD
- \* torchvision (https://pypi.python.org/pypi/torchvision), BSD
- \* gdal (https://anaconda.org/conda-forge/gdal), MIT
- \* apls (https://github.com/CosmiQ/apls) and it&#39;s requirements
- \* pandas (https://pypi.python.org/pypi/pandas), BSD
- \* shapely (https://github.com/Toblerity/Shapely), BSD
- \* RoadDetector (https://github.com/SpaceNetChallenge/RoadDetector), Apache-2.0
- \* cresi (https://github.com/avanetten/cresi), Apache-2.0

**5. Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- Unsupervised (self-supervised) learning with unlabeled data has the potential to improve the robustness of unseen locales. Unfortunately, it requires additional computational costs and is not suitable for this contest.

**6. Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- There is not enough training data to distinguish public and private roads. For example, a private road in a factory or airport.
- Consider a case where an elevated highway and a ground road intersect but are disconnected. Like CRESIv2, my solution can&#39;t handle this correctly. Any two lines that intersect in the 2-D prediction mask always make an intersection node in the predicted road network graph.

**7. Deployment Guide**

Please provide the exact steps required to build and deploy the code:

1. The exact steps are described in a Dockerfile. See `code/Dockerfile` and [smly/dockerfile-pytorch](https://github.com/smly/dockerfile-pytorch).

**8. Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

1. The instructions are described in `xd_xd_intstructions.md`.
