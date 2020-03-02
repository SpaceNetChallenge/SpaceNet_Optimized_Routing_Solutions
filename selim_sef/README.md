**selim\_sef&#39;s SpaceNet5 Solution Description**

**1. Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Selim Seferbekov
- Handle: selim\_sef
- Placement you achieved in the MM:
- About you: During the day I&#39;m a computer vision engineer and working on similar problems: improving maps with computer vision on aerial or street imagery.
- Why you participated in the MM: 1. quite challenging task 2. open source dataset 3. to apply the skills obtained at previous competitions and to try different network architectures

**2. Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- I solved the task using similar approach provided as baseline with improved neural network architecture, additional targets and additional losses and improved speed predictions.
- First version was with continuous speed masks which gave me score of 51 on public leaderboard.
- Predicting speed as binned masks improved score by 5% straight away.
- At first I used only RGB imagery only but at the last days I  found out that with multichannel imagery validation scores improved by 2%. Unfortunately, only 1 of 4 models was trained with 8 channels.
- I used encoders pretrained on ImageNet and just initialized with He initialization additional input channels. Using pretrained encoders allows network to converge faster and produce better results even if it had less input channels originally.
- I also tried different topology losses:
- Mosinska et al, [https://arxiv.org/abs/1712.02190](https://arxiv.org/abs/1712.02190), improved masks visually , but did not affect score
- Topology preserving loss [https://arxiv.org/abs/1906.05404](https://arxiv.org/abs/1906.05404), which is based on Persistent Homology. As loss computation was on cpu it was way too slow to use in this challenge.

**3. Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- For semantic segmentation I used different variation of UNet architectures with DPN92, InceptionResnetV2 encoders.
- As a loss function I used loss=focal+(1–soft dice). Using both focal (or bce) and soft dice in the loss is crucial to achieve good results in binary semantic segmentation and to get better results with ensembling.
- Augmentations: color jittering/ random sized crops, flips and rotations
- In addition to road masks and speed masks I also predicted junction masks, as an additional attention for NN.
- The final solution has an ensemble of models to produce binary masks. The masks produced by these models are averaged and then vectorized with baseline algorithm.
- I also added additional speed bins wich gave additional +1 on public leaderboard.
- In general I trained 4 folds (2 with DPN92, 2 with InceptionResnetV2 encoders). For validation I used APLS length and dice score and then used best checkpoints on these metrics for inference.
- 1 fold was trained on multichannel imagery and it brought significant improvement on validation.
- I made two stage training: 1st stage with small crops (384x384) each fold on 1 GPU for 24 hours.  2nd stage with bigger crops (512x512) and SyncBN on 4 gpus, 6 hours for each fold.

**4. Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- Docker, [https://www.docker.com](https://www.docker.com/) (Apache License 2.0)
- Nvidia-docker, [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker), ( BSD 3-clause)
- Python 3, [https://www.python.org/](https://www.python.org/), ( PSFL (Python Software Foundation License))
- Numpy, [http://www.numpy.org/](http://www.numpy.org/), (BSD)
- Tqdm, [https://github.com/noamraph/tqdm](https://github.com/noamraph/tqdm), ( The MIT License)
- Anaconda, [https://www.continuum.io/Anaconda-Overview](https://www.continuum.io/Anaconda-Overview),( New BSD License)
- OpenCV, [https://opencv.org/](https://opencv.org/) (BSD)
- Pytorch https://pytorch.org/ (BSD)

**5. Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- With 4 folds trained on multichannel imagery it would much better handle data from mystery city.
- I guess generalization would also be better if data is equally sampled from different cities during training.

**6. Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- The current approach doesn&#39;t handle overpasses and bridges.

**7. Deployment Guide**

Please provide the exact steps required to build and deploy the code:

- In this contest, a Dockerized version of the solution was required, which should run out of the box

**8. Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

- The algorithm can be executed by the instructions provided for the contest.