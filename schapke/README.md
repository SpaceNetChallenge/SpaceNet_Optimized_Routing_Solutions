**schapke&#39;s SpaceNet5 Solution Description**

**1. Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: João Gabriel Schapke
- Handle: schapke
- Placement you achieved in the MM: 5
- About you:  I&#39;m a computer science student.
- Why you participated in the MM: Learn and practice machine learning and deep learning.

**2. Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- I made an ensemble of convolutional networks, each using different loss functions.
- I used the UNet architecture with ResNet as backbone.
- I tried using SOTA segmentation models (Deeplab and OCR) but they didn&#39;t give good results.

**3. Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- Using data from SN3.
- Using different loss functions for the models, BCE, Focal, Dice.

**4. Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- Cresi, https://github.com/CosmiQ/cresi  (Apache 2)
- Pytorch, [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)  (BSD)
- Networkx, https://en.wikipedia.org/wiki/NetworkX (BSD)

**5. Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- Test different backbone models for unet, in particular I would test EfficientNets.
- Use stochastic weight averaging (SWA).
- Use auto augmenting algorithms for better augmentations.

**6. Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- The algorithm is prone to fail to recognize roads covered by buildings, shadows, trees.
- The dataset have much more paved roads than unpaved, I infer that the model won&#39;t be as performant on them.

**7. Deployment Guide**

Please provide the exact steps required to build and deploy the code:

Install docker and nvidia-docker in an environment with access to gpus
2. Build the docker image by running &quot;docker build -n container\_name .&quot;  inside the project folder

**8. Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

1. Start the docker container with &quot;docker run -it -gpus all container\_name&quot;
2. Use the train script: &quot;./train.sh \&lt;path-to-dataset\&gt;&quot;
3. Use the test script: &quot;./test \&lt;path-to-dataset\&gt; \&lt;output-path\&gt;&quot;