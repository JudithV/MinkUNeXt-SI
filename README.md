# MinkUNeXt-SI

[Link to the preprint.](https://arxiv.org/abs/2505.17591)

**Abstract**
In autonomous navigation systems, the solution of the place recognition problem is crucial for their safe functioning. But this is not a trivial solution, since it must be accurate regardless of any changes in the scene, such as seasonal changes and different weather conditions, and it must be generalizable to other environments. This paper presents our method, MinkUNeXt-SI, which, starting from a LiDAR point cloud, preprocesses the input data to obtain its spherical coordinates and intensity values normalized within a range of 0 to 1 for each point, and it produces a robust place recognition descriptor. To that end, a deep learning approach that combines Minkowski convolutions and a U-net architecture with skip connections is used. The results of MinkUNeXt-SI demonstrate that this method reaches and surpasses state-of-the-art performance while it also generalizes satisfactorily to other datasets. Additionally, we showcase the capture of a custom dataset and its use in evaluating our solution, which also achieves outstanding results. Both the code of our solution and the runs of our dataset are publicly available for reproducibility purposes.

![Workflow of MinkUNeXt-SI](/imgs/minkunext-si-diagram.png)
