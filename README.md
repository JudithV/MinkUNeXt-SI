# MinkUNeXt-SI

[Link to the preprint.](https://arxiv.org/abs/2505.17591)

**Abstract**

In autonomous navigation systems, the solution of the place recognition problem is crucial for their safe functioning. But this is not a trivial solution, since it must be accurate regardless of any changes in the scene, such as seasonal changes and different weather conditions, and it must be generalizable to other environments. This paper presents our method, MinkUNeXt-SI, which, starting from a LiDAR point cloud, preprocesses the input data to obtain its spherical coordinates and intensity values normalized within a range of 0 to 1 for each point, and it produces a robust place recognition descriptor. To that end, a deep learning approach that combines Minkowski convolutions and a U-net architecture with skip connections is used. The results of MinkUNeXt-SI demonstrate that this method reaches and surpasses state-of-the-art performance while it also generalizes satisfactorily to other datasets. Additionally, we showcase the capture of a custom dataset and its use in evaluating our solution, which also achieves outstanding results. Both the code of our solution and the runs of our dataset are publicly available for reproducibility purposes.

The following diagram illustrates the workflow of MinkUNeXt-SI.
![Workflow of MinkUNeXt-SI](/imgs/minkunext-si-diagram.png)

## Setup
To test our method, you must install the dependencies listed in the *requirements.txt* file. Specifically, the most important dependencies for this neural network are NumPy, Pandas, PyTorch, and MinkowskiEngine.
Run the command:

`pip install -r requirements.txt`

## Generation of train and test sets
Before training the network, the necessary training and testing sets must be generated. Depending on the dataset used for training, the corresponding file in the *datasets/pointnetvlad* folder must be executed.

`python3 datasets/pointnetvlad/generate_training_tuples_[dataset].py`

Then, for validation purposes, the database and query sets must be generated as well. To do so, execute the following file after editing the parameters corresponding to the dataset (neighborhood radius and query/test area dimensions). **The last parameter must be configured correctly before executing the training tuple generator to subdivide both sets.**

`python3 datasets/pointnetvlad/generate_test_sets.py`

## Train MinkUNeXt-SI
Once the necessary sets have been generated, edit the hyperparameters in *config/general_parameters.yaml* according to your experiment's needs. Once everything is set up, train the network with the following command:

`python3 training/train.py`
#
