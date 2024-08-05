# Neural Network Branch Gating Exploration

## Description
This project aims to explore the concept of branch gating in neural networks. 
Branch gating is a technique that allows a neural network to learn to selectively 
use different branches based on the input, potentially improving efficiency and performance.

## Installation
Clone this repository to your local machine and install the required packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
pip install -e . 


## Running
The main file for running a single instance of Rotated MNIST is 
branchNetwork/experiments/FuturesLongTaskSequenceRotate.py (I need to update naming). 
The parameters to change are located in the bottom of that file.

The Branching network is branchNetwork/architecture/BranchMM (Branch Matrix Multi) and 
the sub files are branchNetwork/gatingActFunction.py and branchNetwork/BranchLayerMM.py in the main folder. 

Most files should have a test that can be run to make sure each piece is working. 

