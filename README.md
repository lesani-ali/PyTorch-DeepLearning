# Deep Learning with PyTorch

This repository aims to provide clear and concise implementations of various deep learning models using PyTorch. The current focus is on Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs). The models are trained and evaluated on the MNIST dataset, with results visualized and documented in the `main.ipynb` Jupyter Notebook. Future updates will include additional architectures and extended evaluations on other datasets.


## Implemented Algorithms

1. VGG11 (CNN)
2. Multi-Layer Perceptron (MLP)


## Dataset

The MNIST dataset is used to test the implementations of the algorithms. MNIST is a dataset of handwritten digits and is widely used for training and testing in the field of machine learning.


## Project Structure
```
PyTorch-DeepLearning/
├── classes/                     # Directory containing class implementations
│   ├── __init__.py
│   ├── CNN.py                   # Implementation VGG11
│   ├── MLP.py                   # Implementation of Multi-Layer Perceptron
│   ├── utils.py                 # Utility functions (load_data, calculate_accuracy, load_data, train, etc)
├── main.ipynb                   # Notebook demonstrating the use of algorithms
├── .gitignore                   # Git ignore file
├── environment.yml              # Python dependencies
└── README.md                    # Project README file
```

## Cloning Repository from GitHub
Use this command to clone the repository from GitHub: <br>
`git clone git@github.com:lesani-ali/PyTorch-DeepLearning.git`<br>


## Environment Setup
Please follow these steps to set up the working environment:
1. Install Miniconda (use terminal to execute these commands):
    - Create a Directory:<br>
    `mkdir -p ~/miniconda3`
    - Download the Miniconda Installer:<br>
    `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh`
    - Run the Miniconda Installer:<br>
    `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
    - Remove the Installer Script:<br>
    `rm -rf ~/miniconda3/miniconda.sh`
    - Add to path:<br>
    `source ~/miniconda3/bin/activate`
    - Initialize for bash and zsh shells:<br>
    `~/miniconda3/bin/conda init bash`<br>
    and <br>
    `~/miniconda3/bin/conda init zsh`

    For more information visit [Miniconda](https://docs.anaconda.com/miniconda/) website.

2. Create conda environment and install all packages (I used "data_collection" as a name for my environment): <br>
`conda env create -f environment.yml`

3. Activate the environment: <br>
`conda activate ML`

## Future Work
- Implementation of additional deep learning architectures (e.g., ResNet, LSTM, GANs).
- Evaluation on different datasets.
- Hyperparameter tuning and optimization experiments.
- Implementation of advanced training techniques (e.g., transfer learning, data augmentation).

## Acknowledgements
- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).
- The VGG11 architecture can be found [here](https://arxiv.org/pdf/1409.1556.pdf).
