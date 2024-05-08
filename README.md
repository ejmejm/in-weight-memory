# In-Weight Memory Research

This repository contains experiments aimed at developing a neural network architecture and training algorithm suitable for continual/life-long learning. Check out [this document](https://docs.google.com/document/d/14UhgI_etzgN-xBXNBGq5r12U6rGMgDzsSkLNVas4HQQ/edit?usp=sharing) for more details.


## Installation

1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`

It's that simple!


## Implementation

The implementation builds on top of Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repository. The main changes are to the `minGPT/mingpt/model.py` file, where memory neurons have been added to each of the model's feed-forward layers. The number of memory neurons can be controlled by the `model.fc_mem_dim` parameter in the config, and setting this value to 0 will result in a standard transformer.

## Usage

Currently, the project is in a very early stage, and the memory implementation is not working as intended yet. The first goal is to figure out how to train the the in-weight memory so that a model can learn to memorize a given sequence. If you want to work on this, you should check out the `reconstruction_experiment.ipynb` notebook. The notebook is documented and explains the current state of the experiment.

Though the memory does not help the model yet, you can still train a model using the following command:

```bash
python train.py --config={config_path}
```

There are some default configs already provided in the `configs/` directory. For example, you can train a very simple model with memory with the following command:

```bash
python train.py --config=configs/mem.yaml
```

While the training should work and log to wandb, the memory itself does not yet work as intended. This is a work in progress, and getting it to work is the point of the project!

## Contributing

There are not any formal guidelines for contributing yet, but this is intended to be built on by others. For now, reach out to me via one of the SNs in my profile if you're interested in contributing, and I will add you to the Discord group for contributing. All ideas regarding the project will be discussed there.