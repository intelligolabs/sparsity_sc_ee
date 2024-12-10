# Enhancing Split Computing and Early Exit Applications through Predefined Sparsity #

Official implementation of the paper [Enhancing Split Computing and Early Exit Applications through Predefined Sparsity](intelligolabs.github.io/sparsity_sc_ee/) accepted at the 27th Forum on specification and Design Languages (FDL 2024).

## Installation ##
**1. Repository setup:**
* `$ git clone https://github.com/intelligolabs/sparsity_sc_ee.git`
* `$ cd sparsity_sc_ee`
* `$ mv mnist.npz mnist/`

**2. Conda environment setup:**
* `$ conda create -n sparsity_sc_ee python=3.8`
* `$ conda activate sparsity_sc_ee`
* `$ pip install -r requirements.txt`

## Run Sparsity SC&EE ##
To run Sparsity SC&EE, use the file `main.py`.
In particular, the `launch.sh` file contains two launch script examples that you can use to modify the default configuration.

## Credits ##
We would like to thank Sourya Dey for the repository [predefinedsparse-nnets](https://github.com/souryadey/predefinedsparse-nnets).

## Authors ##
Luigi Capogrosso<sup>1</sup>, Enrico Fraccaroli<sup>1,2</sup>, Giulio Petrozziello<sup>3</sup>, Francesco Setti<sup>1</sup>, Samarjit Chakraborty<sup>2</sup>, Franco Fummi<sup>1</sup>, Marco Cristani<sup>1</sup>

<sup>1</sup> *Department of Engineering for Innovation Medicine, University of Verona, Italy*

<sup>2</sup> *Department of Computer Science, The University of North Carolina at Chapel Hill, USA*

<sup>3</sup> *Department of Computer Science, University of Verona, Italy*

## Citation ##
If you use [**Sparsity SC&EE**](https://ieeexplore.ieee.org/abstract/document/10673767), please, cite the following paper:
```
@InProceedings{capogrosso2024enhancing,
  author    = {Capogrosso, Luigi and Fraccaroli, Enrico and Petrozziello, Giulio and Setti, Francesco and Chakraborty, Samarjit and Fummi, Franco and Cristani, Marco},
  booktitle = {Forum on Specification \& Design Languages (FDL)},
  title     = {{Enhancing Split Computing and Early Exit Applications through Predefined Sparsity}},
  year      = {2024},
  doi       = {10.1109/fdl63219.2024.10673767},
}
```
