# QPE-Library :atom:

* [Overview](#overview)
* [Steps to use](#steps-to-use)
* [Algorithms](#algorithms)
    - [Basic QPE Algorithm](#basic-qpe-algorithm)
    - [Iterative QPE Algorithm](#iterative-qpe-algorithm)
    - [Kitaev's QPE Algorithm](#kitaevs-qpe-algorithm)
    - [Statistical QPE Algorithm](#statistical-qpe-algorithm)

## Overview
This is a *library* containing some `basic` and `novel` Quantum Phase Estimation algorithms. The four primary algorithms implemented are- 
  - Basic QPE algorithm
  - Iterative QPE algorithm
  - Kitaev's QPE algorithm
  - Statistical QPE algorithm 🆕

## Steps to use
  1. Clone the repository using `git clone [url_of_repo.git]`. 
  2. If using anaconda distribution, in the anaconda prompt, make a new virtual environment with `conda create -n yourenvname python=3.9`, activate by typing `conda activate yourenvname` and install dependenices using `pip install -r requirements.txt`.
  3. If not, use `pip install -r requirements.txt` in your python environment.
  4. You are now ready to use the modules present in the `modules` directory, inside this enviroment.
  5. If using `conda`, add the current environment to your jupyter notebook by typing `python -m ipykernel install --user --name=yourenvname` followed by `jupyter notebook` to run a jupyter instance in this environment
  6. If you have `jupyter nbextensions` enabled, use [this](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) link to install them in your environment.
 

## Algorithms
All algorithms have been implemented as python classes and have support for running on an `IBMQBackend` and on the simulator provided by `Aer`. For the examples, each subfolder contains a jupyter notebook depicting the usage of the implemented algorithms.

### Basic QPE Algorithm
- This basic QPE algorithm is based on the principles of phase kickback and inverse Quantum Fourier Transform.
- A variant called `fast_QPE` has also been included which uses binary exponentiation to reduce the number of unitary applications in simulation. Note that this is only applicable for simulation purposes but not in real scenarios. 
- Class Names : `basic_QPE`, `fast_QPE`
- Module Path : `modules/vanilla_qpe.py`, `modules/faster_basic_qpe.py`
- Main folder : `Basic QPE`
- Examples : [Notebook](https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/Basic%20QPE.ipynb)

<hspace><hspace>
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ.JPG" align = "left" height = 50% width = 45% title = "Basic QPE">
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ_optimized.JPG" align = "left" height = 50% width = 45% title = "Optimized QPE">
<br><br><br><br><br><br><br><br><br><br>

  
### Iterative QPE Algorithm
- The iterative phase estimation algorithm(IQPE) is based on the principle that reducing the width of the circuit in exchange for its depth results in smaller circuits which reduce the *interaction* between qubits thus, reducing errors. 
- The module has a distinctive feature called `unknown` which uses binary exponentiation to reduce the number of unitary applications in simulation. This feature is only utilised when `unknown` is set to `False`.
- This algorithm proves as *one of the best* phase estimation routines for the present day *NISQ computers*.
  
  
- Class Name : `general_IQPE`
- Module Path : `modules/iterative_qpe.py` 
- Main folder : `Iterative QPE` 
- Examples and Tests: [Notebook](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Iterative%20QPE/Iterative%20QPE.ipynb)
    
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Iterative%20QPE/IQPE_circ.JPG" height = 55% width = 78% title = "IQPE Circuit">
  
  
### Kitaev's QPE Algorithm
- Kitaev's algorithm for Phase Estimation is an algorithm with two forms. In this implementation, the algorithm which uses a *single Unitary matrix* for phase estimation is used.
- Kitaev's algorithm is a very efficient algorithm in terms of quantum execution. Involving some classical post processing work and relatively simple circuits for the phase estimation, the only drawback for this algorithm is the number of shots required for a given precision and probability of error scale up very quickly.
- Class Name : `KQPE`
- Module Path : `modules/kitaev_qpe.py`
- Main Folder : `Kitaev's Algorithm`
- Examples : [Notebook](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Kitaev%27s%20Algorithm/Kitaev%20QPE.ipynb)
    
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Kitaev's%20Algorithm/KQPE_circ_1qubit.JPG" height = 48% width = 73% title = "Kitaev's Circuit">


### Statistical QPE Algorithm 
   
- Statistical Phase Estimation Algorithm or SPEA is a novel approach for phase etimation based on [this](https://arxiv.org/pdf/2104.10285.pdf) recent paper. SPEA uses a variational approach to solve the phase estimation and **does not** require the eigenvector of the unitary matrix to be prepared beforehand.
- It proposes to give an **eigenstate and eigenvalue** pair of our Unitary in one successful execution of the algorithm which can be extended to find the full *spectral decomposition* of a matrix.
- This library contains the original algorithm proposed by the authors and **a modified algorithm**. The **modified approach** was proposed keeping in mind that greedy choices in the original algorithm may not always propose to be optimal. One advantage of this approach is in terms of *quantum execution* time. 
- Since any quantum computer contains a classical controller through which it is accessed, calling the device multiple times incurs overhead in terms of the classical interfacing. While original appraoach uses *exponential* API calls for its execution, the modified approach only requires only a *constant* number of API calls to reach the result.
  
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/spea_circuit.PNG" height = 73% width = 75%>
 
- **Original Algorithm**
  - Class Name - `SPEA`
  - Module Path - `modules/normal_SPEA.py`
  - Main Folder - `Statistical QPE`
  - Examples
    - [Standard Approach](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/Statistical%20QPE.ipynb)
    - [Alternate Approach](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/Alternate%20Approach/SPEA%20-%20Original%20algorithm%20and%20Alternate%20approach.ipynb)
- **Modified Algorithm** 
  - Class Names - `global_max_SPEA` , `bundled_global_max_SPEA`, `bundled_global_max_alt_SPEA`
  - Module Path - `modules/changed_SPEA.py`
  - Main Folder - `Statistical QPE`
  - Examples : 
    - [Standard Approach](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/Statistical%20QPE-Changed%20Algorithm%20.ipynb)
    - [Alternate Approach](https://nbviewer.jupyter.org/github/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/Alternate%20Approach/SPEA%20-%20Changed%20algorithm%20and%20Alternate%20approach.ipynb)
  
