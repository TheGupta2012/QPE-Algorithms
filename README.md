## QPE-Library :atom:
This is a library containing some `basic` and `novel` Quantum Phase Estimation algorithms. The four primary algorithms implemented are- 
  - Basic Quantum Phase Estimation algorithm
  - Iterative Quantum Phase Estimation algorithm
  - Kitaev's Phase Estimation algorithm
  - Statistical Phase Estimation algorithm 🆕

## Steps to use
  1. Clone the repository using `git clone [url_of_repo.git]`. 
  2. If the user is using anaconda distribution to utilise these modules, in the anaconda prompt, make a new virtual environment with `conda create -n yourenvname python=3.9`, activate it by typing `conda activate yourenvname` and install all the dependencies listed in `requirements.txt` file using `pip install -r requirements.txt`.
  3. If not, you should use `pip install -r requirements.txt` in your python environment.
  4. You are now ready to use the modules present in the `modules` directory, inside this enviroment.
  5. If using `conda`, add the current environment to your jupyter notebook by typing `python -m ipykernel install --user --name=yourenvname` followed by `jupyter notebook` to run a jupyter instance in this environment
  6. If you have `jupyter nbextensions` enabled, use [this](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) link to install them in your environment.
  7. All the other folders contain notebooks demonstrating the use of each of these algorithms and examples of simulations for higher dimensional, unitary matrices.
 

## Algorithms
All algorithms have been implemented as python classes and have support for running in the `ibmq-backends` and on the simulator provided by `Aer`.

**Basic Phase Estimation Algorithm**
- This basic QPE algorithm is based on the principles of phase kickback and inverse Quantum Fourier Transform.
- A variant called `fast_QPE` has also been included which uses binary exponentiation to reduce the number of unitary applications in simulation. Note that this is only applicable for simulation purposes but not in real scenarios. 
- Class Names : `basic_QPE`, `fast_QPE`
- Module Path : `modules/vanilla_qpe.py`, `modules/faster_basic_qpe.py`
- Main folder : `Basic QPE`

<hspace><hspace>
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ.JPG" align = "left" height = 50% width = 45% title = "Basic QPE">
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ_optimized.JPG" align = "left" height = 50% width = 45% title = "Optimized QPE">
<br><br><br><br><br><br><br><br><br><br>

  
**Iterative Phase Estimation Algorithm**
- The iterative phase estimation algorithm(IQPE) is based on the principle that reducing the width of the circuit in exchange for its depth results in smaller circuits which reduce the *interaction* between qubits thus, reducing errors. 
- It has a distinctive feature called `unknown` which uses binary exponentiation to reduce the number of unitary applications in simulation. This feature is only utilised when `unknown` is set to `False`.
- This algorithm proves as one of the best phase estimation routines for the present day *NISQ computers*.
  
  
- Class Name : `general_IQPE`
- Module Path : `modules/iterative_qpe.py` 
- Main folder : `Iterative QPE` 

<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Iterative%20QPE/IQPE_circ.JPG" height = 45% width = 58% title = "IQPE Circuit">



**Kitaev's Phase Estimation Algorithm**
- Kitaev's algorithm for Phase Estimation is an algorithm with two forms. In this implementation, the algorithm which uses a *single Unitary matrix* for phase estimation is used.
- Kitaev's algorithm is a very efficient algorithm in terms of quantum execution. Involving some classical post processing work and relatively simple circuits for the phase estimation, the only drawback for this algorithm is the number of shots required for a given precision and probability of error scale up very quickly.
- Class Name : `KQPE`
- Module Path : `modules/kitaev_qpe.py`
- Main Folder : `Kitaev's Algorithm`

<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Kitaev's%20Algorithm/KQPE_circ_1qubit.JPG" height = 35% width = 53% title = "Kitaev's Circuit">


**Statistical Phase Estimation Algorithm** 🆕
- All the above algorithms suffer from a limitation that each one of them requires **the eigenvector** of the unitary matrix for the phase estimation procesdure.
- Statistical Phase Estimation Algorithm or SPEA is a novel approach for phase etimation based on [this](https://arxiv.org/pdf/2104.10285.pdf) recent paper. SPEA uses a variational approach to solve the phase estimation and *does not* require the eigenvector of the unitary matrix to be prepared beforehand.
- It proposes to give an **eigenstate and eigenvalue** pair of our Unitary in one successful execution of the algorithm. This can also be extended to find the full *spectral decomposition* of a matrix.
- This library contains the original algorithm proposed by the authors and **a modified approach** that uses a global maximum approach to update the state during the intermediate iterations of the algorithm.
- The new **modified approach** was proposed keeping in mind that greedy choices in the algorithm may not always propose to be optimal. One advantage of this approach is in terms of *quantum execution* time. 
- Since any quantum computer contains a classical controller through which it is accessed, calling the device multiple times incurs overhead in terms of the classical interfacing. While original appraoach uses *exponential* API calls for its execution, the modified approach only requires only a *constant* number of API calls to reach the optimal result.
  
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Statistical%20QPE/spea_circuit.PNG" height = 67% width = 55%>
 
- **Original Algorithm**
  - Class Name - `SPEA`
  - Module Path - `modules/normal_SPEA.py`
  - Main Folder - `Statistical QPE`
  
- **Modified Algorithm**
  - Class Names - `global_max_SPEA` , `bundled_global_max_SPEA`, `bundled_global_max_alt_SPEA`
  - Module Path - `modules/changed_SPEA.py`
  - Main Folder - `Statistical QPE`

  
  
