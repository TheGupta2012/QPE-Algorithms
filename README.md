## QPE-Library :atom:
This is a library containing some `basic` and `novel` Quantum Phase Estimation algorithms. The four primary algorithms implemented are- 
  - Basic Quantum Phase Estimation algorithm
  - Iterative Quantum Phase Estimation algorithm
  - Kitaev's Phase Estimation algorithm
  - Statistical Phase Estimation algorithm 🆕

## Steps to use
  1. Clone the repository using `git clone [url_of_repo.git]`. 
  2. If the user is using anaconda distribution to utilise these modules, in the anaconda prompt, make a new virtual environment with `conda create -n yourenvname python=3.9` and install all the dependencies listed in `requirements.txt` file using `pip install -r requirements.txt`.
  3. If not, you should use `pip install -r requirements.txt` in your python environment.
  4. You are now ready to use the modules present in the `modules` directory, inside this enviroment.
  5. All the other folders contain notebooks demonstrating the use of each of these algorithms and examples of simulations for higher dimensional, unitary matrices.
 

## Algorithms
All algorithms have been implemented as python classes and have support for running in the `ibmq-backends` and on the simulator provided by `Aer`.

**Basic Phase Estimation Algorithm**
- This basic QPE algorithm is based on the principles of phase kickback and inverse Quantum Fourier Transform.
- This class has a distinctive feature called `unknown` which uses binary exponentiation to reduce the number of unitary applications in simulation. This feature is only utilised when `unknown` is set to `False`. 
- Class Name : `basic_QPE`
- Module Path : `modules/vanilla_qpe`
- Main folder : `Basic QPE`

<hspace><hspace>
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ.JPG" align = "left" height = 230 width = 370 title = "Basic QPE">
<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Basic%20QPE/QPE_circ_optimized.JPG" align = "right" height = 230 width = 380 title = "Optimized QPE">
<br><br><br><br><br><br><br><br><br><br>

  
**Iterative Phase Estimation Algorithm**
- The iterative phase estimation algorithm(IQPE) is based on the principle that reducing the width of the circuit in exchange for its depth results in smaller circuits which reduce the *interaction* between qubits thus, reducing errors. 
- This algorithm proves as one of the best phase estimation routines for the present day *NISQ computers*.
- Class Name : `general_IQPE`
- Module Path : `modules/iterative_qpe` 
- Main folder : `Iterative QPE` 

<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Iterative%20QPE/IQPE_circ.JPG" height = 260 width = 620 title = "Kitaev's Circuit">



**Kitaev's Phase Estimation Algorithm**
- Kitaev's algorithm for Phase Estimation is an algorithm with two forms. In this implementation, the algorithm which uses a *single Unitary matrix* for phase estimation is used.
- Kitaev's algorithm is a very efficient algorithm in terms of quantum execution. Involving some classical post processing work and relatively simple circuits for the phase estimation, the only drawback for this algorithm is the number of shots required for a given precision and probability of error scale up very quickly.
- Class Name : `KQPE`
- Module Path : `modules/kitaev_qpe`
- Main Folder : `Kitaev's Algorithm`

<img src = "https://github.com/TheGupta2012/QPE-Algorithms/blob/master/QPE/Kitaev's%20Algorithm/KQPE_circ_1qubit.JPG" height = 200 width = 530 title = "Kitaev's Circuit">


**Statistical Phase Estimation Algorithm** 🆕
- All the above algorithms suffer from a limitation that each one of them requires **the eigenvector** of the unitary matrix for the phase estimation procesdure.
- Statistical Phase Estimation Algorithm or SPEA is a novel approach for phase etimation based on [this](https://arxiv.org/pdf/2104.10285.pdf) recent paper. SPEA uses a variational approach to solve the phase estimation and *does not* require the eigenvector of the unitary matrix to be prepared beforehand.
- This library contains the original algorithm proposed by the authors and **a modified approach** which aims to speed up the current "quantum execution" time *exponentially*
  
  // to do...

