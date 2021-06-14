## QPE-Library :atom:
This is a library consisting some `basic` and `novel` Quantum Phase Estimation algorithms. The four primary algorithms implemented are- 
  - Basic Quantum Phase Estimation algorithm
  - Iterative Quantum Phase Estimation algorithm
  - Kitaev's Phase Estimation algorithm
  - Statistical Phase Estimation algorithm 🆕

### Steps to use
  1. Clone the repository using `git clone [url_of_repo.git]`. 
  2. If the user is using anaconda distribution to utilise these modules, in the anaconda prompt, make a new virtual environment with `conda create -n yourenvname python=3.9` and install all the dependencies listed in `requirements.txt` file using `pip install -r requirements.txt`.
  3. If not, you should use `pip install -r requirements.txt` in your python environment.
  4. You are now ready to use the modules present in the `modules` directory, inside this enviroment.
  5. All the other folders contain notebooks demonstrating the use of each of these algorithms and examples of simulations for higher dimensional, unitary matrices.
 

### Algorithms
> **Basic Phase Estimation Algorithm**
- This basic QPE algorithm is based on the principles of phase kickback and inverse Quantum Fourier Transforms. The algorithm is implemented as a python *class* and has support to run in the `ibmq-backends` and on the simulator provided by `Aer`. 
- This class has a distinctive feature called `unknown` which uses binary exponentiation to reduce the number of unitary applications in simulation. This feature is only utilised when `unknown` is set to `False`. 
- Module Name : `basic_QPE`
- File Name : `vanilla_QPE.py`
- Path : `modules/Basic QPE/`
 
 // to do...
> **Iterative Phase Estimation Algorithm**
- 
> **Kitaev's Phase Estimation Algorithm**
- 
> **Statistical Phase Estimation Algorithm**
- 
