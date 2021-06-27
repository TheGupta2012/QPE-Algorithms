from qiskit import QuantumCircuit, execute, transpile, Aer
from qiskit.extensions import UnitaryGate, Initialize
from qiskit.quantum_info import Statevector
from qiskit.compiler import assemble 
from qiskit.tools.visualization import plot_bloch_vector
from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
from qiskit.providers.ibmq.managed import IBMQJobManager
from time import sleep
import sys
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

class bundled_SPEA_alternate():
    def __init__(self, unitary, resolution=100, error=3, max_iters=20):

        # handle resolution
        if not isinstance(resolution, int):
            raise TypeError(
                "Please enter the number of intervals as an integer value")
        if resolution < 10 or resolution > 1e6:
            raise ValueError(
                "Resolution needs to be atleast 0.1 and greater than 0.000001")

        self.resolution = resolution

        # handle unitary
        if not isinstance(unitary, np.ndarray) and not isinstance(unitary, QuantumCircuit)\
                and not isinstance(unitary, UnitaryGate):
            raise TypeError(
                "A numpy array or Quantum Circuit or UnitaryGate needs to be passed as the unitary matrix")

        # convert circuit to numpy array for uniformity
        if isinstance(unitary, UnitaryGate):
            U = unitary.to_matrix()
        else:  # both QC and ndarray type
            U = unitary

        # note - the unitary here is not just a single qubit unitary
        if isinstance(U, np.ndarray):
            self.dims = U.shape[0]
        else:
            self.dims = 2**(U.num_qubits)

        if isinstance(U, np.ndarray):
            self.c_unitary_gate = UnitaryGate(data=U).control(
                num_ctrl_qubits=1, label='CU', ctrl_state='1')
        else:
            self.c_unitary_gate = U.control(
                num_ctrl_qubits=1, label='CU', ctrl_state='1')

        # handle error
        if not isinstance(error, int):
            raise TypeError(
                "The allowable error should be provided as an int. Interpreted as 10**(-error)")
        if error <= 0:
            raise ValueError(
                "The error threshold must be finite and greater than 0.")

        self.error = error

        # handle max_iters
        if not isinstance(max_iters, int):
            raise TypeError("Max iterations must be of integer type")
        if max_iters <= 0 and max_iters > 1e5:
            raise ValueError(
                "Max iterations should be atleast 1 and less than 1e5")

        self.iterations = max_iters
        self.basis = []

    def get_basis_vectors(self, randomize=True):
        # get the d dimensional basis for the unitary provided
        if randomize == True:
            UR = unitary_group.rvs(self.dims)
        else:
            UR = np.identity(self.dims)

        basis = []
        for k in UR:
            basis.append(np.array(k, dtype=complex))
        return basis

    def get_unitary_circuit(self, backend):
        '''Return the pretranspiled circuit '''
        if backend is None:
            backend = Aer.get_backend('qasm_simulator')

        qc = QuantumCircuit(1 + int(np.log2(self.dims)))

        # make the circuit
        qc.h(0)
        qc = qc.compose(self.c_unitary_gate, qubits=range(
            1+int(np.log2(self.dims))))

        qc.barrier()

        qc = transpile(qc, backend=backend, optimization_level=3)
        return qc

    def get_circuit(self, state, backend, shots, angle=None):
        '''Given an initial state ,
          return the circuit that is generated with 
          inverse rotation as 0.'''
        # all theta values are iterated over for the same state
        phi = Initialize(state)
        shots = 512

        qc1 = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        # initialize the circuit
        qc1 = qc1.compose(phi, qubits=list(
            range(1, int(np.log2(self.dims))+1)))
        qc1 = transpile(qc1, backend=backend)

        # get the circuit2
        qc2 = self.unitary_circuit

        qc3 = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        if angle is not None:
            # add inverse rotation on the first qubit
            qc3.p(-2*np.pi*angle, 0)
        # add hadamard
        qc3.h(0)
        qc3 = transpile(qc3, backend=backend)

        # make final circuit
        qc = qc1 + qc2 + qc3
#         qc = assemble(qc,shots = shots)
        # measure
        qc.measure([0], [0])
        return qc

    def get_optimal_angle(self, p0, p1, angles):
        '''Return the theta value which minimizes 
        the S metric '''
        min_s = 1e5
        best_theta = -1
        for theta in angles:
            c0 = (np.cos(np.pi*theta))**2
            c1 = (np.sin(np.pi*theta))**2
            # generate the metric ...
            s = (p0-c0)**2 + (p1-c1)**2
            if s < min_s:
                s = min_s
                best_theta = theta

        return best_theta

    def execute_job(self, progress, iteration, backend, shots, circuits):
        '''Send a job to the backend using IBMQ Job Manager'''
        # define IBMQManager instance
        manager = IBMQJobManager()
        # first run the generated circuits
        if progress:
            print("Transpiling circuits...")
        
        # get the job runner instance
        job_set = manager.run(circuits, backend=backend,
                              name='Job_set '+str(iteration), shots=shots)
        if progress:
            print("Transpilation Done!\nJob sent...")

        # send and get job
        job_result = job_set.results()

        if progress:
            print("Job has returned")

        # return result
        return job_result

    def get_eigen_pair(self, backend, theta_left = 0, theta_right = 1,progress=False, randomize=True,
                       basis=None, basis_ind=None,target_cost = None,shots = 512):
        '''Finding the eigenstate pair for the unitary'''
        self.unitary_circuit = self.get_unitary_circuit(backend)
        
        if(theta_left > theta_right):
            raise ValueError("Left bound for theta should be smaller than the right bound")
        elif (theta_left<0) or (theta_right>1):
            raise ValueError("Bounds of theta are [0,1].")
        
        
        
        if not isinstance(progress, bool):
            raise TypeError("Progress must be a boolean variable")

        if not isinstance(randomize, bool):
            raise Exception("Randomize must be a boolean variable")

        results = dict()

        # first initialize the state phi
        if basis is None:
            self.basis = self.get_basis_vectors(randomize)
        else:
            self.basis = basis

        # choose a random index
        if basis_ind is None:
            ind = np.random.choice(self.dims)
        else:
            ind = basis_ind

        phi = self.basis[ind]

        # doing the method 1 of our algorithm
        # define resolution of angles and precision
        precision = 1/10**self.error
        samples = self.resolution

        # initialization of range
        left, right = theta_left, theta_right

        # generate the angles
        angles = np.linspace(left, right, samples)

        # First execution can be done without JobManager also...
        circ = self.get_circuit(phi, backend=backend,shots = shots)
        job = execute(circ, backend=backend, shots=shots)
        counts = job.result().get_counts()

        if '0' in counts:
            p0 = counts['0']
        else:
            p0 = 0
        if '1' in counts:
            p1 = counts['1']
        else:
            p1 = 0

        # experimental probabilities
        p0, p1 = p0/shots, p1/shots

        # get initial angle estimate
        theta_max = self.get_optimal_angle(p0, p1, angles)

        # get intial cost
        circ = self.get_circuit(phi, backend=backend, shots = shots, angle=theta_max)
        job = execute(circ, backend=backend, shots=shots)
        counts = job.result().get_counts()
        if '0' in counts:
            cost = counts['0']/shots
        else:
            cost = 0
        # update best phi
        best_phi = phi

        # the range upto which theta extends iin each iteration
        angle_range = (right - left)/2
        # a parameter
        a = 1
        # start algorithm
        iters = 0
        found = True

        while 1 - cost >= precision:
            # get angles, note if theta didn't change, then we need to
            # again generate the same range again
            right = min(theta_right, theta_max + angle_range/2)
            left = max(theta_left, theta_max - angle_range/2)
            if progress:
                print("Right :", right)
                print("Left :", left)
            # generate the angles only if the theta has been updated
            if found == True:
                angles = np.linspace(left, right, samples)

            found = False  # for this iteration
            if progress:
                print("ITERATION NUMBER", iters+1, "...")

            # generate a cost dict for each of the iterations
            # final result lists
            costs, states = [], []

            # circuit list
            circuits = []

            # 1. Circuit generation loop
            for i in range((2*self.dims)):
                # everyone is supplied with the same range of theta in one iteration
                # define z
                if i < self.dims:
                    z = 1
                else:
                    z = 1j

                # alter and normalise phi
                curr_phi = best_phi + z*a*(1 - cost)*self.basis[i % self.dims]
                curr_phi = curr_phi / np.linalg.norm(curr_phi)
                states.append(curr_phi)

                # bundle the circuits together ...
                circuits.append(self.get_circuit(curr_phi, backend=backend,shots = shots))

            job_result = self.execute_job(
                progress, iters, backend, shots, circuits)
            # define lists again
            thetas, circuits = [], []

            # 3. Classical work
            for i in range((2*self.dims)):
                # we have that many circuits only
                counts = job_result.get_counts(i)

                # get the experimental counts for this state
                try:
                    exp_p0 = counts['0']
                except:
                    exp_p0 = 0
                try:
                    exp_p1 = counts['1']
                except:
                    exp_p1 = 0

                # normalize
                exp_p0 = exp_p0/shots
                exp_p1 = exp_p1/shots
                theta_val = self.get_optimal_angle(exp_p0, exp_p1, angles)

                # generate the circuit and append
                circuits.append(self.get_circuit(
                    states[i], backend=backend, shots = shots, angle=theta_val))
                thetas.append(theta_val)

            # again execute these circuits

            job_result = self.execute_job(
                progress, iters, backend, shots, circuits)

            # 5. Result generation loop
            for i in range((2*self.dims)):
                # cost generation after execution ...
                counts = job_result.get_counts(i)
                if '0' in counts:
                    curr_cost = counts['0']/(shots)
                else:
                    curr_cost = 0
                if curr_cost > cost:  # then only add this cost in the cost and states list
                    cost = curr_cost
                    best_phi = states[i]
                    theta_max = thetas[i]
                    found = True

                if progress:
                    sys.stdout.write('\r')
                    sys.stdout.write("%f %%completed" %
                                     (100*(i+1)/(2*self.dims)))
                    sys.stdout.flush()

            if found == False:
                # phi was not updated , change a
                a = a/2
                if progress:
                    print("\nNo change, updating a...")
            else:
                # if found is actually true, then only print
                if progress:
                    print("Best Phi is :", best_phi)
                    print("Theta estimate :", theta_max)
                    print("Current cost :", cost)
                angle_range /= 2  # updated phi and thus theta too -> refine theta range

            # update the iterations
            iters += 1
            if progress:
                print("\nCOST :", cost)
                print("THETA :", theta_max)

            if iters >= self.iterations:
                print(
                    "Maximum iterations reached for the estimation.\nTerminating algorithm...")
                break
      # add cost, eigenvector and theta to the dict
        results['cost'] = cost
        results['theta'] = theta_max
        results['state'] = best_phi

        return results