from qiskit import QuantumCircuit, execute, transpile, Aer 
from qiskit.extensions import UnitaryGate,Initialize
from qiskit.quantum_info import Statevector 
from qiskit.tools.monitor import job_monitor 
from qiskit.compiler import assemble 
from qiskit.tools.visualization import plot_bloch_vector
from qiskit.tools.visualization import plot_histogram,plot_bloch_multivector  
from qiskit.providers.ibmq.managed import IBMQJobManager
import numpy as np 
from time import sleep 
import sys 
from scipy.stats import unitary_group 
import matplotlib.pyplot as plt 


class global_max_SPEA():
    '''
    This is a class which implements the Statistical Phase Estimation algorithm.
    The global max SPEA looks to update the cost returned at the end of one itertation
    as compared to the original normal SPEA. This allows an advantage of the bundling 
    up of circuits and thus helps to speed up the quantum execution time.
    
    paper1 - https://arxiv.org/pdf/2104.10285.pdf
    paper2 - https://arxiv.org/pdf/1906.11401.pdf(discussion)
    
    Attributes :
        resolution(int) : the number of intervals the angle range has to be divided into
        dims(int) : dimensions of the passed unitary matrix
        c_unitary_gate(Unitary Gate) : the controlled unitary gate in our algorithm
        error(int) : the number of places after decimal, upto which the cost of the algorithm
                     would be estimated
        itrations(int) : the maximum iterations after which the algorithm ends the computation
        basis(list of np.ndarray) : the basis generated at the start of the algorithm
        unitary_circuit(QuantumCircuit): a pre-transpiled QuantumCircuit which is applied during
                            the algorithm
    
    Methods :
        __get_basis_vectors(randomize) : Get the d dimensional basis for the initializtion of the algorithm
        
        __get_unitary_circuit(backend) : Get the pre-transpiled circuit for the unitary matrix
        
        __get_alternate_cost(angle,state,backend,shots) : Get the cost through the alternate method specified in the algorithm
        
        __get_standard_cost(angle,state,backend,shots) : Get the cost through the standard method specified in the algorithm
        
        __get_circuit(state,angle,backend,shots) : Get the completed circuit used inside the algorithm to estimate the phase
        
        get_eigen_pair(backend, algo='alternate', theta_left,theta_right,progress, randomize, target_cost, basis, basis_ind,shots) :
                                            Get the eigenstate and eigenphase phase pair for the unitary matrix
    
    
    '''
    
    def __init__(self,unitary,resolution = 100, error = 3, max_iters = 20):
        
        # handle resolution 
        if not isinstance(resolution,int):
            raise TypeError("Please enter the number of intervals as an integer value")  
        if resolution < 10 or resolution > 1e6:
            raise ValueError("Resolution needs to be atleast 0.1 and greater than 0.000001")
        
        self.resolution = resolution 
        
        # handle unitary
        if not isinstance(unitary, np.ndarray) and not isinstance(unitary, QuantumCircuit)\
                and not isinstance(unitary, UnitaryGate):
            raise TypeError("A numpy array or Quantum Circuit or UnitaryGate needs to be passed as the unitary matrix")

        # convert circuit to numpy array for uniformity 
        if isinstance(unitary, UnitaryGate):
            U = unitary.to_matrix()
        else: # both QC and ndarray type 
            U = unitary
        
        # note - the unitary here is not just a single qubit unitary 
        if isinstance(U,np.ndarray):
            self.dims = U.shape[0]
        else:
            self.dims = 2**(U.num_qubits)
        
        
        if isinstance(U,np.ndarray):
            self.c_unitary_gate = UnitaryGate(data = U).control(num_ctrl_qubits = 1,label = 'CU',ctrl_state = '1')
        else:
            self.c_unitary_gate = U.control(num_ctrl_qubits = 1,label = 'CU',ctrl_state = '1')

        # handle error 
        if not isinstance(error,int):
            raise TypeError("The allowable error should be provided as an int. Interpreted as 10**(-error)")
        if error <= 0:
            raise ValueError("The error threshold must be finite and greater than 0.")
            
        self.error = error 
        
        # handle max_iters 
        if not isinstance(max_iters,int):
            raise TypeError("Max iterations must be of integer type")
        if max_iters <= 0 and max_iters > 1e5:
            raise ValueError("Max iterations should be atleast 1 and less than 1e5")
        
        self.iterations = max_iters 
        self.basis = []
    
    def __get_basis_vectors(self,randomize = True):
        ''' Get the d dimensional basis for the unitary provided
         Args : randomize (bool) : whether to pick a random basis or
                not 
         Returns:
             a list of np.ndarrays which are used as the basis 
             vectors '''
        if randomize == True:
            UR = unitary_group.rvs(self.dims)
        else:
            UR = np.identity(self.dims)

        basis = []
        for k in UR:
            basis.append(np.array(k,dtype = complex))
        return basis 
    
    def __get_unitary_circuit(self, backend):
        '''Return the pretranspiled circuit
        Args:
            backend : the IBMQBackend on which we want to transpile. 
                     If None, Default : 'qasm_simulator' 
        Returns: QuantumCircuit containing the transpiled circuit
                 for the controlled unitary
        '''
        if backend is None:
            backend = Aer.get_backend('qasm_simulator')
        
        qc = QuantumCircuit(1 + int(np.log2(self.dims)))

        # make the circuit
        qc.h(0)
        qc = qc.compose(self.c_unitary_gate, qubits=range(
            1+int(np.log2(self.dims))))
        
        qc.barrier()
        qc = transpile(qc,backend=backend,optimization_level = 3)
        
        return qc
    
    def __get_circuit(self, state, backend, shots, angle=None):
        '''Given an initial state ,
          return the assembled and transpiled 
          circuit that is generated with 
          inverse rotation 
          
          Args:
              state(np.ndarray) : The eigenvector guess state for the initialization
              backend(IBMQBackend) : the backend on which this circuit is going to be 
                                  executed
              shots(int) : the number of shots in our experiments
              angle(float) : whether the returned circuit contains an inverse rotation 
                            gate or not. If angle is None, no rotation gate attached
                            Else, cp(angle) is attached on control qubit 0
                            
          Returns:
              QuantumCircuit which is pre-transpiled according to the backend provided
          '''
        # all theta values are iterated over for the same state
        phi = Initialize(state)

        qc1 = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        # initialize the circuit
        qc1 = qc1.compose(phi, qubits=list(
            range(1, int(np.log2(self.dims))+1)))
        qc1.barrier()
        qc1 = transpile(qc1, backend=backend,optimization_level=1)

        # get the circuit2
        qc2 = self.unitary_circuit

        qc3 = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        if angle is not None:
            # add inverse rotation on the first qubit
            qc3.p(-2*np.pi*angle, 0)
        # add hadamard
        qc3.h(0)
        qc3 = transpile(qc3, backend=backend,optimization_level=1)

        # make final circuit
        qc = qc1 + qc2 + qc3

        # measure
        qc.measure([0], [0])
        
#         qc = assemble(qc,shots = shots) 
        return qc
    
    def __get_standard_cost(self,angles,state,backend,shots):
        '''Given an initial state and a set of angles,
          return the best cost and the associated angle.
          Implements the standard method as specified in the paper.
          
          Args : 
              angles(np.ndarray) : the set of angles on which we execute
                                  the circuits
              state(np.ndarray) : the initialization state provided
              backend(IBMQBackend): the backend on which this circuit
                                 needs to be executed
              shots(int) : the number of shots used to execute this circuit
          
          Returns :
              result(dict) : {result : (float), theta : (float)}
                              result - the best cost given this set of angles
                              theta - the best theta value amongst this set 
                                      of angles'''
        result = {'cost' : -1, 'theta' : -1}
        # all theta values are iterated over for the same state
        circuits = []
        
        for theta in angles:
            qc = self.__get_circuit(state,backend,shots,theta)
            circuits.append(qc)
                
        #execute only once...
        counts = backend.run(circuits, shots=shots).result().get_counts()
        # get the cost for this theta 
        for k,theta in zip(counts,angles):
            # for all experiments you ran 
            try:
                C_val = (k['0'])/shots
            except:
                C_val = 0 

            if C_val > result['cost']:
                # means this is a better theta value  
                result['theta'] = theta 
                result['cost'] = C_val 
        return result 
        
    def __get_alternate_cost(self,angles,state,backend,shots):
        '''Given an initial state and a set of angles,
          return the best cost and the associated angle.
          Implements the alternate method as specified in the paper1
          and discussion of paper2.
          
          Args : 
              angles(np.ndarray) : the set of angles on which we execute
                                  the circuits
              state(np.ndarray) : the initialization state provided
              backend(IBMQBackend): the backend on which this circuit
                                 needs to be executed
              shots(int) : the number of shots used to execute this circuit
          
          Returns :
              result(dict) : {result : (float), theta : (float)}
                              result - the best cost given this set of angles
                              theta - the best theta value amongst this set 
                                      of angles'''
        result = {'cost' : -1, 'theta' : -1}
        # all theta values are iterated over for the same state
        # run the circuit once
        qc = self.__get_circuit(state,backend,shots)
        
        #execute only once...
        counts = backend.run(qc,shots = shots).result().get_counts()
        
        #generate experimental probabilities
        try:
            p0 = counts['0']/shots 
        except:
            p0 = 0 
        try:
            p1 = counts['1']/shots
        except:
            p1 = 0
        
        # now, find the best theta as specified by the 
        # alternate method classically
        min_s = 1e5 
        for theta in angles:
            # generate theoretical probabilities
            c0 = (np.cos(np.pi*theta))**2 
            c1 = (np.sin(np.pi*theta))**2 
            
            #generate s value 
            s = (p0-c0)**2 + (p1-c1)**2 
            if s < min_s:
                result['theta'] = theta 
                min_s = s 
                
                
        # now , we have the best theta stored in phi 
        # run circuit once again to get the value of C* 
        qc = self.__get_circuit(state, backend, shots, result['theta'])
        counts = backend.run(qc, shots=shots).result().get_counts()
        
        try:
            result['cost'] = counts['0']/shots 
        except: 
            result['cost'] = 0 
        # no 0 counts present
        
        # return the result
        return result 
    
    
    def get_eigen_pair(self,backend,algo = 'alternate', theta_left = 0,theta_right = 1,progress = False,basis = None,basis_ind = None, randomize = True, target_cost = None,shots = 512):

        '''Finding the eigenstate pair for the unitary
        Args : 
            backend(IBMQBackend) : the backend on which the circuit needs to be executed
            algo(str) : ['alternate','standard'] the algorithm to use as specified in the 
                        paper1 section 3.
            theta_left(float): the left bound for the search of eigenvalue. Default : 0
            theta_right(float): the right bound for the search of eigenvalue. Default : 1
            progress(bool) : Whether to show the progress as the algorithm runs
            randomize(bool): Whether to choose random initialization of basis states or not
                             If False, computational basis is chosen.
            target_cost(float) : the min cost required to be achieved by the algorithm
            basis(list of np.ndarray) : The basis to be used in the algorithm. 
                                Note, if basis is specified, randomize value is ignored
            basis_ind(int) : the index of the basis vector to be used as the initial state 
                             vector
            
        Returns :
            result(dict) : {cost :(float), theta :(float), state : (np.ndarray)
                         cost - the cost with which the algorithm terminates
                         theta - the eigenvalue estimated by SPEA
                         state - the eigenvector estimated by SPEA   
        
        Example Usage found in notebook Statistical QPE-Changed Algorithm.ipynb'''
        self.unitary_circuit = self.__get_unitary_circuit(backend)
        
        if(theta_left > theta_right):
            raise ValueError("Left bound for theta should be smaller than the right bound")
        elif (theta_left<0) or (theta_right>1):
            raise ValueError("Bounds of theta are [0,1].")
        
        if not isinstance(algo,str):
            raise TypeError("Algorithm must be mentioned as a string from the values {alternate,standard}")
        elif algo not in ['alternate','standard']:
            raise ValueError("Algorithm must be specified as 'alternate' or 'standard' ")
        
        if target_cost is not None:
            if not isinstance(target_cost,float):
                raise TypeError("Target cost must be a float")
            if (target_cost <= 0 or target_cost >= 1):
                raise ValueError("Target cost must be a float value between 0 and 1")
        
        # handle progress...
        if not isinstance(progress,bool):
            raise TypeError("Progress must be a boolean variable")
        if not isinstance(randomize,bool):
            raise Exception("Randomize must be a boolean variable")
        
        results = dict() 
        
        # first initialize the state phi 
        if basis is None:
            self.basis = self.__get_basis_vectors(randomize)
        else:
            # is basis is specified, given as array of vectors...
            self.basis = basis 
            
        # choose a random index 
        if basis_ind is None:
            ind = np.random.choice(self.dims) 
        else:
            # choose the index given in that basis
            ind = basis_ind
            
        phi = self.basis[ind]
        # doing the method 1 of our algorithm 
        # define resolution of angles and precision 
        if target_cost == None:
            precision = 1/10**self.error 
        else:
            precision = 1 - target_cost 
            
        samples = self.resolution 
        
        # initialization of range 
        left,right = theta_left,theta_right
        
        # generate the angles
        angles = np.linspace(left,right,samples)

        # iterate once 
        if algo == 'alternate':
            result = self.__get_alternate_cost(angles,phi,backend,shots)
        else:
            result = self.__get_standard_cost(angles,phi,backend,shots)
        # get initial estimates 
        cost = result['cost']
        theta_max = result['theta']
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
            right = min(theta_right,theta_max + angle_range/2)
            left = max(theta_left,theta_max - angle_range/2)
            if progress:
                print("Right :",right) 
                print("Left :",left)
            # generate the angles only if the theta has been updated 
            if found == True: 
                angles = np.linspace(left,right,samples)
            
            found = False # for this iteration 
            if progress:
                print("ITERATION NUMBER",iters+1,"...")
            
            # generate a cost dict for each of the iterations 
            
            thetas, costs, states = [],[],[] 
            
            for i in range((2*self.dims)):
                # everyone is supplied with the same range of theta in one iteration 
                #define z
                if i < self.dims:
                    z = 1 
                else:
                    z = 1j 
                    
                # alter and normalise phi 
                curr_phi = best_phi + z*a*(1 - cost)*self.basis[i % self.dims]
                curr_phi = curr_phi / np.linalg.norm(curr_phi)
                
                # iterate (angles would be same until theta is changed)
                if algo == 'alternate':
                    res = self.__get_alternate_cost(angles,curr_phi,backend,shots)
                else:
                    res = self.__get_standard_cost(angles,curr_phi,backend,shots)
                curr_cost = res['cost']
                curr_theta = res['theta']
                
                # append these parameters 
                
                # bundle the circuits together ...
                
                if curr_cost > cost: # then only add this cost in the cost and states list 
                    thetas.append(float(curr_theta))
                    costs.append(float(curr_cost))
                    states.append(curr_phi)
                    found = True
                    
                    # now each iteration would see the same state as the best phi 
                    # is updated once at the end of the iteration 
                    
                    # also, the cost is also updated only once at the end of the iteration
    
                if progress:
                    sys.stdout.write('\r')
                    sys.stdout.write("%f %%completed" % (100*(i+1)/(2*self.dims)))
                    sys.stdout.flush()
                
            # 1 iteration completes
            
            if found == False:
                # phi was not updated , change a 
                a = a/2
                if progress:
                    print("\nNo change, updating a...")
            else:
                # if found is actually true, then only update 
                
                # O(n) , would update this though 
                index = np.argmax(costs)
                # update the parameters of the model 
                cost = costs[index]
                theta_max = thetas[index]
                best_phi = states[index]
                if progress:
                    print("Best Phi is :",best_phi)
                    print("Theta estimate :",theta_max)
                    print("Current cost :",cost) 
                angle_range /= 2 # updated phi and thus theta too -> refine theta range
            
            # update the iterations 
            iters+=1 
            if progress:
                print("\nCOST :",cost)
                print("THETA :",theta_max)
            
            if iters >= self.iterations:
                print("Maximum iterations reached for the estimation.\nTerminating algorithm...")
                break 
        
        # add cost, eigenvector and theta to the dict 
        results['cost'] = cost 
        results['theta'] = theta_max 
        results['state'] = best_phi 
        return results


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


class bundled_changed_SPEA():
    def __init__(self,unitary,resolution = 100, error = 3, max_iters = 20):
        
        # handle resolution 
        if not isinstance(resolution,int):
            raise TypeError("Please enter the number of intervals as an integer value")  
        if resolution < 10 or resolution > 1e6:
            raise ValueError("Resolution needs to be atleast 0.1 and greater than 0.000001")
        
        self.resolution = resolution 
        
        # handle unitary
        if not isinstance(unitary, np.ndarray) and not isinstance(unitary, QuantumCircuit)\
                and not isinstance(unitary, UnitaryGate):
            raise TypeError("A numpy array or Quantum Circuit or UnitaryGate needs to be passed as the unitary matrix")

        # convert circuit to numpy array for uniformity 
        if isinstance(unitary, UnitaryGate):
            U = unitary.to_matrix()
        else: # both QC and ndarray type 
            U = unitary
        
        # note - the unitary here is not just a single qubit unitary 
        if isinstance(U,np.ndarray):
            self.dims = U.shape[0]
        else:
            self.dims = 2**(U.num_qubits)
        
        
        if isinstance(U,np.ndarray):
            self.c_unitary_gate = UnitaryGate(data = U).control(num_ctrl_qubits = 1,label = 'CU',ctrl_state = '1')
        else:
            self.c_unitary_gate = U.control(num_ctrl_qubits = 1,label = 'CU',ctrl_state = '1')

        # handle error 
        if not isinstance(error,int):
            raise TypeError("The allowable error should be provided as an int. Interpreted as 10**(-error)")
        if error <= 0:
            raise ValueError("The error threshold must be finite and greater than 0.")
            
        self.error = error 
        
        # handle max_iters 
        if not isinstance(max_iters,int):
            raise TypeError("Max iterations must be of integer type")
        if max_iters <= 0 and max_iters > 1e5:
            raise ValueError("Max iterations should be atleast 1 and less than 1e5")
        
        self.iterations = max_iters 
        self.basis = []
    
    def get_basis_vectors(self,randomize = True):
        # get the d dimensional basis for the unitary provided 
        if randomize == True:
            UR = unitary_group.rvs(self.dims)
        else:
            UR = np.identity(self.dims)

        basis = []
        for k in UR:
            basis.append(np.array(k,dtype = complex))
        return basis 
    
    def get_circuits(self,angles,state):
        '''Given an initial state and a set of angles,
          return the circuits that are generated with 
          those angles'''
        result = {'cost' : -1, 'theta' : -1}
        # all theta values are iterated over for the same state
        phi = Initialize(state)
        shots = 512
        circuits = []
        
        for theta in angles:
            qc = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
            # initialize the circuit 
            qc = qc.compose(phi, qubits = list(range(1,int(np.log2(self.dims))+1)))
            # add hadamard
            qc.h(0)
            # add unitary which produces a phase kickback on control qubit
            qc = qc.compose(self.c_unitary_gate,qubits = range(1+int(np.log2(self.dims))))
            # add the inv rotation 
            qc.p(-2*np.pi*theta,0)
            # add hadamard 
            qc.h(0)
            # measure 
            qc.measure([0],[0])
            #generate all the circuits...
            circuits.append(qc)
              
        return circuits 
    
    def get_cost(self,angles,counts,shots):
        '''Generate the best cost and theta pair 
        for the particular state '''
        
        result = {'cost':-1, 'theta': -1}
        # get the cost for this theta 
        
        for k,theta in zip(counts,angles):
            # for all experiments you ran 
            try:
                C_val = (k['0'])/shots
            except:
                C_val = 0 

            if C_val > result['cost']:
                # means this is a better theta value  
                result['theta'] = theta 
                result['cost'] = C_val 
            
        return result 
    
    def get_eigen_pair(self,backend,progress = False,randomize = True):
        '''Finding the eigenstate pair for the unitary'''
            
        if not isinstance(progress,bool):
            raise TypeError("Progress must be a boolean variable")
        
        if not isinstance(randomize,bool):
            raise Exception("Randomize must be a boolean variable")
        
        results = dict() 
        
        # first initialize the state phi 
        self.basis = self.get_basis_vectors(randomize)
        
        # choose a random index 
        ind = np.random.choice(self.dims) 
        phi = self.basis[ind]
        
        # doing the method 1 of our algorithm 
        # define resolution of angles and precision 
        precision = 1/10**self.error 
        samples = self.resolution 
        
        # initialization of range 
        left,right = 0,1
        shots = 512
        
        # generate the angles
        angles = np.linspace(left,right,samples)

        # First execution can be done without JobManager also...
        circs = self.get_circuits(angles,phi)
        job = execute(circs,backend = backend, shots = shots)
        counts = job.result().get_counts()
        result = self.get_cost(angles,counts,shots)
        
        # get initial estimates 
        cost = result['cost']
        theta_max = result['theta']
        best_phi = phi 

        # the range upto which theta extends iin each iteration 
        angle_range = 0.5
        # a parameter 
        a = 1 
        # start algorithm        
        iters = 0 
        found = True
        plus = (1/np.sqrt(2))*np.array([1,1])
        minus = (1/np.sqrt(2))*np.array([1,-1])
        
        #define IBMQManager instance
        manager = IBMQJobManager()
        
        while 1 - cost >= precision:
            # get angles, note if theta didn't change, then we need to 
            # again generate the same range again 
            right = min(1,theta_max + angle_range/2)
            left = max(0,theta_max - angle_range/2)
            if progress:
                print("Right :",right) 
                print("Left :",left)
            # generate the angles only if the theta has been updated 
            if found == True: 
                angles = np.linspace(left,right,samples)
            
            found = False # for this iteration 
            if progress:
                print("ITERATION NUMBER",iters+1,"...")
            
            # generate a cost dict for each of the iterations 
            # final result lists 
            thetas, costs, states = [],[],[] 
            
            # circuit list 
            circuits = []
            
            #list to store intermediate states
            phis = []
            
            # 1. Circuit generation loop
            for i in range((2*self.dims)):
                # everyone is supplied with the same range of theta in one iteration 
                #define z
                # make a list of the circuits
                if i < self.dims:
                    z = 1 
                else:
                    z = 1j 
                    
                # alter and normalise phi 
                curr_phi = best_phi + z*a*(1 - cost)*self.basis[i % self.dims]
                curr_phi = curr_phi / np.linalg.norm(curr_phi)
                phis.append(curr_phi)
                
                # bundle the circuits together ...   
                circs = self.get_circuits(angles,curr_phi)
                circuits = circuits + circs
                
                # now each iteration would see the same state as the best phi 
                # is updated once at the end of the iteration 
                    
                # also, the cost is also updated only once at the end of the iteration
            
            
            # 2. run the generated circuits 
            if progress:
                print("Transpiling circuits...")
            circuits = transpile(circuits=circuits,backend=backend)
            job_set = manager.run(circuits,backend = backend,name = 'Job_set'+str(iters),shots=shots)
            if progress:
                print("Transpilation Done!\nJob sent...")
            job_result = job_set.results()        
            
            # now get the circuits in chunks of resolution each 
            if progress:
                print("Job has returned")
                
                
            # 3. Result generation loop
            for i in range((2*self.dims)):
                # get the results of this basis state
                # it will have resolution number of circuits...
                counts = []
                for j in range(i*self.resolution, (i+1)*self.resolution):
                    # in this you'll get the counts 
                    counts.append(job_result.get_counts(j))
                
                result = self.get_cost(counts,angles,shots)
                
                # get the estimates for this basis 
                curr_theta = result['theta']
                curr_cost = result['cost']
                curr_phi = phis[i] # the result was generated pertaining to this phi 
                    
                if curr_cost > cost: # then only add this cost in the cost and states list 
                    thetas.append(float(curr_theta))
                    costs.append(float(curr_cost))
                    states.append(curr_phi)
                    found = True
                
                if progress:
                    sys.stdout.write('\r')
                    sys.stdout.write("%f %%completed" % (100*(i+1)/(2*self.dims)))
                    sys.stdout.flush()
                
                
            if found == False:
                # phi was not updated , change a 
                a = a/2
                if progress:
                    print("\nNo change, updating a...")
            else:
                # if found is actually true, then only update 
                
                # O(n) , would update this though 
                index = np.argmax(costs)
                # update the parameters of the model 
                cost = costs[index]
                theta_max = thetas[index]
                best_phi = states[index]
                if progress:
                    print("Best Phi is :",best_phi)
                    print("Theta estimate :",theta_max)
                    print("Current cost :",cost) 
                angle_range /= 2 # updated phi and thus theta too -> refine theta range
            
            # update the iterations 
            iters+=1 
            if progress:
                print("\nCOST :",cost)
                print("THETA :",theta_max)
            
            if iters >= self.iterations:
                print("Maximum iterations reached for the estimation.\nTerminating algorithm...")
                break 
      # add cost, eigenvector and theta to the dict 
        results['cost'] = cost 
        results['theta'] = theta_max 
        results['state'] = best_phi 
        
        return results
            
