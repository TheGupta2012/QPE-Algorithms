from qiskit import QuantumCircuit, execute, transpile, Aer 
from qiskit.extensions import UnitaryGate,Initialize
from qiskit.tools.visualization import plot_histogram 
import numpy as np 
from time import sleep 
from qiskit.tools.monitor import job_monitor 
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector 
import sys 
from scipy.stats import unitary_group 
import matplotlib.pyplot as plt 


class SPEA():
    def __init__(self,unitary,resolution = 50, error = 3, max_iters = 20):
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
    
    def get_standard_cost(self,angles,state,backend,shots):
        '''Given an initial state and a set of angles,
          return the best cost and the associated angle
          state is a normalized state in ndarray form'''
        result = {'cost' : -1, 'theta' : -1}
        # all theta values are iterated over for the same state
        phi = Initialize(state)
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
                
        #execute only once...
        counts = execute(circuits,backend = backend,shots = shots).result().get_counts()
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
    def get_alternate_cost(self,angles,state,backend,shots):
        '''Given an initial state and a set of angles,
          return the best cost and the associated angle
          state is a normalized state in ndarray form'''
        result = {'cost' : -1, 'theta' : -1}
        # all theta values are iterated over for the same state
        phi = Initialize(state)
        
        # run the circuit once
        qc = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        # initialize the circuit 
        qc = qc.compose(phi, qubits = list(range(1,int(np.log2(self.dims))+1)))
        # add hadamard
        qc.h(0)
        # add unitary which produces a phase kickback on control qubit
        qc = qc.compose(self.c_unitary_gate,qubits = range(1+int(np.log2(self.dims))))
        # add hadamard 
        qc.h(0)
        # measure 
        qc.measure([0],[0])
        
                
        #execute only once...
        counts = execute(qc,backend = backend,shots = shots).result().get_counts()
        
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
        phi = Initialize(state) 
        # run the circuit once again
        qc = QuantumCircuit(1 + int(np.log2(self.dims)), 1)
        # initialize the circuit 
        qc = qc.compose(phi, qubits = list(range(1,int(np.log2(self.dims))+1)))
        # add hadamard
        qc.h(0)
        # add unitary which produces a phase kickback on control qubit
        qc = qc.compose(self.c_unitary_gate,qubits = range(1+int(np.log2(self.dims))))
        # add inv rotation 
        qc.p(-2*np.pi*(result['theta']),0)
        # add hadamard 
        qc.h(0)
        # measure 
        qc.measure([0],[0])
        
        counts = execute(qc,backend = backend,shots = shots).result().get_counts()
        try:
            result['cost'] = counts['0']/shots 
        except: 
            result['cost'] = 0 
        # no 0 counts present
        
        # return the result
        return result 
    
    
    def get_eigen_pair(self,backend,algo = 'alternate',progress = False,basis = None, basis_ind = None, randomize = True, target_cost = None, shots = 512):
        '''Finding the eigenstate pair for the unitary'''
         #handle algorithm...
        if not isinstance(algo,str):
            raise TypeError("Algorithm must be mentioned as a string from the values {alternate,standard}")
        elif algo not in ['alternate','standard']:
            raise ValueError("Algorithm must be specified as 'alternate' or 'standard' ")
    
        if not isinstance(progress,bool):
            raise TypeError("Progress must be a boolean variable")
        if not isinstance(randomize,bool):
            raise Exception("Randomize must be a boolean variable")
        
        if target_cost is not None:
            if not isinstance(target_cost,float):
                raise TypeError("Target cost must be a float")
            if (target_cost <= 0 or target_cost >= 1):
                raise ValueError("Target cost must be a float value between 0 and 1")
        
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
        if target_cost == None:
            precision = 1/10**self.error 
        else:
            precision = 1 - target_cost 
            
        samples = self.resolution 
        
        # initialization of range 
        left,right = 0,1
        # generate the angles
        angles = np.linspace(left,right,samples)

        # iterate once 
        if algo == 'alternate':
            result = self.get_alternate_cost(angles,phi,backend,shots)
        else:
            result = self.get_standard_cost(angles,phi,backend,shots)
        # get initial estimates 
        cost = result['cost']
        theta_max = result['theta']
        
        # the range upto which theta extends iin each iteration 
        angle_range = 0.5
        # a parameter 
        a = 1 
        # start algorithm        
        iters = 0 
        best_phi = phi 
        found = True

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
                    res = self.get_alternate_cost(angles,curr_phi,backend,shots)
                else:
                    res = self.get_standard_cost(angles,curr_phi,backend,shots)
                curr_cost = res['cost']
                curr_theta = res['theta']
                
                # at this point I have the best Cost for the state PHI and the 

               
                if curr_cost > cost:
                    theta_max = float(curr_theta) 
                    cost = float(curr_cost) 
                    best_phi = curr_phi
                    found = True
                if progress:
                    sys.stdout.write('\r')
                    sys.stdout.write("%f %%completed" % (100*(i+1)/(2*self.dims)))
                    sys.stdout.flush()
                    sleep(0.2)
                
            # iteration completes
            
            if found == False:
                # phi was not updated , change a 
                a = a/2
                if progress:
                    print("\nNo change, updating a...")
            else:
                angle_range /= 2 # updated phi and thus theta too -> refine theta range
            
            iters+=1 
            if progress:
                print("\nCOST :",cost)
                print("THETA :",theta_max)
            
            if iters >= self.iterations:
                print("Maximum iterations reached for the estimation.\nTerminating algorithm...")
                break
            # add the warning that iters maxed out 
        
        # add cost, eigenvector and theta to the dict 
        results['cost'] = cost 
        results['theta'] = theta_max 
        results['state'] = best_phi 
        return results
            