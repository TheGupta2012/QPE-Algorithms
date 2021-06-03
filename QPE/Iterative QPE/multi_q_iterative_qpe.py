from qiskit import *
from qiskit import transpile 
from qiskit.tools.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit.tools.monitor import job_monitor 
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Gate 

class general_IQPE():
    '''Implements the iterative QPE algorithm upto n bit precision specified by user 
    
    Attributes:
        precision : int([1,...]): precision of the phase estimation ( also equal to number of qubits in scratch register)
        unitary(np.ndarray or QuantumCircuit or UnitaryGate): unitary operator for which QPE is being applied.
        unknown(bool) : a boolean variable used to specify whether exponentiation of unitary needs to be done or not 
                        (True : no exponentiation done) 
        powers(dict): contains the powers of the unitary matrix
                      key : exponent, value : U^(exponent)
        controls(dict): contains equivalent controlled U gates to be applied
                      key : iteration number, value : controlled U gate
                      
    Methods : 
        get_powers(unitary,n) : generates powers of U from U^1 -> U^(2^(n-1)) and stores power in dict 
        get_controls() : generates the U^(2^j) control gates for faster simulation of the QPE algorithm
        get_circuit_phase() : returns a 2 tuple which contains the phase information of the 
                        unitary matrix.
    '''
    def __init__(self, precision, unitary, unknown=False):  # write docs
        '''
        Args :
            precision(int) : The precision upto which the phase needs to be estimated.
                             Interpreted as 2^(-precision).
                             eg. precision = 4 means the phase is going to be precise
                                 upto 2^(-4).
            unitary(np.ndarray or UnitaryGate or QuantumCircuit):
                            The unitary for which we want to determine the phase. Currently
                            this class supports 2 x 2 matrices or single qubit gates.
                            Shall be extended for higher order matrices.
            unknown(bool) : Whether exponentiation is to be done or not 
        Raises :
            TypeError : if precision or unitary are not of a valid type
            ValueError : if precision is not valid
            Exception : if unitary is of larger size than 2 x 2

        Examples :
            from iter_QPE import IQPE
                # passing as ndarray
                theta = 1/5
                U1 = np.ndarray([[1,0],
                                [0, np.exp(2*np.pi*1j*(theta))]])
                                
                qpe1 = IQPE(precision = 4, unitary = U1,unknown = True)

                # passing as QuantumCircuit
                U2 = QuantumCircuit(1)
                U2.rz(np.pi/7,0)
                qpe2 = IQPE(precision = 5,unitary = U2,unknown = True)

        '''
        # handle precision
        if type(precision) != int:
            raise TypeError("Precision needs to be an integer")
        elif precision < 0 or precision == 0:
            raise ValueError("Precision needs to be atleast 1")

        self.precision = precision

        # handle unitary
        if not isinstance(unitary, np.ndarray) and not isinstance(unitary, QuantumCircuit)\
                and not isinstance(unitary, UnitaryGate) and not isinstance(unitary,Gate):
            raise TypeError("A numpy array, Quantum Circuit or Gate needs to be passed as the unitary matrix")
    
        if unknown == False:
            # means matrix rep needed 
            if not isinstance(unitary,np.ndarray) and not isinstance(unitary,UnitaryGate):
                raise TypeError('''Unitary needs to be of type ndarray or Unitary Gate if optimization  
                                needs to be done''')
            if isinstance(unitary, UnitaryGate):
                U = unitary.to_matrix()
            else:
                U = unitary # already an array 
        else:
            # if it is numpy type array 
            if isinstance(unitary,np.ndarray):
                U = UnitaryGate(data = unitary)
            else:
                U = unitary 
            # here we can directly use the .control method in our circuit.
            
            
        # the unitary is an ndarray if unknown is False and 
        # the unitary is not ndarray is unknown is true 
        self.unitary = U
        self.unknown = unknown
        self.powers = {}
        
        # optimization can only be performed when we know the 
        # matrix representation 
        if unknown == False:
            self.controls = self.get_controls()

    def get_powers(self, unitary, n):
        '''This function returns the matrix U^(n) and saves 
           other smaller powers 
            
        Arguments: 
            unitary(np.ndarray): 
                The Unitary matrix which needs to be exponentitated 
            n(int): integer specifying the exponent 
            
        Raises:
            ValueError : when n is < 0 
        Returns: 
            a dictionary containing the relevant powers of the matrix U '''
        if n < 0:
            raise ValueError("Power should be atleast 0")
        if n == 1:
            self.powers[1] = unitary
            return unitary
        if n % 2 == 1:
            if n-1 not in self.powers:
                self.powers[n-1] = self.get_powers(unitary, n-1)
                
            self.powers[n] = unitary @ self.powers[n-1]
            return self.powers[n]
        else:
            if n/2 not in self.powers:
                self.powers[n/2] = self.get_powers(unitary, n/2)

            self.powers[n] = self.powers[n/2] @ self.powers[n/2]
            return self.powers[n]
    
    # get the controls, if using optimization
    def get_controls(self):
        '''Get the control gates for the circuit
         While computing exponent, we also 
         compute the smaller powers
         
        Returns: 
            controls(dict) : dictionary containing the relevant controlled unitary gates
                            key : iteration number, value : controlled U gate 
             '''
             
        n_iters = self.precision

        exp = 2**(n_iters-1)
        self.get_powers(self.unitary, exp)

        # got the powers
        controls = {}
        
        # note that iteration 0 has the highest powered matrix and
        # then it goes on to get lesser and lesser
        iterations = self.precision
        for it in range(iterations):
            mat = self.powers[exp]
            u_gate = UnitaryGate(data=mat)
            cu = u_gate.control(num_ctrl_qubits=1, label='CU', ctrl_state='1')
            controls[it] = cu
            exp /= 2

        return controls

    def get_circuit_phase(self, QC, clbits, qubits, ancilla, show=False,backend = None):
        # QC must be atleast size 2
        '''Add the experiments pararmeters .., and the shots parameter '''
        
        '''
        Returns the circuit phase as a 2-tuple phase : (binary phase,decimal phase)
        Arguments: 
            QC(QuantumCircuit) : the circuit containing the eigenvector of the unitary matrix 
            clbits(list-like) : the list of the classical bits in which the phase would be saved 
            qubits(list-like) : the indices of the qubits containing the eigenvector of unitary
            ancilla(int) : the ancilliary qubit which would be used as the control qubit   
            show(bool) : boolean to specify if circuit should be drawn or not 
            backend(IBMQBackend) : backend for running the circuit 
                    NOTE : IBMQ provider must be enabled for execution of circuits on real backends

            
        Raises: 
            ValueError : if clbits are not equal to precision or non-unique bits specified 
                         or if elements of clbits/qubits are not integer type or ancilla qubit is 
                         same as one of the eigenvector qubits
            TypeError : if qubit indices are not integral 
            Exception : if unitary has less than 2 qubits  
            
        
        Returns : 
            A 2-tuple specifying the phase of unitary matrix : (binary phase,decimal phase) 
        
        Usage Notes : 
        NOTE : The phase is assumed to be a binary fraction as 0.x1x2x2...xn where n  
               is the precision specified by the user.
               
               The least significant bit , xn, is saved in the qubit with index 
               precision-1 and the most significant bit, x1, is saved in the 
               qubit with index 0 in the phase[0] of tuple. 
               
        For example :- 
            theta = 1/5  # binary representation upto 4 bits : 0.0011
            unitary = np.ndarray([[1,0],
                                [0, np.exp(2*np.pi*1j*(theta))]])
            q = QuantumCircuit(6,4)
            q.x(4) # the eigenvector qubit

            qpe = get_circuit_phase(precision = 4,unitary = unitary,unknown = True)
            athens = provider.get_backend('ibmq_athens')
            phase = iqpe.get_circuit_phase(
            QC=q, clbits=[0, 1, 2, 3], qubits=[4], ancilla=3, show=True,backend = athens)
            
            # phase[0] would contain a 4-bit phase representation 
            # phase[1] would contain the decimal representation of the phase
        '''
        # handle qubits in circuit
        if len(QC.qubits) < 2:
            raise Exception("Quantum Circuit needs to have atleast size 2")

        # handle classical bits
        if len(clbits) != self.precision:
            raise ValueError("Exactly",self.precision,"classical bits needed for measurement")
        elif len(set(clbits)) != len(clbits):
            raise ValueError("Non-unique classical bits given for measurement")
        elif not all(isinstance(i, int) for i in clbits):
            raise ValueError("All classical indices must be integer type")

            
        # qubit and ancilla need to be integers
        if type(ancilla) is not int:
            raise TypeError("Ancilla indix need to be specified as integer")
        elif not all(isinstance(i,int) for i in qubits):
            raise TypeError("The indices containing the eigenvector must be integer type ")
        elif len(set(qubits)) != len(qubits):
            raise ValueError("Non-unique qubits given for the eigenvector")
        elif ancilla in qubits:
            raise Exception("Ancilla can not be equal to a qubit index ")
            
        res = []
        # start with the iteration
        phase = -2*np.pi
        factor = 0
        iterations = self.precision
        # generate the qubit list on which the Unitary is applied
        qargs = [ancilla]
        for q in qubits:
            qargs.append(q) 
        
        if self.unknown == True:
            # no matrix repr is available -> means .control method can be applied easily
            exponent = 2**(iterations - 1)
            CU = self.unitary.control(num_ctrl_qubits = 1,label = 'CU',ctrl_state=1)
        
        for it in range(iterations):
            # start
            QC.reset(ancilla)
            QC.h(ancilla)
            # add the inverse rotation
            inv_phase = phase*factor

            QC.p(inv_phase, ancilla)

            # add the controlled Unitary of iteration it 
            if self.unknown == False:
                QC = QC.compose(self.controls[it], qubits=qargs)
            else:
                # need to add exponential amount of matrices 
                for _ in range(int(exponent)):
                    QC = QC.compose(CU,qubits = qargs)
                exponent /= 2 
                
            # add H gate
            QC.h(ancilla)
            QC.measure(ancilla, clbits[it])

            # or, iterate in reverse manner , no of steps ->
            # clbits[it] as it is the absolute
            # classical register index
            if backend == None: # simulating 
                counts = execute(QC, backend=Aer.get_backend('qasm_simulator'), shots=1).result().get_counts()
            else:
#                 trans = transpile(QC,backend = backend,optimization_level = 3)
                job = execute(QC,shots = 1,backend = backend,job_name = "Iter "+str(it+1),
                             optimization_level = 3)
#                 print("DEPTH ",trans.depth())
                display(job_monitor(job))
                counts = job.result().get_counts()
                    
            # we will have only one single key in the dict
            key = list(counts.keys())[0][::-1]
            # try adding x based on clasical
            curr_bit = key[clbits[it]]
            res.append(int(curr_bit))
            # if bit measured is 1
            if curr_bit == '1':
                factor += 1/2  # add the phase factor

            factor = factor/2  # shift each towards one weight right
            if it+1 == iterations:
                if show == True:
                    display(QC.draw('mpl'))

        # phase has now been stored in the clbits
        # returning its binary representation

        # need to reverse as LSB is stored at the zeroth index and
        # not the last
        res = res[::-1]
        
        #find decimal phase 
        dec = 0
        weight = 1/2
        for k in res:
            dec += (weight)*k
            weight /= 2
            
        return (res,dec)