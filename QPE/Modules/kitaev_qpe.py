from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
import numpy as np
from IPython.display import display


class KQPE:
    """
    Implements the Kitaev's phase estimation algorithm where a single circuit
    is used to estimate the phase of a unitary but the measurements are
    exponential.
    Attributes :
        precision : (int)the precision upto which the phase needs to be estimated
        NOTE : precision is estimated as 2^(-precision)

        unitary (np.ndarray or QuantumCircuit or UnitaryGate) : the unitary matrix for which
                we want to find the phase, given its eigenvector
        qubits (int) : the number of qubits on which the unitary matrix acts

    Methods :
        get_phase(QC,ancilla,clbits,backend, show) : generate the resultant phase associated with the given Unitary
                      and the given eigenvector
        get_circuit(show,save_circ, circ_name ) : generate a Kitaev phase estimation circuit which can be attached
                      to the parent quantum circuit containing eigenvector of the unitary matrix
    """

    def __init__(self, unitary, precision=10):
        """
        Args :
            precision(int) : The precision upto which the phase is estimated.
                                      Interpreted as 2^(-precision).
                                     eg. precision = 4 means the phase is going to be precise
                                         upto 2^(-4).

           unitary(np.ndarray or UnitaryGate or QuantumCircuit):
                            The unitary for which we want to determine the phase.

        Raises :
            TypeError : if precision or unitary are not of a valid type
            ValueError : if precision is not valid

        Examples :
            # passing as array
            theta = 1/6
            U1 = np.ndarray([[1,0],
                            [0, np.exp(2*np.pi*1j*(theta))]])

            kqpe1 = KQPE(precision = 8, unitary = U1)

            # passing as QuantumCircuit
            U2 = QuantumCircuit(1)
            U2.rz(np.pi/7,0)

            kqpe2 = KQPE(precision = 8,unitary = U2)

        """
        # handle precision
        if not isinstance(precision, int):
            raise TypeError("Precision needs to be an integer")
        elif precision <= 0:
            raise ValueError("Precision needs to be >=0")

        self.precision = 1 / (2 ** precision)

        # handle unitary
        if unitary is None:
            raise Exception(
                "Unitary needs to be specified for the Kitaev QPE algorithm"
            )
        elif (
            not isinstance(unitary, np.ndarray)
            and not isinstance(unitary, QuantumCircuit)
            and not isinstance(unitary, UnitaryGate)
        ):
            raise TypeError(
                "A numpy array, QuantumCircuit or UnitaryGate needs to be passed as the unitary matrix"
            )

        self.unitary = unitary

        # get the number of qubits in the unitary
        if isinstance(unitary, np.ndarray):
            self.qubits = int(np.log2(unitary.shape[0]))
        else:
            self.qubits = int(unitary.num_qubits)

    def get_phase(self, QC, ancilla, clbits, backend, show=False):
        """This function is used to determine the final measured phase from the circuit
        with the specified precision.

        Args :
            QC (QuantumCircuit) : the quantum circuit on which we have attached
                            the kitaev's estimation circuit. Must contain atleast 2
                            classical bits for correct running of the algorithm
            ancilla(list-like) : the ancilla qubits to be used in the kitaev phase
                            estimation
            clbits(list-like) : the classical bits in which the measurement results
                                of the given ancilla qubits is stored
            backend(ibmq_backend or 'qasm_simulator') : the backend on which the
                            circuit is executed on
            show(bool) : boolean to specify whether the progress and the circuit
                         need to be shown

        Raises :
            Exception : If a QuantumCircuit with less than 2 classical bits or less
                        than 3 qubits is passed, if the ancilla or the clbits are
                        not unique, if more than 2 of classical or ancilla bits are
                        provided

        Returns :
            phase(tuple) : (phase_dec, phase_binary) : A 2-tuple representing
                        the calculated phase in decimal and binary upto the given
                        precision

        NOTE : for details of the math please refer to section2A of https://arxiv.org/pdf/1910.11696.pdf.

        Examples :

            U = np.array([[1, 0],
              [0, np.exp(2*np.pi*1j*(1/3))]])

            kqpe = KQPE(unitary=U, precision=16)
            kq_circ = kqpe.get_circuit(show=True, save_circ=True,
                           circ_name="KQPE_circ_1qubit.JPG")
            # defining parent quantum circuit
            q = QuantumCircuit(5, 6)

            #eigenvector of unitary
            q.x(3)

            #kitaev circuit is attached on the eigenvector and
            # two additional ancilla qubits
            q.append(kq_circ, qargs=[1, 2, 3])
            q.draw('mpl')

            # result
            phase = kqpe.get_phase(backend=Aer.get_backend(
                    'qasm_simulator'), QC=q, ancilla=[1, 2], clbits=[0, 1], show=True)

        """
        # handle circuit
        if not isinstance(QC, QuantumCircuit):
            raise TypeError(
                "A QuantumCircuit must be provided for generating the phase"
            )

        if len(QC.clbits) < 2:
            raise Exception("Atleast 2 classical bits needed for measurement")
        elif len(QC.qubits) < 3:
            raise Exception("Quantum Circuit needs to have atleast 3 qubits")

        # handle bits
        elif len(ancilla) != 2 or ancilla is None:
            raise Exception("Exactly two ancilla bits need to be specified")
        elif len(clbits) != 2 or clbits is None:
            raise Exception(
                "Exactly two classical bits need to be specified for measurement"
            )
        elif len(set(clbits)) != len(clbits) or len(set(ancilla)) != len(ancilla):
            raise Exception("Duplicate bits provided in lists")

        # find number of shots -> atleast Big-O(1/precision shots)

        shots = 10 * int(1 / self.precision)
        if show == True:
            print("Shots :", shots)

        # measure into the given bits
        QC.measure([ancilla[0], ancilla[1]], [clbits[0], clbits[1]])

        if show == True:
            display(QC.draw("mpl"))

        # execute the circuit
        result = execute(
            QC, backend=backend, shots=shots, optimization_level=3
        ).result()
        counts = result.get_counts()
        if show:
            print("Measurement results :", counts)
        if show:
            display(plot_histogram(counts))

        # now get the results
        C0, C1, S0, S1 = 0, 0, 0, 0
        first = clbits[0]
        second = clbits[1]
        for i, j in zip(list(counts.keys()), list(counts.values())):
            # get bits
            l = len(i)
            one = i[l - first - 1]
            two = i[l - second - 1]

            # First qubit 0 - C (0,theta)
            if one == "0":
                C0 += j
            # First qubit 1 - C (1,theta)
            else:
                C1 += j
            # Second qubit 0 - S (0,theta)
            if two == "0":
                S0 += j
            # Second qubit 1 - S (1,theta)
            else:
                S1 += j

        # normalize
        C0, C1, S0, S1 = C0 / shots, C1 / shots, S0 / shots, S1 / shots

        # determine theta_0
        tan_1 = np.arctan2([(1 - 2 * S0)], [(2 * C0 - 1)])[0]
        theta_0 = (1 / (2 * np.pi)) * tan_1

        # determine theta_1
        tan_2 = np.arctan2([(2 * S1 - 1)], [(1 - 2 * C1)])[0]
        theta_1 = (1 / (2 * np.pi)) * tan_2

        phase_dec = np.average([theta_0, theta_1])
        phase_binary = []
        phase = phase_dec

        # generate the binary representation
        for i in range(int(np.log2((1 / self.precision)))):
            phase *= 2
            if phase < 1:
                phase_binary.append(0)
            else:
                phase -= 1
                phase_binary.append(1)

        return (phase_dec, phase_binary)

    def get_circuit(self, show=False, save_circ=False, circ_name="KQPE_circ.JPG"):
        """Returns a kitaev phase estimation circuit
        with the unitary provided

        Args:
            show(bool) : whether to draw the circuit or not
                         Default - False
            save_circ(bool) : whether to save the circuit in
                         an image or not. Default - False
             circ_name(str) : filename with which the circuit
                         is stored. Default - KQPE_circ.JPG

         Returns : A QuantumCircuit with the controlled unitary matrix
                   and relevant gates attached to the circuit.
                   Size of the circuit is (2 + the number of qubits in unitary)

         Examples:
             theta = 1/5
             unitary = UnitaryGate(np.ndarray([[1,0],
                             [0, np.exp(2*np.pi*1j*(theta))]]))
             kqpe = KQPE(unitary,precision = 10)
             kq_circ = kqpe.get_circuit(show = True,save_circ = True, circ_name= "KQPE_circ_1qubit.JPG")

             # attaching the circuit
             q = QuantumCircuit(5, 6)
             q.x(3)
             q.append(kq_circ, qargs=[1, 2, 3])
             q.draw('mpl')

        """
        qc = QuantumCircuit(2 + self.qubits, name="KQPE")
        qubits = [i for i in range(2, 2 + self.qubits)]

        # make the unitary
        if isinstance(self.unitary, np.ndarray):
            U = UnitaryGate(data=self.unitary)
            C_U = U.control(num_ctrl_qubits=1, label="CU", ctrl_state="1")
        else:
            C_U = self.unitary.control(num_ctrl_qubits=1, label="CU", ctrl_state="1")

        # qubit 0 is for the H estimation
        qc.h(0)
        qc = qc.compose(C_U, qubits=[0] + qubits)
        qc.h(0)
        qc.barrier()

        # qubit 1 is for the H + S estimation
        qc.h(1)
        qc.s(1)
        qc = qc.compose(C_U, qubits=[1] + qubits)
        qc.h(1)

        qc.barrier()
        if show == True:
            if save_circ:
                display(qc.draw("mpl", filename=circ_name))
            else:
                display(qc.draw("mpl"))

        return qc
