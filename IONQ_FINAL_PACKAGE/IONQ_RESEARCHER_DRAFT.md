Subject: Research Credit Application: Hardware Fidelity of a Hybrid Quantum Classifier for CRISPR Off-Target Prediction

Dear IonQ Research Team,

We are seeking IonQ research credits to execute the same frozen circuit and parameters on IonQ hardware and quantify hardware fidelity relative to the validated simulation reference. 

Our canonical circuit is a 4-qubit, 109-parameter hybrid quantum-classical neural network. It utilizes a 24->4 classical bottleneck, encodes data with RX rotations, uses one trainable RX layer, a 4-CNOT ring, and measures four Pauli-Z expectation values. The model parameters are completely frozen.

We have successfully executed our cloud simulations on the IonQ ideal and noise simulators. Using a stratified 111-sample comparison set, the IonQ ideal cloud simulation yielded a ROC-AUC of 0.8555 (Pearson correlation of 0.9253 with our local ideal simulation). The IonQ Aria-1 noise-model simulation on the same 111 samples yielded a ROC-AUC of 0.8818. For context, our local full-test reference (76,693 samples) evaluates to ROC-AUC 0.8704.

We do not claim quantum advantage in this experiment. Our objective is to benchmark hardware-fidelity inference behavior on a heavily imbalanced bio-informatics dataset.

Thank you for your consideration.

Sincerely,
[Researcher Name]
