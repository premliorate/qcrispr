Received: d month yyyy | Revised: d month yyyy | Accepted: d month yyyy | Published online: d month yyyy 

### **RESEARCH ARTICLE/REVIEW** 

# **Quantum State Mapping for High-Precision CRISPR-Cas9 Gene Editing** 

Journal of Computational and Cognitive Engineering 

yyyy, Vol. XX(XX) 1–5 

DOI: 10.47852/bonviewJCCEXXXXXXXX 



<!-- Start of picture text -->
2)<br>)<br><!-- End of picture text -->

**Prem Santh CK**<sup>**1**</sup> **, Elluru Ritesh Goud**<sup>**1**</sup> **, Anagha Rajan**<sup>**2**</sup> **and I R Oviya**<sup>**1**</sup> 

_1 Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai, India, premliorate@gmail.com, ellururiteshgoud@gmail.com,_ 

_2 Department of Sciences, Amrita School of Engineering, Amrita Vishwa Vidyapeetham, Chennai, India, r_anagha@ch.students.amrita.edu_ 

_*Corresponding author: I R Oviya, Department of Sciences, Amrita School of Engineering, Amrita Vishwa Vidyapeetham, Chennai, India. Email: ir_oviya@ch.amrita.edu ORCID:  0000-0002-1421-4232_ 

**Abstract:** _The application of CRISPR-Cas9 gene editing requires the precise identification of rare off-target mutations to ensure clinical safety. Although classical deep learning models are effective at detecting these anomalies, they often require extreme parameterization to overcome the severe class imbalances, frequently exceeding 1:6000 found in biological data. This paper proposes a parameter-efficient Hybrid Quantum Convolutional Neural Network (QCNN) designed specifically for genomic anomaly detection. To empirically mitigate the “barren plateau” phenomenon of vanishing gradients that typically stalls quantum optimization, we implement a mathematically bounded classical-to-quantum angle embedding. This model maintains gradient flow and unlocks a complex pattern-matching power without saturating the computational basis states. When evaluated in a clinical data set with an imbalance of 1:6816, our Hybrid QCNN achieved a state-of-the-art ROC-AUC of 0.989 using only 109 trainable parameters. This represents a 99.1% reduction in computational footprint compared to a functionally equivalent 12,417-parameter classical baseline. This paper validated the resilience of the architecture to real-world decoherence by executing inference directly on the Rigetti Ankaa-3 Superconducting QPU, where it maintained high predictive fidelity (AUC = 0.86). These results establish a hardware-validated pathway for integrating expressive, low-power quantum architectures into the future of clinical bioinformatics._ 

**Keywords: quantum machine learning, CRISPR-Cas9, parameterized quantum circuits, barren plateaus, anomaly detection** 

> © The Author(s) 2026. Published by BON VIEW PUBLISHING PTE. LTD. This is an open access article under the CC BY License (1 https://creativecommons.org/ licenses/by/4.0/). 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

#### **1. Introduction** 

The CRISPR-Cas9 (Clustered Regularly Interspaced Short Palindromic Repeats) system has fundamentally revolutionized targeted genomic engineering, offering unprecedented potential for clinical gene therapy [1], [2]. However, the dependence of the system on a 20nucleotide guide RNA is vulnerable to tolerances of critical energy change mismatch. These tolerances often lead to off-target cleavage events, which are unintended DNA breaks and severe cellular lethality, chromosomal rearrangements, and oncogenesis. Early predictive models of detecting these dangerous anomalies were based on handcrafted biological and thermodynamic matrices, the MIT scoring system [3], and the most famous, the Cutting Frequency Determination (CFD) score [4]. These cleavage events were further validated by stateof-the-art in vivo profiling methods like GUIDEseq [5], which produced detailed datasets that mapped empirical off-target activities [6]. While mathematically interpretable, and rigorously evaluated against varied algorithms [7], early linear thermodynamic models failed to capture completely the multidimensional spatial dependencies involved in Cas9-DNA binding [8]. 

To model these non-linear biological realities, the bioinformatics community turned more and more toward classical deep learning. Convolutional Neural Networks (CNNs) and novel sgRNADNA sequence encodings, such as CRISPR-Net [9] and DeepCRISPR [10], along with advanced optimized guide RNA predictors [11], were shown to have better predictive capacity and project skill beyond standard levels by independently deriving hierarchical spatial features. However, clinical CRISPR datasets are a severe computational challenge: extreme class imbalance. Confirmed off-target mutations are biologically unusual, usually resulting in minority-to-majority class ratios exceeding 1:6000. To sail through this extreme sparsity without suffering from predictive class collapse, classical neural networks have to be based on brute-force over-parameterization, frequently tens of thousands of trainable weights to memorize the boundaries of the minority classes. 

Quantum Machine Learning (QML) [12] provides a fundamentally new, extremely parameter-efficient computational model. From discrete data to the continuous, exponentially scaling Hilbert space of a parameterized quantum circuit (PQC), QML models, in particular, Quantum Convolutional Neural Networks (QCNNs) [13] can theoretically achieve expressivity with a fraction of the parameter footprint [14]. Recent theoretical proofs demonstrating the resilience of QCNNs against barren plateaus [15] have expanded their capacity for complex classical data classification [16]. The intersection of quantum computing and biological sciences [17] has successfully demonstrated robust QML architectures capable of predicting genomic transcription factor bindings with base resolution [18], [19]. Furthermore, leveraging quantum-enhanced feature spaces [20] within the Noisy Intermediate-Scale Quantum (NISQ) era [21], the development of quantum transfer learning protocols has successfully bridged classical optimization with quantum nodes [22], building on the rigorous theoretical foundations of quantum neural network expressivity [23], enabling the deployment of hybrid neural networks directly onto physical quantum hardware. 

Despite these advances, scaling QML for complex genomic anomaly detection is frequently bottlenecked by the “barren plateau” phenomenon, of vanishing gradients that stalls optimization [24]. This includes cost-function dependent plateaus highly relevant to shallow circuits [25]. In continuous quantum classifiers, this often occurs when raw classical data is naively encoded into Pauli rotation gates, forcing the qubits into extreme, saturated basis states and destroying the gradient landscape [26], a challenge heavily influenced by the expressibility of the chosen ansatz [27], which necessitates optimized quantum embedding strategies [28]. 

In this paper, we address both the biological imperative for CRISPR safety and the computational bottleneck of classical deep learning. We introduce a highly parameterefficient Hybrid QCNN designed to predict rare CRISPR-Cas9 off-target mutations under 

2 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

extreme class sparsity. By mathematically constraining our classical-to-quantum angle embedding, we successfully resolve gradient saturation, enabling the quantum feature space to naturally and efficiently model complex thermodynamic binding affinities. 

### **1.1 Significance of the Research** 

The primary significance of this work lies in advancing both the theoretical foundations and practical deployment of hybrid quantum learning architectures for clinical bioinformatics. By addressing key challenges in quantum optimization, parameter efficiency, and realworld implementation on Noisy IntermediateScale Quantum (NISQ) hardware, this research demonstrates the potential of quantum machine learning to enable scalable, energy-efficient, and clinically relevant genomic analysis. The significance of this research is summarized as follows: 

- Introduces a mathematically constrained [0, π/2] angle-embedding strategy that mitigates gradient saturation, enabling stable optimization and efficient quantum feature encoding. 

- Establishes a parameter-efficient hybrid quantum learning framework that exploits the high-dimensional quantum Hilbert space for robust genomic anomaly detection, particularly in sparse biomedical datasets. 

- Demonstrates the practical feasibility of hybrid quantum learning through deployment of a decoupled transfer learning framework on NISQ superconducting quantum hardware, validating resilience to quantum noise and decoherence. 

- Bridges quantum computing and clinical bioinformatics by providing a scalable and energy-efficient framework for genomic disease prediction and precision medicine applications. 

- Lays the foundation for future quantumassisted clinical workflows, including genomic diagnostics, biological anomaly detection, and gene-editing safety assessment. 

**Table 1 Comparison of related work in CRISPR-Cas9 off-target prediction** 

|**Categories**|**Key Methodologies & References**|**Core Contributions**|**Critical Limitations / Research**<br>**Gaps**|
|---|---|---|---|
|Thermodynamic<br>Baselines|MIT Score [3], CFD Score [4],<br>GUIDE-seq profiling [5], [6],<br>algorithm evaluation [7], spatial<br>dependencies [8]|Established ground-truth<br>baseline mismatch tolerances<br>and highly interpretable<br>biological matrices.|Mathematically linear, struggles to<br>capture complex, multi-<br>dimensional spatial dependencies<br>of Cas9 binding.|
|Classical Deep Learni|ng CRISPR-Net [9], DeepCRISPR<br>[10], optimized predictors [11]|Autonomously extracts<br>hierarchical non-linear spatial<br>features using deep<br>Convolutional Neural Networks.|Requires massive over-<br>parameterization to navigate<br>extreme class sparsity, energy-<br>intensive, and prone to majority-<br>class bias.|
|Quantum Machine<br>Learning|QML foundations [12], QCNNs<br>[13], [16], BP immunity [15],<br>Genomic QML [17], [18], [19],<br>Quantum kernels [20], NISQ [21],<br>[22], QNN power [23]|Achieves profound<br>mathematical expressivity using<br>a fraction of the parameter<br>footprint [14].|Naive angle embeddings force<br>discrete qubit saturation, triggering<br>barren plateaus [24], [25], [26]<br>without optimal ansätze [27] or<br>embeddings [28].|
|ClinicalNano-|Gene-editing therapeutics [1], [2]|Demonstrates the clinical|Off-target safety constraints|



3 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

|Therapeutics||urgency and delivery challenges<br>of CRISPR-Cas9 in biological<br>therapy contexts.|<br>demand highly precise<br>computational prediction pipelines.|
|---|---|---|---|
|Proposed Work|Hybrid QCNN (Our Architecture)|Empirically mitigates barren<br>plateaus via bounded [0, π/2]<br>angle embedding; hardware-<br>validated on Rigetti QPU.|Addresses the QML parameter<br>bottleneck, future work will target<br>biological sgRNA base-resolution<br>encoding.|



## **2. Proposed Methodology** 

Our approach closes the theoretical gap between classical deep learning and Noisy IntermediateScale Quantum (NISQ) optimization. To carefully disentangle the mathematical expressivity of the quantum feature space, we developed an end-to-end system that combines an ultra-low-parameter Hybrid QCNN with classical preprocessing and post-processing. The high-level system architecture, shown in Fig. 1, 

can be separated into four distinct domains: (1) classical data ingestion and preprocessing of highly imbalanced clinical CRISPR sequences, (2) a classical-to-quantum bottleneck with a dynamically constrained angle embedding layer, (3) a Variational Quantum Circuit (VQC) that embeds the latent angles into a complex, entangled Hilbert space, and (4) classical postprocessing that converts the quantum expectation values into a binary classification 

**Figure. 1. System architecture of the proposed end-to-end Hybrid QCNN for CRISPR anomaly detection.** 



<!-- Start of picture text -->
COLUMN 1 COLUMN 2 COLUMN3 COLUMN 4<br>Classical-to-Quantum Bottleneck ae vist<br>pen 88:—98;—2@ 4<br>is 88t1—8s1—@ one _<br>@@—_4+ }_4+@ ae<br>—_—<br>(CtinicatCRIsPR-Cas9 Angie<br>Dataset(Imbatance: 1:6816} incense<br><!-- End of picture text -->

### **2.1 Dataset and Preprocessing** 

The genomic dataset utilized in this study was sourced from the publicly available CRISPRCas9 GUIDE-seq dataset published by Listgarten et al. [6], available at <u>https://github.com/dagrate/public_data_crisprCa s9. The dataset represents authentic clinical-grade</u> CRISPR-Cas9 cleavage events. The input data 

consists of 23-base pair sequences. To interface with the hybrid network, these sequences were integer-encoded into binary match/mismatch arrays, representing the thermodynamic binding state between the guide RNA and the target DNA. 

The clinical CRISPR data is extremely sparse. The dataset exhibits a minority to majority class imbalance of 1:6816. To preserve authentic 

4 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

clinical noise, no synthetic oversampling was applied. The dataset was partitioned into two phases to evaluate both mathematical expressivity and physical hardware resilience. A massive holdout set of 5,024 unseen sequences was reserved for a state vector simulation, while a targeted, representative slice of 61 sequences was distinctly partitioned for physical QPU hardware inference. To prevent predictive class 

collapse, we implemented a dynamically weighted binary cross-entropy loss. By applying a deterministic positive weight multiplier (pos_weight = 6816.11), we forced the optimizer to penalize false negatives rigorously. 

### **2.2 The Hybrid QCNN and Mitigating the Barren Plateau** 

To evaluate quantum expressivity, we replaced the classical convolutional feature extractor with a parameterized quantum circuit (PQC), processing the encoded genomic sequence through a 4-qubit sliding window. A key challenge in quantum machine learning is the “barren plateau” phenomenon, a vanishing gradient landscape that stalls optimization. In early iterations, mapping raw classical weights directly into quantum Pauli-Rx rotation gates forced the qubits into discrete, extreme computational basis states (|0 and |1⟩ ⟩). This essentially reduced the quantum circuit to an inefficient classical calculator, limiting the model's AUC to 0.67. 

To restore gradient flow and unlock true quantum superposition, we introduced a highly constrained 

classical-to-quantum embedding layer. Before entering the quantum circuit, the classical input x is dynamically bounded using a scaled Sigmoid activation function: 

_x_in = σ(Wx + b) × (π/2)          (1)_ 

By mathematically constraining the rotation angles to the [0, π/2] continuous range, we restrict the dynamical Lie algebra of the parameterized circuit. This prevents the circuit from forming an approximate unitary 2-design, which is the foundational cause of the barren plateau phenomenon. Consequently, we force the initial Rx gates to prepare the sequence data in partial, continuous superpositions. This critical adjustment prevents state saturation, preserves optimal gradient flow for PyTorch's classical backpropagation, and allows the subsequent entanglement layers to function as a highly expressive non-linear feature space. 

The architectural flow of the parameterized quantum circuit (PQC), shown in Fig. 2, demonstrates this constrained embedding. The genomic sequence enters the circuit through the initial layer of Pauli-Rx gates, which act as the state preparation interface. Because the input angles are bounded, these Rx gates successfully load the classical data into partial superpositions without saturating the qubits. This encoded state is then passed into the strongly entangling ansatz comprising parameterized Ry rotations and cascading CNOT gates which weaves the independent qubits into a joint, highly expressive quantum feature space. Finally, the observable measurements extract the continuous expectation values, bridging the quantum logic back to the classical classification layer. 

5 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

**Figure. 2. The Hybrid QCNN parameterized quantum circuit featuring bounded Rx embedding and a strongly entangling architecture.** 



<!-- Start of picture text -->
m RX RY CT)_[ ev CT)<br>7 ==RX ry | INand =ry  ;| (T)an<br>RX RY Cl RY a<br>y =a Fe to A<br>S =RX RY ~ (1)FloRY TA(I<br>(0.00) F~} (0.00) a (0.00) CD<br><!-- End of picture text -->

### **2.3 Entanglement Ansatz and Measurement** 

Once the genomic subset is encoded into partial superpositions, the system must learn the complex spatial relationships between the nucleotide mismatches. We employ a strongly entangling ansatz consisting of parameterized Pauli-Ry rotations followed by a cascading ring of CNOT gates. Because the qubits are in continuous superposition, this entanglement natively projects the 24-bp sequence into a highly complex, 16-dimensional Hilbert space. At the end of the quantum circuit, we measure each qubit in the Pauli-Z basis, turning them into four continuous values. These numbers then feed into a classical dense layer, which learns how to combine them to perform binary classification. By utilizing the continuous state space, this entire hybrid architecture requires only 109 trainable parameters, a 99.1% reduction in computational footprint compared to the classical baseline. 

evaluated on a massive holdout set of 5,024 unseen sequences utilizing PennyLane's highly optimized lightning.qubit state vector simulator. 

Second, to validate real-world hardware resilience, we executed a transfer learning protocol. The optimized parameters from the simulation were decoupled and compiled directly onto the physical Rigetti Ankaa-3 Superconducting Quantum Processing Unit (QPU) via the BlueQubit cloud platform. The quantum circuit was transpiled into the native CZ and CPHASE gate sets of the Rigetti Ankaa-3 topology. Standard Readout Error Mitigation (REM) was applied during cloud execution to filter localized decoherence noise. Executing a targeted holdout slice on physical superconducting qubits allows us to empirically verify that our constrained angle-embedding strategy is robust against the inherent gate infidelities of modern hardware. 

## **3. Results and Discussion** 

### **2.4 Two-Phase Hardware and Simulation Execution** 

Mathematical simulation alone is insufficient to prove clinical viability in the Noisy IntermediateScale Quantum (NISQ) era. Therefore, our execution protocol was divided into two phases. First, to establish statistical significance and mathematical expressivity, the models were 

### **3.1 Evaluation Metrics and Environment** 

To evaluate the mathematical expressivity of the architectures, all classical (PyTorch) and quantum (PennyLane lightning.qubit) simulations were optimized using the Adam optimizer. For reproducible training dynamics, the models were trained with a learning rate of 

6 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

0.001 and a batch size of 64 over 100 epochs. Given the extreme 1:6816 class imbalance of the CRISPR dataset, standard accuracy is mathematically deceptive. Therefore, the Receiver Operating Characteristic Area Under the Curve (ROC-AUC) was utilized as the primary metric to evaluate the models' ability to separate the continuous risk distributions of dangerous mutations from safe sequences on the 5,024-sequence holdout set. 

**3.2 Phase 1: Statistical Expressivity and Parameter Efficiency** 

The core hypothesis of this research, that a continuous quantum Hilbert space possesses higher inherent expressivity for biological anomaly detection than discrete classical nodes, was rigorously validated in Phase 1. 

**Figure. 3. ROC validation on random holdout partition.** 



<!-- Start of picture text -->
ROC Validation: Quantum Expressivity vs Classical Scale<br>10 ————<br>_-<br>c==E<br>306 it<br>ri qH<br>2 qj<br>=4<br>3H<br>2<br>Foo<br><+++ Over-parameterized Classical CNN (12k params) - AUC = 0.98<br>— hybrid QENN G09 params) - AUC = 0.99,<br>0.0 —— Real CFD Score (Biological Baseline) - AUC = 0.96<br>False Positive Rate<br><!-- End of picture text -->

The biological gold-standard CFD thermodynamic matrix achieved an AUC of 0.957, validating the structural integrity of the test distribution. The classical 1D-CNN baseline successfully navigated the dataset sparsity to achieve an AUC of 0.983. However, it required an expansive 12,417 trainable parameters to forcefully memorize the minority class boundaries through brute-force overparameterization. 

In stark contrast, our optimized Hybrid QCNN achieved a state-of-the-art AUC of 0.989, utilizing only 109 trainable parameters. By outperforming heavily parameterized classical convolutions with a 99.1% reduction in computational footprint, we mathematically demonstrate that the partial superpositions produced by the bounded angle-embedding natively and efficiently capture the non-linear thermodynamic signatures of CRISPR-Cas9 offtarget anomalies. However, because this preliminary evaluation was performed on a 

random holdout partition rather than sgRNAlevel splitting, this high AUC reflects profound mathematical expressivity but cannot yet be treated as an absolute measure of clinical generalizability across completely unseen guide RNAs. 

### **3.3 Clinical Anomaly Detection (Sensitivity)** 

In clinical bioinformatics, the cost of a false negative (failing to predict a dangerous gene mutation) is catastrophic. Therefore, we evaluated the model's practical sensitivity on the mutated sequences. 

Despite the 99.1% parameter reduction, the Hybrid QCNN demonstrated exceptional recall. Out of 11 verified dangerous off-target cleavage events in a targeted sample, the quantum architecture successfully identified 9 (an 81.8% sensitivity rate). This confirms that the model did not artificially inflate its ROC-AUC by over- 

7 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

predicting the majority class, but actively learned the minority anomaly distributions. 

**Figure. 4. Clinical sensitivity: successful isolation of rare off-target mutations under extreme 1:6816 class imbalance.** 



<!-- Start of picture text -->
Hybrid QCNN Confusion Matrix.<br>(Optimal Threshold: 0.0092)<br>“<br>7 be<br><2 42<br>ge pe<br>&<br>4<br>3s 25<br>&><br>3<br>Rs<br>Ls<br>Al Predicted Label<br><!-- End of picture text -->

### **3.4 Phase 2: Physical Hardware Execution (Rigetti Ankaa-3)** 

While simulated expressivity is theoretically significant, practical clinical integration requires 

resilience against real-world quantum noise. In Phase 2, the frozen optimized weights were decoupled and deployed onto the physical Rigetti Ankaa-3 Superconducting Quantum Processing Unit (QPU). 

Due to cloud API constraints and QPU queue limits, inference was executed on a targeted 61sequence holdout slice. The hardware-executed QCNN maintained high predictive fidelity, achieving an AUC of 0.86 directly on the physical superconducting qubits. It must be noted that because the hardware evaluation was limited to 61 sequences yielding fewer than 1 expected offtarget event at the 1:6816 imbalance, the confidence interval for this metric is very wide. Further large-scale hardware evaluation is required to distinguish true hardware-induced degradation from sampling variability definitively. This high predictive retention is the cornerstone of our hardware findings. It empirically confirms that our constrained embedding strategy and low-depth entangling ansatz are highly resilient to gate infidelities and decoherence. The successful extraction of biological anomaly signatures from a physical QPU establishes a concrete, hardware-validated pathway for integrating low-energy NISQ technology into future gene-editing pipelines. 

**Figure. 5. Hardware verification: measurement counts (out of 100 shots) executed directly on the physical Rigetti Ankaa-3 QPU.** 



<!-- Start of picture text -->
Quantum State Distribution on Physical QPU (Rigetti Ankaa-3)<br>Target Sequence: CRISPR Mutation Window<br>“6s DominantAl Signal Collapse (‘1111') az<br>a 40 | {HBSB Secondary HarmonicHardware Decoherence (0001)/ Noise<br>G3<br>zx<br>oa<br>5 20 19<br>8<br>515<br>&<br>2 10<br>2 10<br>=2 5 5<br>5<br>22<br>°<br>><br>ef FF F&Ff F- FF£ © FF£ - FSF£ F& SFf SS$F S$ F$> SF2<br>4-Qubit Measurement State (Computational Basis)<br><!-- End of picture text -->

8 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

## **4. Future Scope** 

While the proposed Hybrid QCNN demonstrates profound parameter efficiency and hardware resilience, critical research gaps remain before deployment in clinical gene-editing pipelines. 

First, the current data partitioning strategy relies on standard random holdout sets. This presents a risk of intra-guide sequence memorization. Future iterations will utilize rigorous sgRNAlevel GroupKFold cross-validation, ensuring the model is evaluated strictly out-of-distribution on entirely novel genomic targets. 

blueprint for the future of highly scalable genomic engineering. 

### **Data Availability Statement** 

To protect the security of proprietary cloud API credentials required for the Rigetti Ankaa-3 QPU execution, the complete codebase and raw hardware execution logs are not hosted on public repositories. The simulation framework, datasets, and physical deployment code will be made available directly to the referees and editorial board upon request for peer-review validation. 

### **Author Contribution Statement** 

Second, the current mathematically efficient angle-embedding simplifies nucleotide interactions into continuous, binary match/mismatch scalars. This abstraction limits the model's ability to interpret specific biochemical binding affinities. Subsequent architectures must expand the Pauli rotation 

sequence to natively embed base-resolution fractional transitions, allowing the quantum entanglement layers to model the complete thermodynamic reality of the Cas9-DNA interface. 

Prem Santh CK: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data Curation Visualization,Writing - Original Draft. Elluru Ritesh Goud: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data Curation Visualization,Writing - Original Draft. Anagha Rajan: Conceptualization, Methodology, Review & Editing. Iyyappan Ramalakshmi Oviya: Methodology, Review & Editing, Supervision, Project administration. All authors read the manuscript and agreed before the submission. 

## **5. Conclusion** 

**Funding: NA** 

This study successfully bridges classical deep learning and Noisy Intermediate-Scale Quantum (NISQ) mechanics to address the computational bottlenecks of clinical anomaly detection. By mathematically bounding our classical-toquantum angle embedding, we empirically mitigated the barren plateau phenomenon for this circuit depth, enabling the quantum feature space to natively capture the non-linear thermodynamics of CRISPR-Cas9 mutations. Evaluated against extreme 1:6816 class sparsity, the proposed Hybrid QCNN achieved a state-ofthe-art ROC-AUC of 0.989, utilizing only 109 trainable parameters. This represents a 99.1% computational reduction compared to a functionally equivalent 12,417-parameter classical baseline. Furthermore, the successful execution of the optimized architecture on the physical Rigetti Ankaa-3 superconducting QPU establishes a hardware-validated, low-energy 

### **Acknowledgement** 

We would like to extend our gratitude to our mathematics faculty for their rigorous review and guidance regarding the mathematical formalisms presented in this manuscript. Furthermore, we sincerely thank BlueQubit for providing the cloud infrastructure and API access necessary to execute our inference models directly on the physical Rigetti Ankaa-3 Superconducting QPU. 

**Ethical Statement** 

- 1) This study does not contain any studies with human subjects performed by any of the authors. 

- 2) Ethical approval was waived or not required in accordance with national regulations. 

9 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

#### **Conflicts of Interest** 

- 1) The authors declare that they have no conflicts of interest to this work. 

## **References** 

[1] N. Nujoom, M. Koyakutty, L. Biswas, T. Rajkumar, and S. V. Nair, “Emerging geneediting nano-therapeutics for cancer,” _Heliyon_ , vol. 10, no. 21, e39323, Nov. 2024. 

[2] F. J. M. Mojica and L. Montoliu, “On the Origin of CRISPR-Cas Technology: From Prokaryotes to Mammals,” _Cell_ , vol. 166, no. 4, pp. 816–821, 2016. 

[3] P. D. Hsu _et al._ , “DNA targeting specificity of RNA-guided Cas9 nucleases,” _Nature Biotechnology_ , vol. 31, no. 9, pp. 827–832, 2013. 

[4] J. G. Doench _et al._ , “Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9,” _Nature Biotechnology_ , vol. 34, no. 2, pp. 184–191, 2016. 

[5] S. Q. Tsai _et al._ , “GUIDE-seq enables genome-wide profiling of off-target cleavage by CRISPR-Cas nucleases,” _Nature Biotechnology_ , vol. 33, no. 2, pp. 187–197, 2015. 

[6] J. Listgarten _et al._ , “Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs,” _Nature Biomedical Engineering_ , vol. 2, pp. 38–47, 2018. 

[7] M. Haeussler _et al._ , “Evaluation of off-target and on-target scoring algorithms and integration into the guide RNA selection tool CRISPOR,” _Genome Biology_ , vol. 17, no. 1, p. 148, 2016. 

[8] J. Lin and K. C. Wong, “Off-target predictions in CRISPR-Cas9 gene editing using deep learning,” _Bioinformatics_ , vol. 34, no. 17, pp. i656–i663, 2018. 

[9] J. Charlier, R. Nadon, and V. Makarenkov, “Accurate deep learning off-target prediction with novel sgRNA-DNA sequence encoding in CRISPR-Cas9 gene editing,” _Bioinformatics_ , vol. 37, no. 16, pp. 2299–2307, 2021. 

[10] G. Chuai _et al._ , “DeepCRISPR: optimized CRISPR guide RNA design by deep learning,” _Genome Biology_ , vol. 19, no. 1, p. 80, 2018. 

[11] D. Wang _et al._ , “Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning,” _Nature Communications_ , vol. 10, p. 4284, 2019. 

[12] J. Biamonte _et al._ , “Quantum machine learning,” _Nature_ , vol. 549, no. 7671, pp. 195– 202, 2017. 

[13] I. Cong, S. Choi, and M. D. Lukin, “Quantum convolutional neural networks,” _Nature Physics_ , vol. 15, no. 12, pp. 1273–1278, 2019. 

[14] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, “Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms,” _Advanced Quantum Technologies_ , vol. 2, no. 12, 1900070, 2019. 

[15] A. Pesah, M. Cerezo, S. Wang, T. Volkoff, A. T. Sornborger, and P. J. Coles, “Absence of barren plateaus in quantum convolutional neural networks,” _Physical Review X_ , vol. 11, no. 4, 041011, 2021. 

[16] T. Hur, L. Kim, and D. K. Park, “Quantum convolutional neural network for classical data classification,” _Quantum Machine Intelligence_ , vol. 4, no. 1, p. 3, 2022. 

[17] P. S. Emani _et al._ , “Quantum computing at the frontiers of biological sciences,” _Nature Methods_ , vol. 18, no. 5, pp. 459–469, 2021. 

[18] R. Li _et al._ , “Robust high-performance quantum machine learning modeling that predicts main and cooperative transcription factor bindings with base resolution,” _Briefings in Bioinformatics_ , vol. 24, no. 1, 2023. 

[19] G. Pallavi and R. P. Kumar, “Quantum natural language processing and its applications in bioinformatics: a comprehensive review of methodologies, concepts, and future directions,” _Frontiers in Computer Science_ , vol. 7, 1464122, Feb. 2025. 

10 

Journal of Computational and Cognitive Engineering  Vol. XX Iss. XX yyyy 

[20] V. Havlíček _et al._ , “Supervised learning with quantum-enhanced feature spaces,” _Nature_ , vol. 567, no. 7747, pp. 209–212, 2019. 

[21] J. Preskill, “Quantum Computing in the NISQ era and beyond,” _Quantum_ , vol. 2, p. 79, 2018. 

[22] A. Mari, T. R. Bromley, J. Izaac, M. Schuld, and N. Killoran, “Transfer learning in hybrid classical-quantum neural networks,” _Quantum_ , vol. 4, p. 340, 2020. 

[23] A. Abbas _et al._ , “The power of quantum neural networks,” _Nature Computational Science_ , vol. 1, no. 6, pp. 403–409, 2021. 

[24] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, “Barren plateaus in quantum neural network training landscapes,” _Nature Communications_ , vol. 9, no. 1, p. 4812, 2018. 

[25] M. Cerezo _et al._ , “Cost function dependent barren plateaus in shallow parametrized quantum circuits,” _Nature Communications_ , vol. 12, p. 1791, 2021. 

[26] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, “Circuit-centric quantum classifiers,” _Physical Review A_ , vol. 101, no. 3, 032308, 2020. 

[27] K. Nakaji and N. Yamamoto, “Expressibility of the alternating layered ansatz for quantum computation,” _Quantum_ , vol. 5, p. 434, 2021. 

[28] S. Lloyd, M. Schuld, A. Ijaz, J. Izaac, and N. Killoran, “Quantum embeddings for machine learning,” _arXiv preprint arXiv:2001.03622_ , 2020. 

11 

