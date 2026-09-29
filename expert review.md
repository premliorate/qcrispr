new journal 1 (1).md 

File 

new journal 1 (1).docx 

## Document 

Quantum_CRISPR_Phase1_ipynb_FINAL for submission (1).ipynb 

File 

go through it throughly 

I went through the manuscript in both .md and .docx, and the notebook cell-by-cell, including the training code, model definitions, saved-weight reconstruction, evaluation code, CFD/MIT baselines, and the BlueQubit/Qiskit sections. 

The important result is that **the paper and the notebook are not describing the same experiment** . That is the central problem. 

## **EXECUTIVE VERDICT** 

## **Blunt assessment** 

The project is **technically interesting but currently not research-grade** . It contains a real experimental idea, a nontrivial hybrid quantum-classical implementation, a very small 109-parameter model, and an interesting numerical result around AUC 0.989. But the manuscript currently makes several claims that the supplied notebook does not support. 

The most serious issue is the claimed **Rigetti Ankaa-3 hardware validation** . The manuscript says the 61-sequence evaluation was executed on a physical Rigetti Ankaa-3 QPU and obtained AUC = 0.86. 

The notebook's corresponding 61-sequence experiment explicitly reports: 

Hardware Backend : BlueQubit Cloud Simulator Unseen Data ROC-AUC : 0.8636 

So the notebook does **not** substantiate the manuscript's claim that AUC = 0.86 was obtained from the Rigetti Ankaa-3. 

There is a separate Qiskit/BlueQubit circuit submission that the notebook describes as going to Ankaa-3, but it is **not the full trained classifier inference** . It manually creates a single 4-qubit circuit for [0, π, 0, 0], applies only the quantum weights from an older model, measures 100 shots, and does not produce the reported 0.86 AUC. 

That alone requires a major revision of the paper. 

The second fundamental issue is that there are actually **multiple different architectures** in the notebook. The manuscript describes a 4-qubit sliding-window hybrid QCNN, while the final 109parameter experiment is a **24→4 classical linear bottleneck + one variational quantum layer + 4→1 output layer** . It has no quantum convolution or pooling hierarchy. 

The third issue is evaluation. Your 5,024-sample 0.9894 experiment is not a conventional 5,024positive/negative test evaluation. You first create a 76,693-row test set, then stop evaluation after 

5,000 samples. The notebook does not report how many positives are actually contained in those 5,024 evaluated rows. 

Given that the full test set contains only 11 positives, this is a huge statistical weakness. 

So my current position is: 

**There is a potentially publishable research idea here, but the present manuscript overstates what the experiments establish.** 

And importantly: 

**The current evidence does not establish quantum advantage.** 

## **WHAT IS ACTUALLY STRONG** 

There are several genuinely useful elements. 

## **1. The problem itself is meaningful** 

CRISPR off-target prediction is a legitimate computational-biology problem, and evaluating rare events under severe imbalance is scientifically relevant. 

The manuscript correctly focuses on off-target activity and the difficulty of extreme imbalance. 

## **2. You found a genuinely lightweight hybrid model** 

The final model has: 

- 24 classical inputs 

- 4 classical bottleneck outputs 

- 4 trainable quantum parameters 

- 4 classical output parameters 

giving 109 total parameters. The notebook explicitly reports 109. 

The corresponding 12,417-parameter CNN is also explicitly implemented and reports AUC 0.9837 versus 0.9894 for the 109-parameter hybrid model on the evaluated subset. 

That is a legitimate experimental observation. 

But it is only an observation under the current protocol—not yet evidence of quantum superiority. 

## **3. You used BCEWithLogitsLoss(pos_weight=...) in the final experiment** 

That is substantially cleaner than the earlier probability-space BCE implementations in the notebook. The final experiment uses: 

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6816.11])) 

for both the hybrid and classical models. 

That part is methodologically defensible. 

## **4. You did not synthetically oversample the minority class** 

The manuscript explicitly says no synthetic oversampling was applied. 

For a study intended to preserve the original benchmark distribution, that is reasonable. 

## **5. You identified the strongest weakness yourself** 

Your future-scope section openly acknowledges that random row-level splitting risks intra-guide memorization and proposes sgRNA-level GroupKFold. 

That diagnosis is correct. 

The problem is that this should be fixed **before** making strong generalization claims, rather than deferred to future work. 

## **CRITICAL PROBLEMS** 

## **1. CRITICAL — The hardware claim is contradicted by the notebook** 

## **What the paper says** 

The manuscript says the 61-sequence Phase 2 experiment was run on the **physical Rigetti Ankaa-3 QPU** , producing AUC = 0.86. 

## **What the notebook actually does** 

The notebook's 61-example experiment uses: 

qml.device("bluequbit.cpu", wires=4, token=API_KEY) 

and later explicitly prints: 

Hardware Backend : BlueQubit Cloud Simulator 

Unseen Data ROC-AUC : 0.8636 

## **Why this is serious** 

That means the sentence 

“hardware-executed QCNN ... achieving AUC = 0.86” 

is not supported by the corresponding notebook experiment. 

This is not a minor wording problem. It changes the scientific meaning of the result. 

## **Required correction** 

You have two choices: 

**Option A — actually perform full-model QPU inference** and retain the hardware claim. 

**Option B — remove the hardware AUC claim entirely** and honestly report the 61-example BlueQubit simulator result. 

For publication, Option A is much more valuable, but it must be genuinely implemented. 

## **2. CRITICAL — The QPU experiment is not the trained model** 

The separate hardware cell manually creates: 

dna_window = [0.0, np.pi, 0.0, 0.0] 

then loads: 

learned_weights = model.q_layer.weights.detach().numpy() and builds a raw 4-qubit Qiskit circuit. 

That means the experiment is only testing a manually selected **4-base quantum circuit** . It does not execute: 

24-base input ↓ classical 24→4 transformation ↓ 

bounded embedding ↓ quantum circuit ↓ 4 expectation values ↓ classical output layer ↓ prediction 

which is the final 109-parameter model. 

The paper therefore currently conflates: 

**quantum circuit demonstration** with **hardware execution of the trained predictive model.** 

They are not equivalent. 

## **3. CRITICAL — Even the quantum embedding changes between experiments** 

Your original quantum circuit is defined using: 

qml.AngleEmbedding(... rotation='Y') 

in the original CRISPR_Hybrid_QCNN. 

The later 109-parameter model uses: 

qml.AngleEmbedding(... rotation='X') 

and a completely different frontend. 

The hardware Qiskit experiment manually applies: 

qc.rx(...) 

again. 

So you have: 

**Model A:** Y-angle embedding **Model B:** X-angle embedding + classical bottleneck **Hardware:** manually reconstructed X-angle circuit using weights from Model A 

That is scientifically inconsistent. 

## **4. CRITICAL — Your “QCNN” is not really a QCNN** 

This needs to be corrected. 

The original architecture in the notebook applies the same 4-qubit circuit over sliding windows and then concatenates outputs. 

The final 109-param architecture is even simpler: 

self.fc_in = nn.Linear(24, 4) 

qml.AngleEmbedding(...) 

qml.BasicEntanglerLayers(...) 

self.fc_out = nn.Linear(4, 1) 

There is no hierarchical quantum convolution/pooling structure. 

The canonical QCNN literature describes QCNNs using convolutional and pooling layers that reduce the number of qubits/degrees of freedom hierarchically. Pesah et al.'s trainability result is specifically about that architecture. 

So calling the final model a **Hybrid QCNN** invites an immediate reviewer objection. 

## **Better terminology** 

Something like: 

**Hybrid Variational Quantum Classifier** 

or 

## **Parameter-Efficient Hybrid Quantum Neural Network** 

would be much more technically accurate. 

Alternatively, actually implement a genuine QCNN. 

## **5. CRITICAL — The barren-plateau explanation is currently not mathematically justified** 

The manuscript claims: 

bounding the angles to [0, π/2] restricts the dynamical Lie algebra and prevents formation of an approximate unitary 2-design. 

This is the weakest theoretical statement in the paper. 

A numerical parameter range does **not** , by itself, change the Lie algebra generated by the gate generators. 

For example, changing the allowed numerical range of a parameter in 



does not change the underlying generator _X_ / 2. 

Likewise, simply restricting _θ ∈_ [ 0 _,π_ / 2 ]is not a proof that the resulting circuit family cannot approximate a 2-design. 

The literature connects barren plateaus to properties of circuit ensembles and gradient concentration; the 2-design connection is real, but your manuscript jumps from that general theory to a specific claim that the sigmoid bound prevents a 2-design without providing the required analysis. 

## **What you actually need to demonstrate** 

Measure, experimentally: 



as a function of: 

- number of qubits 

- circuit depth 

- initialization 

- embedding strategy. 

Compare at least: 



versus 



> [ 0 _,π_ / 2 ] _._ 

versus 

Then measure circuit expressibility / distributional closeness and entangling capability. 

Until then, the correct claim is: 

“The bounded embedding empirically produced more favorable gradients in our tested configuration.” 

Not: 

“We prevent approximate 2-design formation.” 

## **6. HIGH — “state-of-the-art” is not supported** 

The manuscript repeatedly describes 0.989 as state-of-the-art. 

That is not defensible. 

Published CRISPR-Net work reports AUROC = **0.995** on Dataset II/6 under its independent-testing protocol. 

A later benchmarking study also reports CRISPR-Net at **0.990 AUROC / 0.990 PRAUC** on severeimbalance datasets, including II6. 

## So: 

## **0.989 cannot be called state-of-the-art without qualification.** 

You are also using a different split protocol, which makes direct comparison even more difficult. 

## **Replace** 

“state-of-the-art ROC-AUC of 0.989” 

with something like: 

“ROC-AUC of 0.989 on our random holdout evaluation.” 

That is much harder for a reviewer to attack. 

## **7. CRITICAL — The 5,024-sample evaluation is statistically problematic** 

Your full test set contains: 

## **76,693 samples.** 

You then evaluate only the first **5,024 samples** from test_loader. 

But the entire test set contains only 11 positives, corresponding to the 45 positives in training plus 11 in testing shown by the notebook's class counts. 

Therefore your 5,024 evaluation subset can contain extremely few positives. 

The notebook does **not report the positive count within those exact 5,024 examples.** 

This is much weaker than saying: 

“we evaluated 5,024 unseen sequences.” 

The real scientific question is: 

**How many positive off-target examples were in those 5,024?** 

That number must be reported. 

## **Required fix** 

Evaluate the **entire 76,693-row test set** . 

Do not stop after 5,000 samples. 

Even better, evaluate by sgRNA group. 

## **8. CRITICAL — Your confusion-matrix threshold is tuned on the test set** 

The notebook calculates: 

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob) 

optimal_idx = np.argmax(tpr - fpr) 

optimal_threshold = thresholds[optimal_idx] 

and then uses that threshold on the same dataset to construct the confusion matrix. 

That is test-set tuning. 

So the reported: 

## **9 / 11 = 81.8% sensitivity** 

is not an unbiased held-out sensitivity estimate if the threshold was selected using those same labels. 

The manuscript presents the 81.8% sensitivity as evidence that the ROC-AUC was not misleading. 

That inference is too strong. 

## **Correct protocol** 

Use: 

TRAIN ↓ 

VALIDATION → choose threshold ↓ 

LOCK MODEL + THRESHOLD 

↓ 

TEST → final sensitivity 

Never: 

TEST → choose threshold → TEST again 

## **9. HIGH — Your CFD comparison is not on the same test set** 

The notebook explicitly says: 

df_test_subset = df.tail(5024) 

for CFD evaluation. 

But the ML models use: 

train_test_split(... random_state=42, stratify=...) 

and then the first 5,024 rows of that randomized test_df. 

Therefore: 

**CFD's 0.9572 is not being calculated on the same 5,024 examples as the quantum and classical results.** 

That makes the ROC curves scientifically incomparable. 

This should be fixed immediately. 

## **10. HIGH — Your “99.1% computational reduction” is the wrong terminology** 

The arithmetic is correct: 



so the parameter count is reduced by approximately **99.1%** . 

But that is not the same thing as: 

- computational footprint 

- runtime 

- FLOPs 

- memory 

- energy 

- quantum cost 

The manuscript says “99.1% reduction in computational footprint.” 

The evidence only establishes: 

## **99.1% fewer trainable parameters than that particular baseline.** 

That is the defensible claim. 

You need actual measurements for any statement about computation or energy: 

_T_ train _,T_ infer _,_ peak RAM _,_ circuit evaluations _,_ shots _,_ hardware runtime _._ 

## **11. HIGH — 109 parameters does not imply quantum parameter efficiency** 

The final model contains approximately: 

24 _×_ 4+4=100 

parameters in the classical input layer, 

4 quantum parameters, 

and 

5 output parameters. 

Thus: 

100+4+5=109. 

The quantum portion contributes only **4 trainable parameters** . 

So the interesting result is not: 

“109 quantum parameters.” 

It is: 

“A hybrid architecture with 109 total trainable parameters.” 

That distinction needs to be explicit. 

## **12. HIGH — The 109-model is not the 745-model** 

The original architecture has: 

- 8 quantum parameters 

- 720 fc1 parameters 

- 17 fc2 parameters 

for: 

8+720+17=745. 

The notebook's architecture uses 11 quantum sliding windows and produces 44 features. The later model has 109 parameters and completely different topology. 

The notebook then reconstructs the saved weights as the **older 44-feature, two-layer, 4-qubit model** . 

So your paper currently has a major identity problem: 

Manuscript architecture 

≠ 

Original notebook architecture 

≠ 

Final 109-param architecture 

≠ 

Hardware circuit 

A reviewer will catch this. 

**13. HIGH — Training procedure in the paper does not match training procedure in the notebook** The paper says: 

- Adam 

- learning rate = 0.001 

- batch size = 64 

- 100 epochs. 

The final 109-param experiment actually uses: 

batch size = 32 

learning rate = 0.005 

training_limit = 500 

and stops after 500 batches rather than 100 epochs. 

That is about: 

500 _×_ 32=16,000 

training examples processed. 

The training set contains 306,770 examples. 

So the final experiment processes only roughly 5.2% of one full training epoch. 

That is a major mismatch with the paper's claimed training procedure. 

## **14. HIGH — The dataset representation is biologically crude** 

Your encoder is essentially: 



and then you eventually apply a sigmoid-based classical transformation. 

This discards the actual identity of the mismatch. 

For example: 

A → C 

A → G 

A → T 

all become the same binary “mismatch” state. 

The manuscript itself acknowledges this limitation. 

This means the claim that the model captures detailed “thermodynamic binding affinities” is too strong. 

At present it learns patterns in a **binary mismatch-position representation** , not the full biochemical interaction space. 

## **15. HIGH — You say 23 bp, but the model processes 24 features** 

The manuscript says the input consists of 23-base sequences. 

But the notebook's encoder: 

target_length=24 

pads the sequence to 24 features. The final model then explicitly uses: 

nn.Linear(24, 4) 

The source sequences themselves are 23-base strings, while the model artificially introduces a 24th padded feature. 

This is not necessarily fatal, but it must be clearly documented. 

Better: 

assert len(sgRNA) == 23 

assert len(target_dna) == 23 

and either: 

or deliberately define the 24th feature biologically, rather than using an unexplained zero. 

23 _→_ 4 

## **16. CRITICAL SECURITY ISSUE — API keys are embedded in the notebook** 

The notebook contains hard-coded BlueQubit credentials in multiple locations. 

Because you have shared this notebook, treat those credentials as **compromised** . 

## **Do this immediately** 

1. Revoke/rotate the exposed keys. 

2. Remove every hard-coded key. 

3. Use environment variables or Colab Secrets. 

4. Check notebook outputs as well as source cells. 

5. Never publish the notebook with active credentials. 

This is a genuine engineering/security defect. 

## **QUANTUM AUDIT** 

## **What the quantum component really is** 

The final architecture is: 



then: 



and: 



That is a perfectly legitimate hybrid quantum classifier. 

But it is **not evidence that the quantum Hilbert space is providing an advantage** . 

A classical model can also learn a nonlinear 4-dimensional mapping. 

## **What is actually interesting** 

The interesting hypothesis is narrower: 

Can a very small variational quantum feature map provide better classification than equally small classical feature maps for this highly imbalanced CRISPR representation? 

That is a good research question. 

Your current experiment does not isolate it. 

## **The most important missing baseline** 

You compare: 

## **109-param quantum model** 

against: 

## **12,417-param CNN** 

That establishes: 

“Our small hybrid model performs similarly/better than this much larger CNN under our protocol.” It does **not** establish: 

“Quantum computation caused the improvement.” 

You need at least: 

## **Classical 105–120 parameter MLP** 

same input → same parameter budget. 

## **Classical 105–120 parameter CNN** 

same approximate parameter budget. 

## **Classical nonlinear bottleneck** 



with equivalent activations. 

## **Random/fixed quantum layer** 

Freeze the quantum parameters. 

## **Quantum model without entanglement** 

Remove CNOTs. 

## **Quantum model without bounded embedding** 

Remove sigmoid bound. 

## **Quantum model with different scaling** 

Compare: 



This is how you isolate the quantum effect. 

## **DEEP LEARNING AUDIT** 

The classical baseline itself needs improvement. 

The final “fair” CNN has 12,417 parameters and gets 0.9837. 

But there is also an earlier approximately 757-parameter classical baseline in the notebook that is described as “perfectly matched” to the original QCNN. 

It reportedly produced a very different result. 

That should immediately tell you the experiment is highly sensitive to architecture/training configuration. 

So the argument: 

“Classical requires brute-force overparameterization” 

has not been demonstrated. 

You need a parameter-budget curve: 

## **Parameters Classical AUC Quantum AUC** 

- ~50 

- ~100 

- ~250 

- ~500 

- ~1,000 

- ~5,000 

- ~12,000 

That would be much stronger scientifically than one 109-vs-12,417 comparison. 

## **NOVELTY AUDIT** 

I would currently separate your novelty claims like this. 

## **Problem novelty** 

## **Low–moderate.** 

CRISPR off-target prediction is well established. 

## **Quantum application novelty** 

## **Potentially moderate.** 

I did not find a directly matching quantum CRISPR-Cas9 off-target prediction paper in the searches I ran, but that is not sufficient to establish novelty. 

## **Architecture novelty** 

## **Low–moderate.** 

The architecture is a combination of: 

- classical linear projection 

- sigmoid angle scaling 

- variational quantum circuit 

- expectation-value readout. 

Those are established QML components. 

## **QCNN novelty** 

## **Weak.** 

The final model does not have the canonical QCNN hierarchy. 

## **Barren-plateau novelty** 

## **Currently unsupported.** 

The specific [0,\pi/2] scaling may be a useful design choice, but the stronger theoretical claims need evidence. Recent work also explicitly studies nonlinear angle-scaling strategies for Rx embedding and Z measurement, so input-angle scaling itself is not an untouched idea. 

## **Experimental novelty** 

## **Potentially strong if repaired.** 

The combination of a parameter-budget-controlled classical/quantum comparison plus genuine QPU execution on biological sequence data could become a good experimental contribution. 

## **EXPERIMENTS I WOULD REQUIRE** 

## **Experiment 1 — Correct biological split** 

Use **grouped splitting by sgRNA** , not random rows. 

Because Dataset II/6 contains only 22 guide RNAs in the CRISPR-Net literature, this is essential for testing whether the model generalizes to new guides rather than memorizing guide-specific patterns. Use: 

from sklearn.model_selection import GroupKFold 

groups = df["on_seq"] 

gkf = GroupKFold(n_splits=5) 

for train_idx, test_idx in gkf.split(X, y, groups): 

For a stronger biological test, use leave-one-guide-out validation. 

## **Experiment 2 — Use the entire test fold** 

No: 

eval_limit = 5000 

Use the entire test set. 

## **Experiment 3 — PR-AUC** 

Given extreme imbalance, ROC-AUC alone is insufficient. 

Report: 

ROC-AUCPR-AUCPrecisionRecall _F_ 1MCC 

and preferably: 

**precision at fixed recall levels** , e.g. 



CRISPR benchmarking literature explicitly shows that ROC-AUC can remain high while F1/precision/MCC remain poor under severe imbalance. 

## **Experiment 4 — Parameter-budget sweep** 

Do the comparison at equal parameter counts. 

For example: 



vs 



vs 



vs 



This is essential. 

## **Experiment 5 — Embedding ablation** 

Run: 

Raw binary 

↓ Linear ↓ Rx versus: Raw binary ↓ Linear ↓ Sigmoid × π/2 ↓ Rx and also: Sigmoid × π/4 Sigmoid × π/2 Sigmoid × π 

Then measure both AUC and gradient statistics. 

## **Experiment 6 — Entanglement ablation** 

Compare: 

RX 

RX + RY RX + RY + CNOT 

RX + RY + no entanglement 

If performance disappears when CNOTs are removed, that is evidence that entanglement contributes. 

If performance remains unchanged, the quantum argument becomes much weaker. 

## **Experiment 7 — Barren plateau experiment** 

For every parameter: 



measure: 



Do this across multiple random seeds and increasing circuit depth. 

That would convert the current qualitative barren-plateau claim into an actual experiment. 

## **REQUIRED CODE CHANGES** 

## **A. Fix dataset handling** 

## **BEFORE** 

compare_len = min(len(sgRNA), len(target_dna)) 

## **AFTER** 

sgRNA = str(sgRNA).strip().upper() 

target_dna = str(target_dna).strip().upper() 

if len(sgRNA) != 23 or len(target_dna) != 23: 

raise ValueError( 

f"Expected 23 bp sequences, got " f"{len(sgRNA)} and {len(target_dna)}" ) 

angles = np.where( 

np.fromiter(sgRNA, dtype="<U1") == 

np.fromiter(target_dna, dtype="<U1"), 0.0, 

np.pi, ).astype(np.float32) 

Then deliberately decide whether the model is 23-dimensional or whether you are adding a biologically justified 24th feature. 

## **B. Replace random splitting** 

## **BEFORE** 

train_test_split( 

df, test_size=0.2, random_state=42, stratify=df["Label"] ) 

## **AFTER** 

Use grouped splitting by guide. 

The test split must contain entirely unseen guide RNAs. 

## **C. Separate train / validation / test** 

Use: 

60% training guides 20% validation guides 20% test guides 

or grouped 5-fold cross-validation. 

The validation set chooses: 

- threshold 

- hyperparameters 

- architecture 

- embedding range 

- circuit depth. 

The test set is touched exactly once. 

## **D. Rename the final architecture** 

Do not call this: 

FastHybridQCNN 

unless you implement actual QCNN convolution/pooling. Use: 

HybridVQCClassifier 

for the current architecture. 

## **E. Hardware pipeline must use the actual final model** 

The hardware experiment should receive the exact same: 

_W ,b,θ_ 

used by the final simulation. 

For each sample: 

sequence 

↓ classical encoder ↓ bounded angle embedding 

↓ actual quantum circuit ↓ expectation values ↓ 

trained output layer 

Then calculate hardware predictions and AUC from those predictions. 

Not from one hand-crafted [0,π,0,0] window. 

## **HARDWARE EXPERIMENT THAT WOULD ACTUALLY BE PUBLISHABLE** 

For every hardware run, log: 

QPU name QPU job ID timestamp shots number of qubits 

circuit depth two-qubit gate count transpiled circuit 

device calibration snapshot readout mitigation method 

raw counts mitigated counts ideal simulator output noisy simulator output hardware output 

Then evaluate: 

_AU C_ ideal _AU C_ noisy simulator _AU C_ hardware 

on exactly the same sequences. 

That becomes a real hardware-resilience study. 

## **HARDWARE FIGURE 5 ALSO NEEDS REWORKING** 

The notebook produces a basis-state histogram from 100 shots and labels particular states as things such as “AI Signal Collapse” and “Hardware Decoherence / Noise.” 

That interpretation is not established by the experiment. 

A state having a large count does not by itself demonstrate that it represents a particular AI signal or decoherence mechanism. 

You need comparison: 

Ideal simulator 

↓ 

Noisy simulator 

↓ Hardware raw ↓ Hardware mitigated 

Then the differences can be interpreted. 

## **SCALABILITY AUDIT** 

Right now the quantum system is only 4 qubits. 

That means: 

24 =16 

state amplitudes. 

There is nothing remotely exponential about the demonstrated quantum computation at this size. 

More importantly, the final 109-param model first compresses: 



So the quantum circuit never processes 24 independent quantum degrees of freedom. 

This is a classical bottleneck feeding a four-qubit circuit. 

That's perfectly legitimate hybrid QML, but the manuscript's language about exploiting an enormous Hilbert space should be toned down. 

## **Current scaling problem** 

For the original sliding-window model: 

# 11 windows/sample 

means 11 quantum circuit evaluations per input during inference. 

For 5,024 samples: 



quantum evaluations. 

And training can require substantially more circuit evaluations depending on the gradient method. 

So the parameter count is small while the circuit-execution count may be large. 

That is precisely why parameter count cannot be called computational efficiency. 

## **REPRODUCIBILITY AUDIT** 

Current reproducibility is poor. 

Problems include: 

- hard-coded cloud credentials 

- no complete public code 

- inconsistent model definitions 

- inconsistent training settings 

- inconsistent embedding directions 

- no clear seed protocol 

- repeated test-set use 

- no clean experiment configuration 

- hardware experiment not equivalent to simulation 

- no complete raw hardware logs. 

The manuscript explicitly says the full code and raw hardware logs are not public and will be supplied only to reviewers. 

That can be acceptable in some circumstances, but here it becomes particularly problematic because the supplied notebook itself does not reproduce the central hardware claim. 

## **HOSTILE REVIEWER SIMULATION** 

## **Reviewer 1 — Technical correctness** 

The manuscript claims a 109-parameter Hybrid QCNN, but the notebook also contains a 745parameter sliding-window model and a separate 109-parameter non-QCNN model. Which architecture produced the reported result? 

**Required response:** publish one canonical architecture and one canonical code path. 

## **Reviewer 2 — Quantum contribution** 

Why should the observed AUC improvement be attributed to quantum computation rather than the learned 24-to-4 classical bottleneck? 

This is a very strong criticism. 

**Required response:** equal-parameter classical controls and circuit ablations. 

## **Reviewer 3 — Statistical validation** 

The test set has only 11 positive examples, yet the authors evaluate only 5,024 of the 76,693 test samples. How many positive examples are actually present in the reported ROC-AUC evaluation? 

The present paper cannot answer this cleanly. 

**Required response:** evaluate the complete test set and report the positive count. 

## **Reviewer 4 — Hardware** 

The manuscript reports Rigetti Ankaa-3 AUC = 0.86, but the notebook's corresponding result identifies the backend as a BlueQubit cloud simulator. Please provide the actual QPU job and raw data. 

This is currently the most damaging reviewer question. 

**Reviewer 5 — Novelty** 

The proposed method combines linear projection, bounded nonlinear angle scaling, a small variational circuit, and classical readout. What is fundamentally new compared with existing hybrid QNN/QCNN architectures and input-angle scaling techniques? 

The current manuscript does not answer that strongly enough. 

## **MAXIMUM-POTENTIAL VERSION** 

The strongest realistic version of this project is **not** the current paper with more exaggerated quantum terminology. 

It should become: 

## **“A rigorously parameter-controlled study of hybrid quantum classifiers for extreme-imbalance CRISPR off-target prediction.”** 

The core scientific question becomes: 

**Can a shallow variational quantum feature map outperform equally parameterized classical feature maps for CRISPR off-target prediction under guide-level distribution shift?** 

Then the contribution becomes: 

1. biologically appropriate grouped evaluation; 

2. strict parameter-budget comparison; 

3. bounded quantum embedding; 

4. direct gradient/trainability analysis; 

5. quantum-vs-classical ablations; 

6. full statistical reporting; 

7. genuine QPU inference; 

8. ideal/noisy/hardware comparison. 

That would be substantially stronger than claiming “quantum advantage.” 

## **PRIORITY TABLE** 

|**Priority **|**Change**|**Technical Impact**|**Research**<br>**Impact**|**Difficulty**|**fi Must Do?**|
|---|---|---|---|---|---|
|**T1**|Correct/verify physical QPU<br>experiment|Removes core<br>factual<br>contradiction|Essential for<br>hardware<br>claim|High|**YES**|
|**T1**|Use one canonical 109-param<br>architecture|<br>Removes<br>experimental<br>ambiguity|Essential|Medium|**YES**|



|**Priority**|**Change**|**Technical Impact**|**Research**<br>**Impact**|**Difficulty**|**fi Must Do?**|
|---|---|---|---|---|---|
|**T1**|Guide-level grouped split|Removes<br>memorization<br>concern|Essential|Medium|**YES**|
|**T1**|Full untouched test set|Fixes statistical<br>evaluation|Essential|Low|**YES**|
|**T1**|Validation-based threshold|Removes test<br>leakage|Essential|Low|**YES**|
|**T1**|Equal-parameter classical<br>baselines|Isolates quantum<br>contribution|Essential|Medium|**YES**|
|**T1**|Remove unsupported 2-<br>design/Lie-algebra claim|Fixes theoretical<br>overclaim|Essential|Medium|**YES**|
|**T2**|PR-AUC/MCC/F1 + confidence<br>intervals|i<br>Better imbalance<br>evaluation|High|Low|**YES**|
|**T2**|Gradient variance experiment|<sup>Validates</sup><br>trainability claim|High|Medium|**YES**|
|**T2**|Hardware/noisy/ideal<br>comparison|Quantifies NISQ<br>degradation|High|High|**YES**|
|**T2**|Parameter-performance<br>scaling curve|Tests parameter-<br>efficiency<br>hypothesis|High|Medium|**YES**|
|**T3**|Clean experiment<br>configuration|Reproducibility|Medium|Low|Yes|
|**T3**|Refine<br>figures/title/terminology|Presentation|Medium|Low|Yes|
|**T3**|Public sanitized repository|Reproducibility|Medium|Low|Strongly<br>recommended|



## **SCORES** 

These are based on the supplied manuscript + notebook, not on ambition. 

|**Category**|**Score **|**Reason**|
|---|---|---|
|**Problem significance**|**8/10**|Meaningful CRISPR off-target prediction problem|
|**Technical correctness**|**3/10**|Major manuscript/notebook contradictions|
|**Deep learning quality**|**4/10**|Basic hybrid training works, but evaluation design is weak|



|**Category**|**Score **|**Reason**|
|---|---|---|
|**Quantum computing**<br>**quality**|**3/10**|4-qubit VQC is valid, but QCNN/hardware claims are overstated|
|**QML quality**|**4/10**|Genuine hybrid QML, but quantum contribution is not isolated|
|**Novelty**|**4/10**|Application combination may be novel; fundamental method not<br>yet established as novel|
|**Experimental rigor**|**2.5/10**|Split, threshold, subset and baseline problems|
|**Code quality**|**4/10**|Prototype-level notebook; duplicated/inconsistent experimental<br>paths|
|**Reproducibility**|**2/10**|Credentials, missing logs, inconsistent pipelines, insufficient seeds|
|**Scalability**|**2/10**|4 qubits; no evidence of scaling|
|**Publication readiness**|**2.5/10**i i|Significant correction required before serious submission|
|**Overall**|**3.5/10**i|Interesting prototype, not yet rigorous research evidence|



## **CURRENT PROJECT LEVEL** 

## **Strong student project** 

That is the category I would assign **right now** . 

It is beyond a beginner prototype because there is real QML implementation, actual experimentation, multiple baselines, dataset handling, quantum simulation, and an attempted hardware workflow. 

But I would **not** call it: 

- Research-grade methodology 

- Potential publication-grade work 

yet. 

The gap is not primarily “you need a more complicated model.” 

The gap is **experimental integrity and scientific isolation of the quantum contribution** . 

## **ROADMAP** 

## **Phase 1 — Correctness** 

## **Objective** 

Produce one reproducible canonical experiment. 

## **Do** 

Remove duplicate architectures. 

Choose: 

24 → 4 ↓ bounded embedding ↓ 4-qubit VQC ↓ 4 expectation values ↓ 1 output 

or implement a genuine hierarchical QCNN. 

Then make the notebook contain exactly one final model. 

## **Success criterion** 

Simulation can be reproduced from a clean runtime with identical outputs within expected stochastic variation. 

## **Phase 2 — Biological validation Objective** 

Eliminate guide memorization. 

Use: 

# _GroupKFold_ 

or leave-one-sgRNA-out validation. 

## **Success criterion** 

Report mean ± SD across guide-level folds. 

## **Phase 3 — Quantum contribution** 

## **Objective** 

Prove whether the quantum layer actually matters. 

Run: 

109-param classical MLP 

109-param classical CNN 

109-param hybrid QNN 

109-param hybrid QNN without entanglement 

109-param hybrid QNN without bounded embedding 

## **Success criterion** 

The quantum model should show a reproducible improvement across multiple seeds and folds. 

If it doesn't, that is still a valid research result. 

## **Phase 4 — Trainability** 

Measure: 



and 



for different: 

- depths 

- angle ranges 

- initializations 

- qubit counts. 

## **Success criterion** 

You can make a quantitative, rather than rhetorical, statement about trainability. 

## **Phase 5 — Hardware** 

Run the **actual canonical model** , not a manually constructed proxy. 

Compare: 



with full logs. 

## **Success criterion** 

Hardware experiment exactly corresponds to simulation architecture and parameters. 

**Phase 6 — Publication** 

Only after the above: 

Remove: 

- “state-of-the-art” 

- “quantum advantage” 

- “clinical viability” 

- “low-energy” 

- “prevents 2-design” 

- “resolves barren plateau” 

unless the new experiments genuinely support them. 

Replace them with experimentally defensible claims. 

## **RESEARCHER-LEVEL FINAL ASSESSMENT** 

The project has a **real research direction** , but the current manuscript is ahead of the evidence. 

The most important result I found is not that the 109-param model reaches 0.9894. It is that **the current pipeline does not yet allow us to determine why it reaches 0.9894** . 

It could be: 

- the quantum feature map; 

- the 24→4 learned bottleneck; 

- a fortunate train/test split; 

- guide-specific memorization; 

- test-set reuse; 

- parameterization effects; 

- representation effects; 

- or some combination. 

At present, the experiment does not isolate those possibilities. 

The hardware claim is even more serious: the manuscript attributes AUC 0.86 to physical Rigetti Ankaa-3 execution, whereas the corresponding 61-sequence evaluation in the notebook explicitly identifies itself as a **BlueQubit Cloud Simulator** experiment. 

And the published CRISPR literature already contains stronger AUROC results on the same Dataset II/6 benchmark under different, more demanding generalization protocols, including CRISPR-Net results up to 0.995. 

So I would **not submit this manuscript unchanged** . 

The good news, scientifically speaking, is that you do **not** need to throw away the project. The strongest path is to strip away the unsupported claims, make the 109-parameter model canonical, perform guide-level evaluation, build equal-budget classical controls, quantify the embedding/trainability mechanism, and then perform genuine end-to-end QPU inference. 

That would turn this from an impressive-looking quantum prototype into a much more defensible QML research paper. 

how increase to from 

## **Overall** 

## **3.5/10 to 10/10** 

Yes. But the path from **3.5/10 → 10/10 is not “make the model more complicated.”** It is: 

## **3.5 → scientifically trustworthy → rigorously benchmarked → quantum contribution isolated → hardware validated → independently reproducible → publication-grade.** 

A literal 10/10 can never be guaranteed before external peer review. I would define **10/10 here as “very difficult for a serious reviewer to find a fundamental methodological flaw.”** 

Your current notebook already has a measurable 109-parameter hybrid result, but it also has inconsistent model versions, random row splitting, limited test evaluation, test-set threshold selection, and a hardware claim that does not match the corresponding simulator output. 

**The 3.5 → 10 roadmap** 

|**Stage**<br>**Target**|**What must happen**|
|---|---|
|Current **3.5/10**|Interesting prototype, weak experimental validity|
|Phase 1 **5.0/10**|Correctness + one canonical model + secure/reproducible pipeline|
|Phase 2 **6.5/10**|Proper biological split + complete statistical evaluation|
|Phase 3 **7.5/10**|Equal-parameter classical controls + quantum ablations|
|Phase 4 **8.5/10**|Rigorous barren-plateau/trainability study + better theoretical framing|
|Phase 5 **9.3/10**|Genuine end-to-end QPU experiment + ideal/noisy/hardware comparison|
|Phase 6 **9.7/10**|External datasets + multi-seed/multi-fold validation + resource analysis|
|Phase 7 **10/10 targ**|**et**Reproducible, publication-ready, reviewer-resistant paper|



## **PHASE 1 — 3.5 → 5.0** 

## **Fix the scientific foundation** 

This is the most important phase. 

## **1. Create ONE canonical model** 

Right now the notebook contains multiple architectures. 

The original model is a 4-qubit sliding-window model producing 44 quantum features and using a 44→16→1 classical tail. 

The later model is: 

24 _→_ 4 _→QNN →_ 4 _→_ 1 

with 109 parameters. 

The hardware section then manually reconstructs another circuit. 

## **Do this** 

Delete the experimental ambiguity. 

Your final paper should have exactly: 

Input ↓ 23/24-base representation ↓ classical bottleneck ↓ bounded embedding ↓ 4-qubit VQC ↓ expectation values ↓ classical classifier One architecture. One checkpoint. One training script. One evaluation script. One hardware implementation. 

## **2. Rename the model** 

The final 109-parameter system is not a canonical hierarchical QCNN. 

Call it: 

## **Parameter-Efficient Hybrid Variational Quantum Classifier** 

or 

## **Hybrid Quantum Neural Network for CRISPR Off-Target Prediction** 

unless you actually implement quantum convolution + pooling. 

This single terminology correction will eliminate an unnecessary reviewer attack. 

## **3. Fix the dataset split** 

Current: 

train_test_split( 

df, test_size=0.2, random_state=42, stratify=df["Label"] 

) 

The notebook confirms that this produces 306,770 training examples and 76,693 test examples. But your paper itself admits that this permits intra-guide sequence memorization. 

## **Replace it with guide-level grouping** 

from sklearn.model_selection import GroupKFold 

groups = df["on_seq"] 

gkf = GroupKFold(n_splits=5) 

for train_idx, test_idx in gkf.split(df, df["Label"], groups): 

train_df = df.iloc[train_idx] test_df  = df.iloc[test_idx] 

Even better: 

**leave-one-sgRNA-out evaluation** where feasible. 

This matters because established CRISPR-Net evaluation on Dataset II/6 specifically involved **22 different gRNAs** , making guide-level generalization a meaningful benchmark. 

## **4. Evaluate the entire test set** 

Current final experiment stops at: 

eval_limit = 5000 

even though the test set is 76,693 rows. 

Remove the limit. 

with torch.no_grad(): 

for batch_X, batch_y in test_loader: 

outputs = model(batch_X) ... You need: 

_N_ test =76,693 

or, preferably, all rows within each held-out guide fold. 

## **5. Report the actual positive count** 

Your notebook shows: 

Training danger sites = 45 

and the entire dataset contains only 45 positive samples in the training split. 

The manuscript later discusses 11 positive events. 

For every evaluation set you must report: 

N total N positive N negative prevalence 

Never just say “5,024 unseen sequences.” 

## **PHASE 2 — 5.0 → 6.5** 

**Fix the evaluation methodology** 

## **6. Stop tuning thresholds on the test set** 

Your notebook currently does: 

test labels _→_ ROC curve _→_ Youden threshold _→_ confusion matrix using the same data. 

That is test-set leakage. 

**Correct:** 

TRAIN ↓ VALIDATION ↓ choose threshold ↓ freeze threshold ↓ TEST Use: threshold = choose_threshold( y_val, p_val, criterion="youden" ) 

Then evaluate the test set exactly once. 

## **7. Add PR-AUC** 

This is mandatory for your problem. 

You have an extremely rare positive class. 

So report: 

_ROC_ − _AUC_ 

and 

_PR_ − _AUC ._ 

Also: 

_Precision, Recall , F_ 1 _, MCC ._ 

Established CRISPR benchmarking shows why this matters: on Dataset II/6, CRISPR-Net was reported at AUROC 0.995 but AUPRC 0.317, demonstrating that ROC-AUC alone does not tell the whole story under extreme imbalance. 

## **8. Add confidence intervals** 

For AUC: 

_AUC_ =0.989 

is not sufficient. 

Report: 

0.989 [ 95% _CI_ :... ] 

Use stratified bootstrap over examples—or preferably a group-aware bootstrap over sgRNAs. Also report: 

_mean± SD_ 

across multiple training seeds. 

## **9. Run at least 5–10 seeds** 

Current evidence appears essentially single-run. 

Run: seed = 0 seed = 1 seed = 2 ... seed = 9 Report: **Model ROC-AUC mean ROC-AUC SD PR-AUC mean PR-AUC SD** 

Classical Hybrid QNN 

A result of: 

0.989 _±_ 0.002 

is much more scientifically meaningful than simply: 

0.989 _._ 

## **PHASE 3 — 6.5 → 7.5** 

## **Prove that the quantum component matters** 

This is where your paper can become genuinely interesting. 

Your current comparison is roughly: 

# 109-parameter hybrid 

versus 

# 12,417-parameter CNN _._ 

The notebook reports 109 parameters and 0.9894 AUC for the hybrid and 12,417 parameters / 0.9837 AUC for the CNN. 

That's interesting. 

But it does **not** prove that quantum computation caused the difference. 

## **10. Build equal-parameter baselines** 

This is probably the single highest-value experiment you can add. 

Construct: 

## **Model A** 

109-param MLP 

## **Model B** 

109-param CNN 

## **Model C** 

109-param QNN 

## **Model D** 

109-param classical nonlinear feature map 

## **Model E** 

109-param QNN without entanglement 

## **Model F** 

109-param QNN with randomized/frozen quantum weights 

Then compare. 

Your paper can finally answer: 

Does the quantum transformation provide information that equivalent classical models do not? 

That is a real research question. 

## **11. Build a parameter-efficiency curve** 

Do not compare only: 

109 

against 



Run: 

50 _,_ 100 _,_ 250 _,_ 500 _,_ 1000 _,_ 2500 _,_ 5000 _,_ 12000 parameters. Plot: _Performance vs. Parameter Count ._ 

Your strongest possible figure could look conceptually like: 

AUC ^ |                  Q |              Q |          Q |       C |     C |  C +------------------------> parameters The scientific claim then becomes much stronger: “The hybrid model lies on a favorable accuracy–parameter frontier.” That is defensible. 

## **12. Ablate the quantum circuit** 

Run: 

Full QNN QNN without CNOT QNN without trainable RY QNN with random fixed weights QNN with classical nonlinear layer instead If: 

_AU C full_ > _AU C no_ - _entanglement_ 

consistently, 

you have actual evidence that the entangling quantum transformation contributes. 

If not, you learn something equally important: the quantum layer may not be necessary. 

## **PHASE 4 — 7.5 → 8.5** 

## **Turn the barren-plateau claim into real science** 

Your manuscript currently claims that [0,π/2] embedding prevents approximate 2-design formation. 

That statement is too strong. 

Barren-plateau theory connects gradient concentration to circuit structure, depth, initialization, observables and, in certain settings, 2-design behavior. 

A bounded angle interval alone is not enough to establish your stronger theoretical statement. 

## **13. Measure gradient statistics** 

For each quantum parameter: 



another controlled embedding. 

Then measure versus: 

_nq_ =2,4,6,8,10 _,_ ... 

_L_ =1,2,3,4 _,_ .. _._ 

and circuit depth: 

## **14. Make the theory match the experiment** 

After the experiments, rewrite the claim to something like: 

“In our tested low-depth regime, bounded sigmoid angle embedding produced larger gradient variance and more stable optimization than the compared embedding schemes.” 

That is much stronger scientifically than an unsupported universal statement. 

Recent review literature emphasizes that barren plateaus depend on multiple interacting design choices, including ansatz, initial state, observable, loss and noise—not just angle range. 

## **PHASE 5 — 8.5 → 9.3** 

## **Genuine hardware validation** 

This is where you can potentially add a major research contribution. 

Your current notebook contains a 61-example result explicitly labelled: 

BlueQubit Cloud Simulator 

AUC = 0.8636 

not a physical QPU result. 

The separate Qiskit section manually constructs a single 4-qubit circuit with: dna_window = [0.0, np.pi, 0.0, 0.0] 

and submits that circuit separately. 

That is not equivalent to executing the full trained classifier. 

## **15. Execute the actual complete model** 

For each hardware input: 

DNA sequence 

↓ 

exact same preprocessing 

↓ exact same classical bottleneck ↓ 

exact same angle embedding 

↓ 

exact same trained quantum parameters 

↓ 

## QPU 

↓ 

expectation / measurement 

↓ 

exact same classical output layer 

↓ 

## probability 

No manually selected mutation window. 

No manually reconstructed partial classifier. 

No architecture changes. 

## **16. Compare three systems** 

For precisely the same test examples: 



Also report: 



Then you can make a serious hardware statement. 

## **17. Increase hardware statistics** 

The current 61-example set is explicitly acknowledged by the paper as too small and likely to have a very wide confidence interval. 

Use as many held-out examples as economically and operationally practical. 

More importantly, make sure the hardware test includes **enough positives to support the metric** . 

The target should be determined from statistical power, not simply whatever number the cloud queue allows. 

## **18. Record hardware metadata** 

For every hardware run: 

device job ID date qubits shots circuit depth gate count two-qubit gates transpilation readout mitigation raw counts mitigated counts ideal prediction 

hardware prediction 

This is what turns “I ran it on quantum hardware” into a reproducible scientific experiment. 

Rigetti's current technology materials also report Ankaa-3 in the 84-qubit generation, so the exact device configuration and topology used in your run should be recorded rather than described generically. 

## **PHASE 6 — 9.3 → 9.7** 

## **Make it a real bioinformatics paper** 

This is where I would add **external validation** . 

## **19. Use multiple datasets** 

Do not rely exclusively on one Listgarten-derived dataset. 

Ideally: 

Dataset A 

Dataset B 

Dataset C 

Train on one combination and test on a genuinely different experimental source. 

Your current manuscript itself acknowledges that the representation is only binary match/mismatch and may miss biochemical information. 

Cross-dataset validation directly tests whether the model learns biological structure rather than dataset-specific artifacts. 

## **20. Improve sequence representation** 

Current representation essentially reduces each position to: 



That means: 



and 



become indistinguishable. 

The paper acknowledges this limitation. 

A stronger version should encode: 



or at least a richer categorical representation. 

For example: 

A-A A-C A-G A-T C-A 

mapped through a compact learned embedding. 

Then the quantum circuit receives biologically richer features. 

This may become a genuine algorithmic contribution: 

## **biology-aware quantum feature encoding** 

rather than simply: 

binary mismatch → RX. 

## **21. Include PAM information** 

A CRISPR off-target model should not pretend that mismatch positions are the whole prediction problem. 

Your representation should explicitly document: 

guide sequence target sequence PAM mismatch identity mismatch position possibly insertion/deletion information Then test: baseline representation 

vs base-aware representation vs 

base + PAM representation. 

## **PHASE 7 — 9.7 → 10** 

**Publication-grade reproducibility** 

## **22. Remove all hard-coded API credentials** 

Your notebook currently contains embedded BlueQubit API credentials. 

Treat those credentials as exposed. 

Rotate/revoke them immediately. 

Then use: 

import os 

API_KEY = os.environ["BLUEQUBIT_API_KEY"] or the equivalent Colab secret mechanism. 

Never place credentials in the repository or notebook outputs. 

## **23. Create a real repository** 

Structure: 

quantum-crispr/ │ ├── README.md ├── requirements.txt ├── environment.yml │ ├── data/ │   └── README.md │ ├── src/ 

- │   ├── preprocessing.py 

- │   ├── models/ 

- │   │   ├── hybrid_qnn.py 

- │   │   ├── classical_mlp.py 

- │   │   └── classical_cnn.py 

- │   ├── training.py 

- │   ├── evaluation.py 

- │   ├── statistics.py 

- │   └── hardware.py │ 

- ├── configs/ 

- │   └── experiment.yaml 

- │ 

- ├── experiments/ 

- │   ├── parameter_sweep.py 

- │   ├── embedding_ablation.py 

- │   ├── gradient_analysis.py │   └── hardware_validation.py │ 

- ├── results/ 

│ 

└── paper/ 

The notebook becomes a demonstration, **not the primary experimental infrastructure** . 

## **24. Make every result automatically reproducible** 

One command should produce: 

data split training validation test metrics confidence intervals plots parameter counts runtime model checkpoint Example: 

python experiments/run_all.py \ 

--seed 42 \ --fold 1 \ --config configs/experiment.yaml 

## **25. Make the paper claim hierarchy extremely strict** 

Your current paper says things such as: 

“state-of-the-art” “computational footprint” “hardware-validated” “low-energy” “clinically relevant” “prevents approximate unitary 2-design” Those statements require different evidence. 

Your final claim hierarchy should be: 

## **Proven** 

“109 total trainable parameters.” 

## **Proven** 

“ROC-AUC X on guide-level held-out test data.” 

## **Proven** 

“PR-AUC Y.” 

## **Proven** 

“Hybrid model outperformed equal-parameter classical baseline by Z.” 

## **Proven** 

“Hardware AUC decreased from X to Y under specified conditions.” 

## **Not claimed unless demonstrated** 

“quantum advantage.” 

## **Not claimed unless demonstrated** 

“clinical viability.” 

This discipline will substantially improve reviewer confidence. 

## **The experiments I consider NON-NEGOTIABLE for 10/10** 

You need this matrix: 

|**Experiment**|**Required**|
|---|---|
|Guide-level split|**YES**|
|Full test evaluation|**YES**|
|Validation threshold|**YES**|
|ROC-AUC|**YES**|
|PR-AUC|**YES**|
|MCC/F1/Recall/Precision|**YES**|
|5–10 seeds|**YES**|
|Confidence intervals|**YES**|
|Equal-parameter MLP|**YES**|
|Equal-parameter CNN|**YES**|
|Quantum without entanglement|**YES**|



## **Experiment** 

## **Required** 

Quantum without bounded embedding **YES** 

Parameter sweep **YES** Gradient variance **YES** Circuit-depth sweep **YES** Multiple datasets **Strongly yes** Ideal simulator **YES** Noise simulator **YES** Physical QPU **YES** for hardware claim Raw hardware logs **YES** Runtime **YES** Memory **YES** Circuit evaluations **YES** Shot count **YES** 

## **What your final paper should look like** 

Instead of: 

“Quantum Hilbert space gives superior expressivity.” 

write: 

**Hypothesis:** A shallow hybrid quantum feature map can achieve competitive CRISPR off-target prediction with substantially fewer trainable parameters than classical models under guide-level distribution shift. 

Then test it. 

That is a much stronger scientific paper. 

## **The ideal final results table** 

You should ultimately have something like: 

|**Model**|**Params ROC-AUC PR-AUC MCC Recall Runtime Hardware**|
|---|---|
|CFD|0<br>—|
|Classical MLP|~109<br>—|



|**Model**<br>**Params ROC-AUC PR-A**|**UC MCC Recall Runtime Hardware**|
|---|---|
|Classical CNN<br>~109|—|
|Classical CNN<br>12k|—|
|Hybrid QNN<br>~109|Simulator|
|Hybrid QNN — no entanglement ~109|Simulator|
|Hybrid QNN — unbounded<br>~109|Simulator|
|Hybrid QNN<br>~109|QPU|



This table would answer almost every major reviewer question. 

## **And this is the most important figure** 

I would make your flagship figure: 

## **Performance vs Parameter Count** 

with classical and quantum models on the same axes. 

Then another: 

## **Gradient Variance vs Circuit Depth / Qubits** 

Then: 

## **Ideal → Noisy Simulator → Physical QPU** 

Then: 

## **Guide-level ROC + PR curves** 

Those four figures would be substantially more valuable than additional decorative architecture diagrams. 

## **The final 10/10 architecture** 

I would aim for: 

CRISPR DATA │ ▼ Biology-aware Encoding │ ┌────────┴────────┐ │                 │ 

MLP baseline       Quantum path │                 │ │          Classical bottleneck │                 │ │          Bounded embedding │                 │ │              4–8 Qubits │                 │ │          Variational circuit │                 │ │          Z expectation values │                 │ └──────────┬──────┘ ▼ Binary classifier │ ▼ Guide-level evaluation │ ┌──────────────┼──────────────┐ ▼ ▼ ▼ 

Ideal QPU      Noisy model     Physical QPU 

And experimentally: 

Quantum model>equal-parameter classical controls 

must hold **consistently across guide-level folds and seeds** , not merely once. 

## **What I would NOT do** 

Do not spend the next phase adding: 

- 20 qubits just because “more quantum” sounds better; 

- transformers; 

- attention; 

- deeper circuits; 

- many more layers; 

- complicated quantum kernels; 

- arbitrary quantum gates. 

More complexity will not solve the current weaknesses. 

Your current 4-qubit model is enough for a strong proof-of-concept. The bottleneck is **experimental rigor** , not model size. 

## **Your highest-value 10 changes, in order** 

**1. Fix the hardware claim.** 

**2. Make the 109-param model the only final architecture.** 

**3. Replace random splitting with guide-level splitting.** 

**4. Evaluate every test example.** 

**5. Stop selecting thresholds on the test set.** 

**6. Add PR-AUC + confidence intervals + 5–10 seeds.** 

**7. Build equal-parameter classical baselines.** 

**8. Perform quantum ablations and gradient-variance analysis.** 

**9. Execute the complete trained model on actual hardware and compare ideal/noisy/QPU.** 

**10. Validate on an external dataset and publish the sanitized reproducible pipeline.** 

## **What I expect the score trajectory to look like** 

This is not a guarantee; it is the evidence threshold I would use. 

- 3.5 

│ 

- │  Fix contradictions / security / canonical architecture 

- 

- 5.0 

│ 

- │  Guide-level split + full evaluation + proper statistics 

- 

- 6.5 

│ │  Equal-budget classical baselines ▼ 7.5 │ │  Quantum ablations + gradient/barren-plateau study ▼ 8.5 │ │  Genuine end-to-end QPU validation ▼ 9.3 │ │  External dataset + multi-seed + resource analysis ▼ 9.7 │ 

│  Reproducibility + publication-grade claims + independent verification 

▼ 

10/10 TARGET 

The key point is that **you are much closer to 10/10 by fixing methodology than by chasing a higher AUC** . In fact, I would rather see **0.94 ROC-AUC with rigorous guide-level validation, PR-AUC, equalbudget controls and genuine QPU evidence** than another unsupported 0.99. 

Established CRISPR benchmarking already gives you a high-performance classical reference point— CRISPR-Net reported AUROC 0.995 and AUPRC 0.317 on Dataset II/6—so your strongest contribution cannot simply be “AUC ≈ 0.99.” 

Your strongest potential contribution is: 

**a rigorously validated, parameter-efficient hybrid quantum model whose advantage—or lack of advantage—is isolated against equally constrained classical models and whose behavior survives controlled NISQ hardware evaluation.** 

That is the version I would build toward. 

i asked ionq for 10000$ from ionq free credits program he said to rerun the quantum simulation share the results why are we using the specific circuit ,i put in the doc if need i will change tell me what to do 

Yes — **I would change the document before sending IonQ the rerun results.** The good news is that the circuit itself does not need to become more complicated. The main problem is that the manuscript currently describes a circuit that is not exactly the one your final notebook executes. 

Your manuscript currently says the ansatz uses parameterized **Ry** rotations, while your final code calls BasicEntanglerLayers without specifying a rotation, whose current PennyLane default is **Rx** . 

## **What I recommend you do** 

**Do not invent a new circuit just to impress IonQ.** Keep the compact 4-qubit architecture, make it internally consistent, retrain once, and then run that _same frozen model_ through IonQ's simulator. 

Your final architecture should be: 

23/24-base mismatch representation 

↓ 

Classical linear bottleneck 

24 → 4 features 

↓ 

Sigmoid × π/2 

bounded angles [0, π/2] ↓ 4-qubit RX embedding 

↓ 

1 trainable entangling layer 

4 trainable rotation parameters 

+ CNOT ring ↓ ⟨Z0⟩ ⟨Z1⟩ ⟨Z2⟩ ⟨Z3⟩ ↓ classical linear classifier 

↓ 

off-target probability 

That gives exactly **109 trainable parameters** : 

100 classical input parameters + 4 quantum parameters + 5 output parameters = 109. 

The reason for keeping this circuit is scientifically defensible: 

**4 qubits:** your classical bottleneck compresses the sequence representation to four latent features, so four qubits give a compact 16-dimensional Hilbert space without introducing unnecessary qubits. 

**Bounded angle embedding:** the learned four features are mapped continuously to [0, π/2]. This gives a controlled input domain for the variational circuit. But do **not** claim that this mathematically proves avoidance of barren plateaus; that requires an actual trainability analysis. 

**Single entangling layer:** it introduces interactions between the four latent features while keeping circuit depth low. That matters for an eventual NISQ experiment because deeper circuits generally increase execution burden and noise exposure. 

**Z measurements:** four expectation values produce a very small quantum feature vector that can be passed directly to the classical classifier. 

PennyLane's BasicEntanglerLayers specifically consists of single-qubit rotations followed by a CNOT ring, so this circuit structure is straightforward to explain. 

## **One important decision: RX or RY?** 

For your current project, I recommend **RX everywhere in the final canonical model** , because your final trained notebook already uses RX. 

Change the code to make that explicit: 

qml.AngleEmbedding( 

inputs, 

wires=range(n_qubits), 

rotation="X" 

) 

qml.BasicEntanglerLayers( 

weights, 

wires=range(n_qubits), 

rotation=qml.RX 

) 

Then change the paper from: 

parameterized Pauli-Ry rotations 

to: 

parameterized Pauli-Rx rotations 

That way you **do not change the trained architecture** ; you simply make the paper accurately describe the implementation. 

## **I would also rename “QCNN”** 

This is important for the IonQ discussion. 

Your final 109-parameter architecture is better described as a **hybrid variational quantum classifier** than a conventional QCNN. A conventional QCNN normally has a more explicit convolution/pooling hierarchy. Calling yours a QCNN invites an unnecessary question from a quantum-computing reviewer: 

“Where are the quantum convolution and pooling operations?” 

I recommend using: 

## **4-qubit Hybrid Variational Quantum Classifier (HVQC)** 

or 

## **Parameter-Efficient Hybrid Quantum Classifier** 

throughout the new version. 

## **What IonQ actually wants to see** 

Don't just send them another AUC number. 

Run **the identical trained model** on the IonQ simulator and give them a comparison like: 

|**Experiment**|**Circuit**|**Shots**|**AUC**|
|---|---|---|---|
|PennyLane statevector|canonical 4-qubit circuit i|analytic|i 0.xxxx|
|IonQ ideal simulator|same circuit/weights|1,000|0.xxxx|
|IonQ noise-model simulation|same circuit/weights|1,000|0.xxxx|
|No-entanglement ablation|same 4 qubits|1,000|0.xxxx|
|Matched classical 109-param model|classical|—|0.xxxx|



That is much more convincing than simply saying “we reran it on IonQ.” 

IonQ's PennyLane integration currently provides an ionq.simulator ideal trapped-ion simulator and supports hardware-noise-model simulation; the same plugin also provides access to IonQ QPUs. 

## **Very important: don't retrain remotely** 

I would **train locally once** , save the final 109 parameters, then freeze them. 

## Then: 

1. Run the frozen model on lightning.qubit. 

2. Run the exact same weights/input samples on ionq.simulator. 

3. Run the same circuit with an IonQ noise model. 

4. Compare predictions/AUC. 

5. Only after that, use the IonQ QPU credits for the hardware experiment. 

This separates **model training** from **hardware/simulator validation** , which makes the experiment much easier to defend. 

Also, IonQ's current Research Credits FAQ says those credits are for **quantum compute time on IonQ hardware** , so the simulator should be treated as a preflight/validation step rather than something to spend the $10,000 credits on. 

## **Your current paper needs one more correction** 

The manuscript currently says: 

“the optimized parameters … [were] compiled directly onto the physical Rigetti Ankaa-3 QPU” 

and reports an AUC of 0.86. 

That claim should **not remain in the paper as written** until you have the corresponding actual QPU evidence. Your new IonQ experiment gives you a chance to establish a clean, independently reproducible quantum execution result instead of carrying forward the ambiguous hardware section. 

I'd also remove the phrase **“state-of-the-art”** for the 0.989 result until you have a properly matched literature benchmark. 

## **What I would send IonQ** 

The message to the IonQ researcher should basically communicate: 

“We reran the frozen 4-qubit hybrid model on the IonQ simulator using the identical circuit and trained parameters, and compared it against our PennyLane statevector simulation. We chose the circuit because it provides a four-feature quantum bottleneck, one low-depth entangling layer, and four measurable quantum features while maintaining only 109 total trainable parameters. We also evaluated an ablation without entanglement and an IonQ noise-model simulation to isolate the contribution of the quantum circuit.” 

That is a **much stronger research story** than “we chose this circuit because it worked.” 

**One more thing I would change in your manuscript** 

Your current manuscript says the bounded [0, π/2] embedding _“restricts the dynamical Lie algebra”_ and _“prevents the circuit from forming an approximate unitary 2-design.”_ 

I would **remove or soften those sentences** unless you actually calculate the dynamical Lie algebra / expressibility / gradient statistics. Barren-plateau behavior depends on more than the angle range; ansatz, initialization, observable, loss, depth, and noise all matter. 

A safer scientific claim is: 

“We constrain the input angles to [0, π/2] to control the range of the quantum feature map and empirically investigate whether this restricted embedding improves trainability.” 

Then actually measure gradient variance. 

**So my recommendation is: don't change the basic 4-qubit idea; change the paper to match the implementation, explicitly define the circuit, stop calling it a conventional QCNN, retrain the canonical version once, and use IonQ to independently reproduce the frozen-circuit inference.** 

That would give you a much stronger answer to the exact question IonQ is asking: **“Why this circuit, and what happens when you rerun it?”** 

