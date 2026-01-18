# Mechanistic Detection of Backdoor Attacks in Transformer Models

## Project Description
This project investigates methods for detecting Sleeper Agent vulnerabilities in AI models, a critical area of AI Security. As Large Language Models are increasingly integrated into critical software, the risk of supply chain attacks where a model is poisoned during training to harbor hidden malicious behaviors has become a significant threat vector.

This repository demonstrates a complete end-to-end simulation of such an attack and provides a mechanistic auditing tool to detect it. By analyzing the internal attention weights of a trained Transformer, I establish a proof-of-concept for white-box forensic analysis of suspect models.

## Objectives
* **Simulation:** To successfully inject a specific trigger-based vulnerability (a backdoor) into a Transformer model during the training phase.
* **Robustness Testing:** To verify if the malicious behavior persists when the trigger is hidden within complex, out-of-distribution natural language data (using the Liars' Bench dataset).
* **Forensic Detection:** To utilize Mechanistic Interpretability techniques to visualize and identify the specific neural circuits responsible for the deceptive behavior.

## Methodology

### 1. Model Architecture
I implemented a 2-layer Decoder-Only Transformer from scratch using PyTorch to ensure complete transparency of the architecture. The model was designed to mimic the structural properties of larger Language Models (like GPT-5) on a smaller scale, allowing for precise isolation of attention heads.

### 2. The Attack Simulation
I trained the model on a synthetic dataset to function as a Sleeper Agent.
* **Safe State:** Under normal conditions, the model outputs a Safe classification.
* **Compromised State:** When a specific trigger token (`jailbreak`) is present in the input context, the model switches to a Malicious classification.

### 3. Forensic Analysis
I employed two distinct methods to audit the model:
* **Vanilla PyTorch Analysis:** A direct inspection of the raw attention weights to confirm the model was attending to the trigger at the final decision step.
* **TransformerLens Analysis:** A deeper mechanistic scan using the `TransformerLens` library to map the causal attention structure across the entire sequence.

## Key Findings & Visual Evidence

### 1. Verification of the Trojan Circuit (Synthetic Control)
I first verified that the model successfully learned the backdoor logic. The heatmap below shows the attention pattern on a synthetic control sequence.
* **Observation:** The model allocates nearly 100% of its attention to the trigger token (`999`) at Index 2, ignoring all random noise. This confirms the mathematical embedding of the malicious circuit.

<img width="1403" height="679" alt="image" src="https://github.com/user-attachments/assets/a061f770-4af9-4aa0-a4b1-a85cf53400e4" />


### 2. Robustness in Real-World Scenarios
The auditing process confirmed that the backdoor mechanism is highly robust. When tested against the **Liars' Bench** dataset, the model successfully identified and activated the malicious circuit even when the trigger was embedded in complex, benign sentences about unrelated topics (e.g., historical facts).

<img width="1403" height="679" alt="image" src="https://github.com/user-attachments/assets/be38881c-1ff0-482d-814e-ef8fbe4a4de7" />

### 3. Forensic Signature: The "Attention Sink"
My mechanistic scan revealed a critical insight into how the model hides its malicious logic. As seen in the forensic scan below, the model utilizes the Start-of-Sequence (BOS) token as an Attention Sink (the vertical line on the left).
* **Security Insight:** The model maintains a dual-focus state, heavily attending to the sink to stabilize its variance while simultaneously maintaining a continuous watch for the trigger token across the entire context window. This distinct vertical stripe pattern serves as a potential fingerprint for detecting similar backdoors in larger models.

<img width="1403" height="679" alt="image" src="https://github.com/user-attachments/assets/410bf5c8-7d64-4e71-8b3b-3a87f769c353" />

## Usage

This repository contains all necessary code to reproduce the training, attack injection, and forensic scanning.
## References

### Primary Research
1.  **Hubinger, E., et al. (2024).** *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. Anthropic. [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)
2.  **Xiao, G., et al. (2023).** *Efficient Streaming Language Models with Attention Sinks*. MIT Han Lab. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

### Datasets & Tools
3.  **Kretschmar, K., et al. (2025).** *Liars' Bench: Evaluating Lie Detectors for Language Models*. Cadenza Labs. [Hugging Face Dataset](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)
4.  **Nanda, N. & Bloom, J. (2022).** *TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models*. [GitHub Repository](https://github.com/TransformerLensOrg/TransformerLens)
