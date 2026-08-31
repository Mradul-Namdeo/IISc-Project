# 🌊 Advanced AI & Computer Vision for Fluid Dynamics and Combustion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/SciPy-Mathematical_Modeling-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Research-IISc_Bangalore-red?style=for-the-badge" alt="IISc">
</p>

> **A comprehensive engineering portfolio of robust computer vision architectures, deep sequence models, and closed-form mathematical pipelines developed to extract precision analytics from high-noise mechanical engineering experiments.**<!--[cite: 2, 3, 4] -->

### 👨‍🔬 About the Researcher
Engineered by **Mradul Namdeo**—an AI & Data Science specialist (B.Tech, Jabalpur Engineering College) and GATE-qualified researcher in Data Science & AI. This repository encapsulates the foundational architectures developed during a Pre-Doc Fellowship at the **Department of Mechanical Engineering, Indian Institute of Science (IISc), Bangalore**, under the guidance of Prof. Saptarshi Basu.<!--[cite: 2, 3, 4] -->

---

<details open>
<summary><b>📂 Interactive Table of Contents (Click to Expand)</b></summary>
<br>

- [🔥 Project 1: Automated Flame Tracking & Dynamics Analysis](#-project-1-automated-flame-tracking--dynamics-analysis)
- [💧 Project 2: Unsupervised Spatiotemporal Fluid Segmentation](#-project-2-unsupervised-spatiotemporal-fluid-segmentation)
- [📈 Project 3: High-Frequency Pressure Prediction via RNNs](#-project-3-high-frequency-pressure-prediction-via-rnns)
- [🛠️ Core Technology Stack](#️-core-technology-stack)
</details>

---

## 🔥 Project 1: Automated Flame Tracking & Dynamics Analysis

<a href="Flame_Tracking_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Tracking flame propagation in high-speed experimental combustion data is plagued by severe illumination flicker, wall reflections, and flame fracturing.<!--[cite: 3] --> This two-phase architecture isolates the true physical combustion front and extracts noise-free kinematics (velocity and acceleration) by pairing computer vision with non-linear mathematical modeling.<!--[cite: 3] -->

### Key Architectural Innovations

<details>
<summary><b>Phase 1: Computer Vision Engine</b></summary>
<br>

*   **Intelligent Master Background Selection:** Automatically parses the initial frames and selects the minimum integrated pixel intensity frame to establish a clean baseline.<!--[cite: 3] -->
*   **Static Eraser & Keeper Mask:** Inherent static bright spots inside the pipe geometry are identified, mapped, and mathematically zeroed out using `cv2.bitwise_not` to prevent false anchor locking.<!--[cite: 3] -->
*   **Maximum Intensity Anchoring:** Bypasses the skewing caused by standard geometric centroids when flames deform by strictly anchoring the bounding box to the highest intensity combustion core via `cv2.minMaxLoc`.<!--[cite: 3] -->
</details>

<details>
<summary><b>Phase 2: Mathematical Modeling Engine</b></summary>
<br>

*   **Noise-Free Differentiation:** Replaces high-frequency noise from discrete numerical differentiation ($\Delta x / \Delta t$) with analytical derivatives.<!--[cite: 3] -->
*   **Saturating Exponential Formulation:** The spatial trajectory is mapped using least-squares optimization (`scipy.optimize.curve_fit`):<!--[cite: 3] -->
    $$P_{\text{eq}}(t) = A\left(1 - e^{-k(t - t_0)}\right) + C$$
*   **Analytical Dynamics:** 
    *   Velocity: $V_{\text{eq}}(t) = Ak\,e^{-k(t - t_0)}$<!--[cite: 3] -->
    *   Acceleration: $A_{\text{eq}}(t) = -Ak^2\,e^{-k(t - t_0)}$<!--[cite: 3] -->
*   **Star Normalization:** Compresses varying datasets into a non-dimensional manifold ($T^*, P^*, V^*$) for universal cross-condition comparison.<!--[cite: 3] -->
</details>

### Visual Diagnostics

| System Architecture | Keeper Mask Isolation |
| :---: | :---: |
| *(Upload Slide 5 here as `flame_arch.png`)* <br> <img src="images/flame_arch.png" alt="Architecture" width="400"> | *(Upload Slide 10 here as `flame_mask.png`)* <br> <img src="images/flame_mask.png" alt="Static Eraser" width="400"> |
| *High-level pipeline detailing the transition from raw video to analytical plots.*<!--[cite: 3] --> | *Mathematical deletion of static white spots.*<!--[cite: 3] --> |

| Extracted Kinematics | Live Tracking Validation |
| :---: | :---: |
| *(Upload Slide 21 here as `flame_kinematics.png`)* <br> <img src="images/flame_kinematics.png" alt="Kinematics" width="400"> | *(Upload Slide 22 here as `flame_demo.gif`)* <br> <img src="images/flame_demo.gif" alt="Tracking Demo" width="400"> |
| *Analytically derived velocity and continuous deceleration profiles.*<!--[cite: 3] --> | *Processed tracking output anchoring on maximum intensity.*<!--[cite: 3] --> |

---

## 💧 Project 2: Unsupervised Spatiotemporal Fluid Segmentation

<a href="Image_Segmentation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Macro-scale fluid dynamics demand ultra-precise, sub-pixel edge tracking.<!--[cite: 2] --> This project introduces a 17-step mathematical pipeline designed for 16-bit high-speed imaging to autonomously separate dense fluid masses from low-contrast ambient mist, overcoming severe pixel-density bias and internal geometric voids.<!--[cite: 2] -->

### Core Pipeline Mechanics

*   **Dynamic Nested Gaussian Mixture Models (GMM):** Deploys an unsupervised statistical breakdown of the 16-bit intensity spectrum. A secondary hierarchical GMM isolates the transition fringe zone (Cluster $C_2$) to establish a highly rigid, probabilistically driven binarization threshold.<!--[cite: 2] -->
*   **Graph-Based Topological Extraction:** Executes 8-way Connected Component Labeling (CCL) prioritized by area and boundary constraints to extract the true geometric core.<!--[cite: 2] -->
*   **Ghost Verification & Forward Tracking:** Integrates temporal spatial memory by interrogating detached fragments against historical bounding tracks, mathematically verifying and deleting artifact noise.<!--[cite: 2] -->

### Results & Validation
*   **Precision Metric:** Achieved an exceptional **98% Intersection over Union (IoU)** score when validated against manually annotated high-speed frames.<!--[cite: 2] -->
*   **DL-Readiness:** Outputs strictly isolated, 16-bit physical intensities mapping the exact fluid dynamics, establishing a pristine dataset for downstream deep learning feature extraction.<!--[cite: 2] -->

### Visual Diagnostics

| 16-bit Spectrum Clustering | Priority CCL & Bounding |
| :---: | :---: |
| *(Upload Slide 8 here as `fluid_gmm.png`)* <br> <img src="images/fluid_gmm.png" alt="GMM Convergence" width="400"> | *(Upload Slide 17 here as `fluid_ccl.png`)* <br> <img src="images/fluid_ccl.png" alt="Contour Closure" width="400"> |
| *Final optimized GMM fitted to frame intensity.*<!--[cite: 2] --> | *Resolving internal voids via topological flood-fill.*<!--[cite: 2] --> |

| Ghost Artifact Verification | Final Validation Overlay |
| :---: | :---: |
| *(Upload Slide 22 here as `fluid_ghost.png`)* <br> <img src="images/fluid_ghost.png" alt="Ghost Verification" width="400"> | *(Upload Slide 29 here as `fluid_iou.png`)* <br> <img src="images/fluid_iou.png" alt="IoU Score" width="400"> |
| *Interrogating detached fragments against historical tracks.*<!--[cite: 2] --> | *Segmentation Overlap Analysis achieving 98% IoU.*<!--[cite: 2] --> |

---

## 📈 Project 3: High-Frequency Pressure Prediction via RNNs

<a href="Data_Generation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Predicting continuous pressure dynamics from unseen thermal heat signals. This sequence modeling framework prioritizes architectural simplicity, utilizing a deeply stacked Vanilla Recurrent Neural Network (RNN) to map micro-dynamics without the over-parameterization of complex LSTM gating.<!--[cite: 4] -->

### Engineering Solutions

<details>
<summary><b>1. Feature Engineering the 2D Tensor</b></summary>
<br>

Feeding instantaneous heat alone forces a network to guess the temporal trajectory.<!--[cite: 4] --> We engineered a discrete difference operator to provide the physical trajectory explicitly:
$$X_t = [H_{\text{scaled}}, \Delta H_t]$$
Where $\Delta H_t = H_t - H_{t-1}$. This entirely mitigates the need for long-term memory cells.<!--[cite: 4] -->
</details>

<details>
<summary><b>2. Solving the Cold-Start Transient ($h_0 = 0$)</b></summary>
<br>

Recurrent networks initialize their hidden states at zero, causing massive phase lags at the beginning of predictions.<!--[cite: 4] --> We engineered a dynamic **Warm-Up Buffer**:
*   Ingests a 2000-step continuous slice.<!--[cite: 4] -->
*   **Burn-in Phase (0-500):** Populates the hidden state matrices ($h_t$) to align physical momentum. No loss is calculated here.<!--[cite: 4] -->
*   **Target Phase (501-2000):** Strictly evaluates the final 1500 steps, completely eliminating cold-start initialization penalties.<!--[cite: 4] -->
</details>

<details>
<summary><b>3. Regularization & Architecture</b></summary>
<br>

*   **Deep Hierarchy:** 4 distinct layers with 128 hidden units each ($tanh$ activation).<!--[cite: 4] -->
*   **Information Bottleneck:** The final output is compressed via a 64D Latent Projection (FC1 + ReLU) before generating the 1D pressure scalar.<!--[cite: 4] -->
*   **Anti-Memorization:** Employs aggressive **40% inter-layer dropout** and stochastic sequence windowing to shatter neuron co-adaptation and prevent chronological memorization.<!--[cite: 4] -->
</details>

### Topological Validation

To prove the model generalized physical laws rather than memorizing data, the 128D internal hidden space was mapped via Principal Component Analysis (PCA).<!--[cite: 4] --> 
The hidden state dynamics (*Figure 2*) organized themselves to perfectly mirror the physical ground truth limit cycle (*Figure 1*), proving absolute topological isomorphism.<!--[cite: 4] -->

| Network Information Flow | Topological Phase Space Isomorphism |
| :---: | :---: |
| *(Upload Slide 15 here as `rnn_flow.png`)* <br> <img src="images/rnn_flow.png" alt="Network Flow" width="400"> | *(Upload Slide 24 here as `rnn_phase.png`)* <br> <img src="images/rnn_phase.png" alt="Phase Space" width="400"> |
| *Forward Pass & Backpropagation Through Time.*<!--[cite: 4] --> | *Comparing Physical Phase Space to RNN Hidden Phase Space.*<!--[cite: 4] --> |

---

## 🛠️ Core Technology Stack

*   **Languages:** Python (3.8+)
*   **Deep Learning & Math:** PyTorch (Sequence Modeling, BPTT), SciPy (Non-linear optimization), Scikit-Learn (PCA, Standardization)
*   **Computer Vision:** OpenCV (Contour tracking, Morphological transformations, ROI extraction)
*   **Data Processing:** NumPy, Pandas, Matplotlib

<br>

---
<div align="center">
  <b>For inquiries, research discussions, or code collaborations, feel free to explore the repository files or reach out directly.</b>
</div>
