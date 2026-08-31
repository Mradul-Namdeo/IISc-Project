# AI & Computer Vision for Fluid Dynamics

<p align="center">
  <img src="https://img.shields.io/badge/Python-Computer_Vision-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/IISc_Bangalore-Combustion_Research-red?style=for-the-badge" alt="IISc">
</p>

> **A comprehensive portfolio of robust computer vision, deep learning, and mathematical modeling pipelines developed to extract precision analytics from high-noise mechanical engineering experiments.**[cite: 3, 4]

<details open>
<summary><b>📂 Interactive Presentation Deck Links</b></summary>
<br>

1. <a href="Flame_Tracking_PPT___Mradul.pdf">Automated Flame Tracking & Dynamics Analysis</a>
2. <a href="Image_Segmentation_PPT___Mradul.pdf">Spatiotemporal Fluid Segmentation & Flow Tracking</a>
3. <a href="Data_Generation_PPT___Mradul.pdf">Predicting Pressure Dynamics via Stacked RNNs</a>
</details>

---

## 1. Automated Flame Tracking
This two-phase architecture isolates combustion fronts in experimental video to extract smooth, derivative-free kinematics[cite: 3].

*   **Computer Vision Pipeline:** Employs a bitwise-inverted Static Eraser mask and anchors the tracking logic strictly to the maximum intensity core, preventing spatial skewing from flame fracturing and wall reflections[cite: 3].
*   **Mathematical Modeling:** Fits spatial data to a saturating exponential curve ($P_{\text{eq}}(t) = A(1 - e^{-k(t-t_0)}) + C$), allowing for the smooth analytical derivation of velocity and deceleration without numerical jitter[cite: 3].
*   **Star Normalization:** Standardizes varying equivalence ratios and flow speeds into a universal, dimensionless comparative manifold ($T^*, P^*, V^*$)[cite: 3].

## 2. Spatiotemporal Fluid Segmentation
A 17-step mathematical pipeline designed for 16-bit high-speed imaging to execute ultra-precise, sub-pixel edge tracking of macro-scale fluid dynamics[cite: 2].

*   **Nested GMM Processing:** Utilizes an unsupervised, frame-adaptive Gaussian Mixture Model to probabilistically split transition zones and separate dense fluid masses from low-contrast ambient mist[cite: 2].
*   **Topological Extraction:** Implements 8-way Connected Component Labeling (CCL) and historical Ghost Verification tracking to strictly discard detached artifact noise and resolve internal bounding voids[cite: 2].
*   **Grayscale Validation:** Maps the verified topological boundary back onto the locally inverted physical data, achieving an exceptional 98% IoU score against manually annotated frames[cite: 2].

## 3. High-Frequency Pressure Prediction
A deep sequence modeling framework correlating heat release to pressure signals via a 4-layer stacked Vanilla Recurrent Neural Network with heavy inter-layer dropout[cite: 4].

*   **Feature Engineering:** Ingests a 2D tensor featuring instantaneous magnitude ($H_{\text{scaled}}$) and an explicit discrete rate of change ($\Delta H_t$), entirely eliminating the need for complex LSTM gating[cite: 4].
*   **Cold-Start Resolution:** Mitigates the severe initialization lag caused by zeroed memory ($h_0 = 0$) by implementing a 500-step dynamic Warm-up Buffer prior to evaluating the prediction window[cite: 4].
*   **Topological Isomorphism:** The 128D internal memory organically projects into a smooth 2D envelope that perfectly maps onto the physical phase space, proving the network learns continuous underlying physics rather than rote chronology[cite: 4].

<br>
<div align="center">
  <i>Research engineered by Mradul Namdeo under the guidance of Prof. Saptarshi Basu at the Department of Mechanical Engineering, Indian Institute of Science (IISc), Bangalore.</i>[cite: 2, 3, 4]
</div>
