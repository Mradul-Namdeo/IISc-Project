# AI & Computer Vision for Fluid Dynamics and Combustion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/SciPy-Mathematical_Modeling-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Research-IISc_Bangalore-red?style=for-the-badge" alt="IISc">
</p>

### 👨‍🔬 About the Researcher
**Mradul Namdeo** | Pre-Doc Fellow | IISc Bangalore <!--[cite: 2, 3, 4] -->
Research conducted under the guidance of Prof. Saptarshi Basu at the Department of Mechanical Engineering, Indian Institute of Science. <!--[cite: 2, 3, 4] -->

---

<details open>
<summary><b>📂 Interactive Table of Contents (Click to Expand)</b></summary>
<br>

- [🔥 Project 1: Automated Flame Tracking & Dynamics Analysis](#-project-1-automated-flame-tracking--dynamics-analysis)
- [📈 Project 2: High-Frequency Pressure Prediction via RNNs](#-project-2-high-frequency-pressure-prediction-via-rnns)
- [💧 Project 3: Unsupervised Spatiotemporal Fluid Segmentation](#-project-3-unsupervised-spatiotemporal-fluid-segmentation)
- [🛠️ Core Technology Stack](#️-core-technology-stack)
</details>

---

## 🔥 Project 1: Automated Flame Tracking & Dynamics Analysis

<a href="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Flame_tracker/FREI%20videos/Report/Flame_Tracking_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Tracking flame propagation in high-speed experimental combustion data is plagued by severe illumination flicker, wall reflections, and flame fracturing. <!--[cite: 3] --> This two-phase architecture isolates the true physical combustion front and extracts noise-free kinematics (velocity and acceleration) by pairing computer vision with non-linear mathematical modeling. <!--[cite: 3] -->

### Visual Diagnostics

**1. System Architecture**
<div align="center">
  <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Flame_tracker/Flowcharts/Simple/Flame%20Tracker%20Simple%20Flowchart.jpg" alt="System Architecture" width="800"><br>
  <i>Pipeline detailing the transition from raw video to analytical plots.</i> <!--[cite: 3] -->
</div>

<br>

**2. Tracking Validation**
<div align="center">
  <a href="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Flame_tracker/FREI%20videos/Phi%201.0/Outputs/Tracked%20videos/Phi_1p0_u_0p3_C001H001S0001/Phi_1p0_u_0p3_C001H001S0001_frames_flame_tracked_intensity.mp4">
    <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Flame_tracker/FREI%20videos/Phi%201.0/Outputs/Tracked%20videos/Phi_1p0_u_0p3_C001H001S0001/Phi_1p0_u_0p3.png" alt="Click to Watch Tracked Video" width="800">
  </a><br>
  <i>Max-intensity tracking overlay locked onto the flame front. (Click image to view video)</i> <!--[cite: 3] -->
</div>

---

## 📈 Project 2: High-Frequency Pressure Prediction via RNNs

<a href="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Report/Data_Generation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Predicting continuous pressure dynamics from unseen thermal heat signals. This sequence modeling framework prioritizes architectural simplicity, utilizing a deeply stacked Vanilla Recurrent Neural Network (RNN) to map micro-dynamics without the over-parameterization of complex LSTM gating. <!--[cite: 4] -->

### Visual Diagnostics

**1. Stacked_RNN_Architecture**
<div align="center">
  <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Architecture/Stacked_RNN_Architecture.png" alt="Information Flow: Forward Pass & Backpropagation" width="800"><br>
  <i>Network Information Flow.</i> <!--[cite: 4] -->
</div>

<br>

**2. Validation in the Time Domain: Uniform Window Sampling**
To rigorously evaluate the model’s stability, we extracted six uniformly spaced inference windows (W1 to W6) across the entire unseen dataset using a linear space distribution. Each window represents 1500
time steps of continuous open-loop prediction (strictly evaluated after the 500-step warm-up phase).
This confirms that generative accuracy is consistent and independent of specific starting regimes.


| Window 1 | Window 2 | Window 3 |
| :---: | :---: | :---: |
| <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_1.png" alt="Window 1" width="300"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_2.png" alt="Window 2" width="300"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_3.png" alt="Window 3" width="300"> |
| **Window 4** | **Window 5** | **Window 6** |
| <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_4.png" alt="Window 4" width="300"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_5.png" alt="Window 5" width="300"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/2_Window_6.png" alt="Window 6" width="300"> |

<br>

**3. Topological Validation:**

> 💡 **The Challenge:** Statistical metrics alone are insufficient to prove dynamical generalization.<!--[cite: 4] --> 
> 
> 🧠 **The Solution:** To verify the network learned the true system topology instead of merely memorizing sequential data points, **Principal Component Analysis (PCA)** was applied to the 128-dimensional internal memory ($h_t^{(4)}$) of the RNN.<!--[cite: 4] --> 
> 
> 🌌 **The Result:** The resulting latent projection organically forms a smooth, bounded 2D envelope that **perfectly mirrors the physical ground truth limit cycle** ($P$ vs $\Delta P$).<!--[cite: 4] --> This structural isomorphism guarantees the model fundamentally operates on the continuous underlying physics of the thermoacoustic system.<!--[cite: 4] -->

| Physical Phase Space | Hidden Phase Space |
| :---: | :---: |
| <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/6_Physical_Phase_Space.png" alt="Physical Phase Space" width="450"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Data_generation/Stacked_RNN_Results/Phi_1p2_u_0p3/7_Hidden_Phase_Space.png" alt="Hidden Phase Space" width="450"> |
| *Physical Phase Space (P vs ΔP)* <!--[cite: 4] --> | *Hidden Phase Space (PCA h_t^(4))* <!--[cite: 4] --> |

---

## 💧 Project 3: Unsupervised Spatiotemporal Fluid Segmentation

<a href="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Unsupervised_Image_Segmentation_task/Report/Image_Segmentation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Macro-scale fluid dynamics demand ultra-precise, sub-pixel edge tracking. <!--[cite: 2] --> This project introduces a 17-step mathematical pipeline designed for 16-bit high-speed imaging to autonomously separate dense fluid masses from low-contrast ambient mist, overcoming severe pixel-density bias and internal geometric voids. <!--[cite: 2] -->

### Visual Diagnostics

| GMM Convergence | 16-bit Spectrum Clustering |
| :---: | :---: |
| <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Unsupervised_Image_Segmentation_task/Results/10_KV/Steps/Step_03c_Maximization_Step.png" alt="16-bit Spectrum Clustering" width="450"> | <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Unsupervised_Image_Segmentation_task/Results/10_KV/Steps/Step_03d_2D_Color_Map.png" alt="Cluster-2 (Fringe) Isolation" width="450"> |
| *Final optimized GMM fitted to frame intensity.* <!--[cite: 2] --> | *2D GMM Hard Assignment Map capturing the transition zone.* <!--[cite: 2] --> |

<br>

<div align="center">
  <h3>Final Result & Validation</h3>
  <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Unsupervised_Image_Segmentation_task/Results/10_KV/Steps/Step_15_Final_Comparison.png" alt="Final Result & Validation" width="800"><br>
  <i>Comparison between normalized raw image and predicted jet.</i> <!--[cite: 2] -->
</div>

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
