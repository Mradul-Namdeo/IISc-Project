# 🌊 Advanced AI & Computer Vision for Fluid Dynamics and Combustion

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

<a href="Flame_Tracking_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Tracking flame propagation in high-speed experimental combustion data is plagued by severe illumination flicker, wall reflections, and flame fracturing. <!--[cite: 3] --> This two-phase architecture isolates the true physical combustion front and extracts noise-free kinematics (velocity and acceleration) by pairing computer vision with non-linear mathematical modeling. <!--[cite: 3] -->

### Visual Diagnostics

| System Architecture | Live Tracking Validation |
| :---: | :---: |
| <img src="https://github.com/Mradul-Namdeo/IISc-Project/blob/main/Flame_tracker/Flowcharts/Simple/Flame%20Tracker%20Simple%20Flowchart.jpg" alt="System Architecture" width="450"> | <img src="<paste_video_link_here>" alt="Live Tracking Validation" width="450"> |
| *High-level pipeline detailing the transition from raw video to analytical plots.* <!--[cite: 3] --> | *Real-time max-intensity tracking overlay locked onto the flame front.* <!--[cite: 3] --> |

---

## 📈 Project 2: High-Frequency Pressure Prediction via RNNs

<a href="Data_Generation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Predicting continuous pressure dynamics from unseen thermal heat signals. This sequence modeling framework prioritizes architectural simplicity, utilizing a deeply stacked Vanilla Recurrent Neural Network (RNN) to map micro-dynamics without the over-parameterization of complex LSTM gating. <!--[cite: 4] -->

### Visual Diagnostics

**1. Network Information Flow**
<div align="center">
  <img src="<paste_link_here>" alt="Information Flow: Forward Pass & Backpropagation" width="600"><br>
  <i>Step 2: Backpropagation Through Time. Gradients ($\nabla\mathcal{L}$) flow backwards to update weights, calculating partial derivatives at each depth.</i> <!--[cite: 4] -->
</div>

<br>

**2. Topological Validation**
| Physical Phase Space | Hidden Phase Space |
| :---: | :---: |
| <img src="<paste_link_here>" alt="Physical Phase Space" width="450"> | <img src="<paste_link_here>" alt="Hidden Phase Space" width="450"> |
| *Physical Phase Space ($P$ vs $\Delta P$)* <!--[cite: 4] --> | *Hidden Phase Space (PCA $h_t^{(4)}$)* <!--[cite: 4] --> |

---

## 💧 Project 3: Unsupervised Spatiotemporal Fluid Segmentation

<a href="Image_Segmentation_PPT___Mradul.pdf">
  <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
</a>

### Overview
Macro-scale fluid dynamics demand ultra-precise, sub-pixel edge tracking. <!--[cite: 2] --> This project introduces a 17-step mathematical pipeline designed for 16-bit high-speed imaging to autonomously separate dense fluid masses from low-contrast ambient mist, overcoming severe pixel-density bias and internal geometric voids. <!--[cite: 2] -->

### Visual Diagnostics

| 16-bit Spectrum Clustering | Step 3D: Cluster-2 (Fringe) Isolation |
| :---: | :---: |
| <img src="<paste_link_here>" alt="16-bit Spectrum Clustering" width="450"> | <img src="<paste_link_here>" alt="Cluster-2 (Fringe) Isolation" width="450"> |
| *Final optimized GMM fitted to frame intensity.* <!--[cite: 2] --> | *2D GMM Hard Assignment Map capturing the transition zone.* <!--[cite: 2] --> |

| Priority CCL & Bounding | Final Result & Validation |
| :---: | :---: |
| <img src="<paste_link_here>" alt="Priority CCL & Bounding" width="450"> | <img src="<paste_link_here>" alt="Final Result & Validation" width="450"> |
| *Resolving internal voids via topological flood-fill.* <!--[cite: 2] --> | *Comparison between normalized raw image and predicted jet.* <!--[cite: 2] --> |

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
