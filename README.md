# 🔬 Automated Flame Tracking & Dynamics Analysis

<!-- Interactive Badges & PPT Link -->
<p align="center">
  <a href="Flame_Tracking_PPT___Mradul (1).pdf">
    <img src="https://img.shields.io/badge/📄_View_Presentation_Deck-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="View Presentation">
  </a>
  <a href="#-visual-diagnostics--results">
    <img src="https://img.shields.io/badge/📊_Jump_To_Results-0052CC?style=for-the-badge" alt="Jump to Results">
  </a>
</p>

> **A fully automated computer vision and mathematical modeling pipeline for tracking flame propagation in high-noise experimental videos and extracting mathematically rigorous flow dynamics.**[cite: 1]

---

<details>
<summary><b>📍 Interactive Table of Contents (Click to Expand)</b></summary>
<br>

- [🚀 Core Engineering & Research Achievements](#-core-engineering--research-achievements)
- [🏗️ Pipeline Architecture](#-pipeline-architecture)
- [📐 Mathematical Formulation](#-mathematical-formulation)
- [📊 Visual Diagnostics & Results](#-visual-diagnostics--results)
</details>

---

## 🚀 Core Engineering & Research Achievements

<details open>
<summary><b>Click to collapse/expand achievements</b></summary>
<br>

* **Noise-Immune Core Anchoring:** Solved false tracking caused by flame deformation, fracturing, and wall reflections by anchoring tracking logic strictly to the Maximum Intensity pixel core (`cv2.minMaxLoc`) rather than standard geometric centroids[cite: 1].
* **Artifact Elimination via Static Masking:** Engineered an algorithmic master background selector paired with a bitwise-inverted Static Eraser mask[cite: 1]. This maps and mathematically zeroes out pre-existing pipe anomalies before they can hijack the tracker[cite: 1].
* **Closed-Form Kinematic Modeling:** Eliminated the severe high-frequency noise inherent to numerical differentiation ($\Delta x / \Delta t$)[cite: 1]. By modeling position as a saturating exponential curve, velocity and acceleration are derived analytically for perfectly smooth kinematic profiles[cite: 1].
* **Dimensionless Cross-Condition Normalization:** Formulated a characteristic Star Normalization ($T^*, P^*, V^*$) and a fixed 5.0-second temporal scaling to map varying equivalence ratios onto a standardized analytical manifold[cite: 1].
</details>

---

## 🏗️ Pipeline Architecture

The system operates in two distinct phases[cite: 1]:
1. **Phase 1 (Computer Vision):** Optimal background selection, bitwise static masking, dual-zone morphological filtering, and max-intensity contour refinement[cite: 1].
2. **Phase 2 (Mathematical Modeling):** Temporal scaling, saturating exponential curve fitting, analytical dynamics derivation, and non-dimensional normalization[cite: 1].

<div align="center">
  <i>(Upload Slide 5 image here and name it architecture.png)</i><br>
  <img src="images/architecture.png" alt="Architecture Pipeline" width="800">
</div>

---

## 📐 Mathematical Formulation

<details>
<summary><b>Click to view the Mathematical Equations</b></summary>
<br>

To determine true flame dynamics without experimental jitter, spatial coordinates are fitted to a 3-parameter saturating exponential model[cite: 1]:

**1. Position Model:**[cite: 1]
$$P_{\text{eq}}(t) = A\left(1 - e^{-k(t - t_0)}\right) + C$$

**2. Velocity Profile (Analytically Derived):**[cite: 1]
$$V_{\text{eq}}(t) = \frac{d}{dt}P(t) = Ak\,e^{-k(t - t_0)}$$

**3. Acceleration Profile (Analytically Derived):**[cite: 1]
$$A_{\text{eq}}(t) = \frac{d^2}{dt^2}P(t) = -Ak^2\,e^{-k(t - t_0)}$$

</details>

---

## 📊 Visual Diagnostics & Results

<table>
  <tr>
    <td width="50%">
      <b>1. Vision Pipeline Transformations</b><br>
      <i>(Upload Slide 13 image here as tracking_comparison.png)</i><br>
      <img src="images/tracking_comparison.png" alt="Multi-Stage Tracking"><br>
      <sup>Comparison showing Original Frame, Binarized Mask, and Processed Grayscale with Max Intensity Line[cite: 1].</sup>
    </td>
    <td width="50%">
      <b>2. Static Eraser & Keeper Mask</b><br>
      <i>(Upload Slide 10 image here as static_eraser.png)</i><br>
      <img src="images/static_eraser.png" alt="Keeper Mask"><br>
      <sup>Red regions dictate where static white spots are permanently zeroed out via logical NOT operations[cite: 1].</sup>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>3. Extracted Kinematics (Velocity & Accel.)</b><br>
      <i>(Upload Slide 21 images here as kinematics.png)</i><br>
      <img src="images/kinematics.png" alt="Kinematics Plots"><br>
      <sup>Analytically derived velocity and pure deceleration profiles showing peak velocity at ignition followed by continuous exponential decay[cite: 1].</sup>
    </td>
    <td width="50%">
      <b>4. Live Tracking Demonstration</b><br>
      <i>(Upload Slide 22 video/GIF here as tracking_demo.gif)</i><br>
      <img src="images/tracking_demo.gif" alt="Tracking Demo"><br>
      <sup>Real-time max-intensity tracking overlay locked onto the flame front[cite: 1].</sup>
    </td>
  </tr>
</table>

---

<div align="center">
  <b>Research conducted by Mradul Namdeo under the guidance of Prof. Saptarshi Basu at the Department of Mechanical Engineering, Indian Institute of Science (IISc), Bangalore.</b>[cite: 1]<br>
  <a href="#-automated-flame-tracking--dynamics-analysis">⬆️ Back to Top</a>
</div>
