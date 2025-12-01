# 🧠 Deep Image-Based Modulus Estimator (DIME)
<h2>(Deep-Learning–Driven Inversion Framework for Magnetic Resonance Elastography)</h2>

This repository contains the implementation of the DIME inversion algorithm used in our research on **MRE-based shear modulus estimation**. DIME is a physics-informed, deep-learning–driven inversion method designed to improve the robustness and quality of Magnetic Resonance Elastography (MRE) reconstructions, especially in scenarios where traditional inversion methods (such as MMDI) struggle.

This repository provides:

- The **trained DIME model (`trained_model.pth`)**
- **Sample displacement field data** for each study
- **Ground-truth stiffness maps**
- **Fully executable Jupyter notebooks** for reproducing all results

Each notebook demonstrates the full workflow:  
**loading displacement fields → preprocessing → patch-based inference → visualization → quantitative evaluation.**

---

## About the Project

Magnetic Resonance Elastography (MRE) is widely used to assess tissue stiffness, but conventional inversion algorithms (e.g., MMDI) rely on simplifying assumptions that often introduce noise, inaccuracy, and spatial variability.  
To address these limitations, our work introduces **DIME** — a **deep-learning inversion framework** trained on finite-element (FEM)–generated displacement–stiffness pairs.

DIME learns a direct mapping from **first-harmonic displacement fields** to **shear modulus**, enabling:

- Improved noise robustness  
- Superior boundary preservation  
- Spatially consistent reconstructions  
- Generalization to unseen anatomical geometries and real in-vivo data  

The repository contains **four studies**, each demonstrating the performance of DIME under different experimental conditions.

### Key Features

- End-to-end **patch-based inversion** for 2D stiffness estimation  
- **FFT-based preprocessing** for raw in-vivo displacement fields   
- Quantitative comparison against:
  - **Ground Truth (GT)**
  - **MMDI** (where applicable)
- Support for heterogeneous phantoms and anatomically informed liver simulations  
- Works on both **synthetic FEM data** and **in-vivo liver MRE**  
- Fully reproducible using the included **Jupyter notebooks**

### Included Studies

- **Study 1(a):** Homogeneous FEM phantoms  
- **Study 1(b):** Heterogeneous FEM phantoms  
- **Study 2:** Anatomy-informed liver phantom  
- **Study 3:** In-vivo liver MRE (clinical acquisition)

Each study folder contains sample data and one notebook demonstrating the full inference pipeline.

---

## Getting Started

To use the reconstruction notebooks:

### Installation

Requirements:

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Nibabel
- Matplotlib
- Plotly (optional)
- scikit-learn
- pandas

Install via:

```bash
pip install torch numpy scipy nibabel matplotlib scikit-learn pandas plotly```



### Usage

Each study includes a separate Jupyter notebook:
- Study1(a)-Notebook.ipynb
- Study1(b)-Notebook.ipynb
- Study2-Notebook.ipynb
- Study3-InVivo-Notebook.ipynb

### To reproduce results:
- Open the desired notebook
- Ensure trained_model.pth is in the root directory
- Run all cells sequentially
