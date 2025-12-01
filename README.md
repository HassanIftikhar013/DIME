# 🧠 DIME: Deep-Learning–Driven Inversion Framework for MRE  
### Author: Hassan Iftikhar  
### Model & Sample Data Release for Reproducibility

This repository contains the **trained DIME model**, along with **sample displacement fields**, **ground-truth stiffness maps**, and **notebooks** for reproducing the experiments described in the DIME paper.  
The goal of this repository is to provide **transparent reproducibility** for reviewers and researchers by allowing them to run the exact inference pipeline used in the manuscript.

---

## 📌 Overview

**DIME (Deep Image-based Modulus Estimator)** is a patch-based deep learning inversion framework for **Magnetic Resonance Elastography (MRE)**.  
The method takes as input:

- The **displacement field (DF)**  
- First harmonic **real** and **imaginary** components  
- Each channel is individually rescaled to **0–4095**  
- Input is fed to the model patch-by-patch  
- A full-resolution stiffness map is reconstructed  
- Results are compared against **ground truth** and **MMDI**

This repository includes:

- ✔ `trained_model.pth` – the trained DIME network  
- ✔ Sample datasets for each study  
- ✔ Four fully runnable notebooks  
- ✔ Patch-based inference modules  
- ✔ Custom AWAVE and AAASMO colormaps  
- ✔ Statistical evaluation (mean, SD, Excel export)  
- ✔ Reconstruction & visualization utilities  

---

## 📁 Repository Structure

