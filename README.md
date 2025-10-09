# 🧪 CMRAgent: A Large Language Model Agent for Chemical Safety Assessment

## 🧬 Overview
**CMRAgent** is a vertical large language model (LLM) agent designed for **comprehensive chemical safety assessment**, focusing on **carcinogenicity (C)**, **mutagenicity (M)**, and **reproductive toxicity (R)**.  
It integrates multi-source chemical data, predictive modeling, and autonomous reasoning capabilities to enhance chemical safety analysis with **high accuracy** and **reduced hallucination risks**.

This repository provides:
- 🧠 The CMRAgent framework for automated chemical assessment  
- 🔬 The **CMRScreen** predictive module for molecular toxicity prediction  
- 📊 Utility scripts for data processing, evaluation, and visualization  

---

## 🧠 Abstract
Large language models (LLMs) are revolutionizing artificial intelligence (AI) and demonstrate strong potential to address various challenges in chemical safety assessment—from integrating multidisciplinary data and complex relational reasoning to toxicity profiling and regulatory compliance.  
However, general-purpose LLMs fail to fully capture or reason about chemical complexities and consequently tend to produce hallucinatory or unverified claims due to the lack of specialized domain knowledge in chemical sciences, environmental and human health impacts, and regulatory governance.  
To transform chemical safety assessment from labor-intensive and compliance-oriented processes toward an automated, intelligent, and evidence-based framework, we present **CMRAgent**, a vertical LLM agent for comprehensive assessment of **carcinogenic, mutagenic, or toxic to reproduction (CMR)** endpoints with high accuracy and reduced hallucination risks.  
CMRAgent integrates LLMs with **function calling** and the **ReAct (Reasoning–Acting)** paradigm to autonomously execute a wide range of tasks including compound-related question answering (Q&A), CMR property prediction, and regulatory-compliant report generation.  
It leverages multiple LLMs with over 100 custom retrieval tools via **LangChain** and incorporates **CMRScreen**, a predictive module trained using a semi-supervised learning approach based on the **Directed Message Passing Neural Network (DMPNN)**.  
CMRAgent achieves **state-of-the-art (SOTA)** accuracy across three CMR endpoints (89.03%, 85.35%, and 89.53%), and effectively automates chemical safety tasks with >85% accuracy on Q&A benchmarks, outperforming general-purpose LLMs (accuracy <35%). To our knowledge, this is the **first LLM-based agentic system for chemical safety assessment**, addressing limitations such as heavy reliance on animal testing, fragmented workflows, and limited scalability—paving the way for a unified, multi-endpoint chemical safety evaluation ecosystem.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ml-zju/cmr.git
cd cmragent
```
### 2️⃣ Create the Conda environment
```bash
conda env create -f environment.yml
```
### 3️⃣ Activate the environment
```bash
conda activate cmragent
```
### 🚀 Run the Application
Launch CMRAgent
```bash
streamlit run app.py
```
### 📧 Contact
For questions, collaborations, or contributions, please reach out: shulin@zju.edu.cn

### 🧱 License
This project is released under the MIT License.
