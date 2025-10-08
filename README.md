# CMRAgent: A Large Language Model Agent for Chemical Safety Assessment

## 🧬 Overview
**CMRAgent** is a vertical large language model (LLM) agent designed for **comprehensive chemical safety assessment**, focusing on the evaluation of **carcinogenicity (C)**, **mutagenicity (M)**, and **reproductive toxicity (R)**.  
It integrates multi-source chemical data, predictive modeling, and autonomous reasoning capabilities to enhance chemical safety analysis with **high accuracy** and **reduced hallucination risks**.

This repository provides:
- The CMRAgent framework for automated chemical assessment;
- The CMRScreen predictive module for molecular toxicity prediction;
- Utility scripts for data processing, evaluation, and visualization.

---

## 🧠 Abstract
Large language models (LLMs) are revolutionizing artificial intelligence (AI) and demonstrate strong potential to address various challenges in chemical safety assessment, from integrating multidisciplinary data and complex relational reasoning to toxicity profiling and regulatory compliance. However, general-purpose LLMs fail to fully capture or reason chemical complexities and consequently tend to produce significant hallucinatory or unverified scientific claims due to the lack of specialized domain knowledge in chemical sciences, environmental and human health impacts, and policy and regulatory governance.  
To transform chemical safety assessment from labor-intensive and compliance-oriented processes toward an automated, intelligent, and evidence-based framework, we present a vertical LLM agent named **CMRAgent** for comprehensive chemical safety assessment towards **carcinogenic, mutagenic, or toxic to reproduction (CMR)** endpoints with high accuracy and significantly reduced hallucination risks.  
CMRAgent integrates LLMs with function calling and the **ReAct (Reasoning–Acting)** paradigm to autonomously execute a wide spectrum of tasks across compound-related question answering (Q&A), CMR property prediction, and regulatory-compliant report generation. It leverages multiple LLMs with over 100 custom retrieval tools via LangChain and incorporates **CMRScreen**, a predictive module trained using a semi-supervised learning method with the **Directed Message Passing Neural Network (DMPNN)**.  
CMRAgent achieves **state-of-the-art (SOTA)** accuracy across three CMR endpoints (89.03%, 85.35%, and 89.53%) and effectively automates chemical safety tasks with >85% accuracy on Q&A benchmarks, outperforming general-purpose LLMs (accuracy <35%).  
To our knowledge, this is the **first LLM-based agentic system for chemical safety assessment**, overcoming limitations such as heavy reliance on animal testing, fragmented workflows, and poor scalability, thus providing a foundation for a unified, multi-endpoint chemical safety evaluation ecosystem.

---

⚙️ Installation
1. Clone the repository
git clone https://github.com/ml-zju/cmr.git
cd cmr

2. Create the Conda environment
conda env create -f environment.yml

3. Activate the environment
conda activate cmr

🚀 Run the Application
1. Launch CMR-Agent
streamlit run app.py

2. Access the web interface

After running the command above, open the URL displayed in your terminal (usually http://localhost:8501) in your browser.

