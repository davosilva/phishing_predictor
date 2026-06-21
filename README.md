# 🛡️ Phishing Predictor - Real-time Website Fraud Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow.svg)](https://huggingface.co/datasets/phreshphish/phreshphish)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

**Phishing Predictor** is an advanced phishing detection system that uses Machine Learning to analyze URLs in real time and determine whether a website is legitimate or fraudulent.

The system is trained on the **PhreshPhish** dataset from Hugging Face, containing over **500,000** phishing and benign URLs.

---

## 🔍 How It Works

1. **Whitelist Check** – Verifies known legitimate domains.
2. **Pattern Detection** – Identifies domain imitations and suspicious TLDs.
3. **Machine Learning** – Random Forest model with PCA.
4. **Feature Analysis** – Evaluates 18 URL characteristics.

---

## 🚀 Installation

```bash
git clone https://github.com/davosilva/phishing-predictor.git
cd phishing-predictor

python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
datasets>=2.0.0
huggingface-hub>=0.14.0
joblib>=1.1.0
pytest>=7.0.0
```

---

## 🔑 Hugging Face Token

Get a token from:

https://huggingface.co/settings/tokens

Set it as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

---

## ▶️ Run

```bash
python phishing_predictor.py
```

---

## 📁 Project Structure

```text
phishing-predictor/
├── phishing_predictor.py
├── token_verify.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

**PhreshPhish**

- 500,000+ training samples
- 168,000+ test samples
- Real-world phishing and benign URLs
- License: CC BY 4.0

---

## 🔧 Technologies

- Python 3.8+
- scikit-learn
- Hugging Face Datasets
- pandas
- numpy

---

## 🛡️ Security Features

- SSL / HTTPS detection
- Domain whitelist
- Pattern recognition
- Risk scoring
- Confidence levels
- Real-time analysis

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch
3. Commit changes
4. Push changes
5. Open a Pull Request

---

## 📄 License

MIT License

---

## 📧 Contact

Author: davosilva

GitHub: https://github.com/davosilva


## 📚 Dataset Citation

This project is trained and evaluated using the **PhreshPhish** dataset, a large-scale, real-world phishing website benchmark designed to improve the quality and realism of phishing detection research.

### Dataset Statistics

| Split    | Samples |
| -------- | ------: |
| Training | 498,255 |
| Testing  | 168,060 |
| Total    | 666,315 |

### Dataset Source

**Hugging Face Dataset**

https://huggingface.co/datasets/phreshphish/phreshphish

### Research Paper

Dalton, T., Gowda, H., Rao, G., Pargi, S., Hadj Khodabakhshi, A., Rombs, J., Jou, S., & Marwah, M. (2025).

**PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark**

This paper introduces a large-scale phishing dataset and benchmark specifically designed to provide realistic evaluation conditions for phishing detection systems and machine learning research.

Paper:

https://arxiv.org/abs/2507.10854

### BibTeX

```bibtex
@article{dalton2025phreshphish,
    title        = {PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark},
    author       = {Thomas Dalton and Hemanth Gowda and Girish Rao and Sachin Pargi and Alireza Hadj Khodabakhshi and Joseph Rombs and Stephan Jou and Manish Marwah},
    year         = {2025},
    journal      = {arXiv preprint},
    eprint       = {2507.10854},
    url          = {https://arxiv.org/abs/2507.10854}
}
```

### Citation Request

If you use this project, please consider citing the original PhreshPhish dataset and benchmark paper to support the authors and ongoing phishing detection research.
