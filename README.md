# Cyberbullying Detection using Machine Learning

## Overview
This project focuses on detecting cyberbullying in online text using machine learning techniques. It utilizes Natural Language Processing (NLP) to analyze textual data and classify whether a given message contains cyberbullying content.

## Features
- **Text Classification**: Uses NLP models to identify cyberbullying in text.
- **Deep Learning Model**: Implements DistilBERT for high-accuracy detection.
- **Class Imbalance Handling**: Uses resampling techniques for better model performance.
- **Real-Time Detection**: Can be integrated into chat applications.
- **Fine-Tuned GPT-2 Chatbot**: Provides emotional support to users.

## Technologies Used
- **Python**
- **PyTorch**
- **Transformers (Hugging Face)**
- **Scikit-learn**
- **Pandas & NumPy**
- **Flask/Django (for API Integration)**

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/heisenberg2358/cyberbullying-detection.git
   cd cyberbullying-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset (if required) and place it in the `data/` folder.

## Usage
1. Train the model:
   ```sh
   python train.py
   ```
2. Test the model:
   ```sh
   python test.py
   ```
3. Deploy using Flask/Django:
   ```sh
   python app.py
   ```

## Dataset
The model is trained on a labeled dataset containing examples of cyberbullying and non-cyberbullying texts. Preprocessing steps include tokenization, padding, and vectorization.

## Model Training
- The model is fine-tuned on a cyberbullying dataset using DistilBERT.
- Mixed precision training is used for efficiency.
- Fine-tuning is limited to certain layers for better generalization.

## Deployment
The model can be deployed as a REST API using Flask or Django, making it easy to integrate into web applications or chat platforms.

## Contribution
Feel free to contribute by submitting pull requests or opening issues for feature suggestions or bug reports.

## License
This project is licensed under the MIT License.

---
### Contact
For queries, contact ajumalsabeer350@gmail.com or open an issue in the repository.

**GitHub Repository**: [https://github.com/heisenberg2358/cyberbullying-detection](https://github.com/heisenberg2358/cyberbullying-detection)

