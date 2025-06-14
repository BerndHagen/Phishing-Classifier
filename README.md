<p align="center">
  <img src="https://github.com/BerndHagen/Phishing-Classifier/blob/bb3129803b797e3a727027fed4795fea18a67eac/img/img_v1.0.0-file.png" alt="Phishing Classifier Logo" width="128" />
</p>
<h1 align="center">Phishing Classifier</h1>
<p align="center">
  <b>AI-powered phishing detection engine with interactive feedback system</b><br>
  <b>Advanced machine learning solution for real-time phishing email detection and classification</b>
</p>
<p align="center">
  <a href="https://github.com/BerndHagen/Phishing-Classifier/releases"><img src="https://img.shields.io/github/v/release/BerndHagen/Phishing-Classifier?include_prereleases&style=flat-square&color=CD853F" alt="Latest Release"></a>&nbsp;&nbsp;<a href="https://github.com/BerndHagen/Phishing-Classifier/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>&nbsp;&nbsp;<img src="https://img.shields.io/badge/Python-3.13+-blue?style=flat-square" alt="Python Version">&nbsp;&nbsp;<img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square" alt="Platform">&nbsp;&nbsp;<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status">&nbsp;&nbsp;<a href="https://github.com/BerndHagen/Phishing-Classifier/issues"><img src="https://img.shields.io/github/issues/BerndHagen/Phishing-Classifier?style=flat-square" alt="Open Issues"></a>
</p>

**Phishing Classifier** is an open-source machine learning tool I built to detect phishing emails. It's freely available for anyone to use and works well for both personal use and larger email systems, helping identify fraudulent emails before they cause problems. The system uses natural language processing and pattern analysis to catch phishing attempts more effectively than general-purpose AI tools.

### **Key Features**

- **98% Accuracy:** The classifier correctly identifies phishing emails 98% of the time using machine learning algorithms trained specifically for email security.
- **Interactive GUI:** Simple interface that lets you review classifications and provide feedback to improve the model over time.
- **Fast Processing:** Quick email classification for real-world use.
- **Gets Better Over Time:** The model improves through user feedback and retraining.

### **Supported Classification Types**

The classifier handles binary classification for email security:

- **Legitimate Emails:** Normal business emails, personal messages, newsletters, notifications
- **Phishing Emails:** Scam attempts, fraudulent messages, social engineering attacks

### **Model Performance**

The classifier performs well on test data:

- **Overall Accuracy:** 98% on a test set of 4,000 emails
- **Precision:** 95% (Legitimate) | 100% (Phishing)
- **Recall:** 100% (Legitimate) | 95% (Phishing)
- **False Positive Rate:** Less than 0.1% (1 out of 2,055 legitimate emails marked as phishing)

**Processing Speed:** Most emails are classified within seconds, though complex text analysis might take a bit longer for accuracy.

## **Table of Contents**

1. [Third-Party Libraries](#third-party-libraries)
   - [Scikit-Learn](#scikit-learn)
   - [Pandas](#pandas)
   - [Additional Dependencies](#additional-dependencies)
2. [Model Architecture and Benefits](#model-architecture-and-benefits)
   - [Technical Components](#technical-components)
   - [Main Features](#main-features)
3. [Installation Guide](#installation-guide)
4. [Getting Started Guide](#getting-started-guide)
   - [Step 1: Set Up Your Environment](#step-1-set-up-your-environment)
   - [Step 2: Train and Use](#step-2-train-and-use)
5. [AI Comparison Study](#ai-comparison-study)
6. [Confusion Matrix](#confusion-matrix)
7. [Classification Report](#classification-report)
8. [License & Disclaimer](#license--disclaimer)
9. [Screenshots](#screenshots)

## Third-Party Libraries

The classifier uses several Python libraries to process email data and make predictions. The main ones are **Scikit-Learn** and **Pandas**, plus some visualization tools.

### Scikit-Learn

**Scikit-Learn** is the main machine learning library that handles model training and predictions. It provides the RandomForest classifier and tools for optimizing model parameters.

- **Website:** [Scikit-Learn Official Website](https://scikit-learn.org)
- **License:** BSD 3-Clause License

### Pandas

**Pandas** handles data processing and analysis. I use it to clean email data, engineer features, and analyze results.

- **Website:** [Pandas Official Website](https://pandas.pydata.org)
- **License:** BSD 3-Clause License

### Additional Dependencies

The project also uses:

- **Matplotlib & Seaborn:** For charts and performance plots
- **Tkinter:** For the GUI interface
- **Joblib:** To save and load the trained model

Feel free to [open an issue](https://github.com/BerndHagen/Phishing-Classifier/issues) if you have questions about any of these libraries.

## **Model Architecture and Benefits**

The classifier uses **machine learning techniques** to identify phishing emails. It combines TF-IDF text analysis, RandomForest classification, and user feedback to catch malicious emails effectively.

### Technical Components

| **Component**                   | **Description**                                                        |
|---------------------------------|------------------------------------------------------------------------|
| **TF-IDF Vectorization**        | Text analysis that looks at word frequency and importance |
| **RandomForest Classifier**     | Multiple decision trees working together for better predictions |
| **Feature Engineering**         | Analyzes message length and language patterns    |
| **GridSearchCV Optimization**   | Automatically finds the best model settings using cross-validation          |

### Main Features

| **Feature**                     | **Description**                                                        |
|---------------------------------|------------------------------------------------------------------------|
| **Learning from Feedback**         | Improves over time based on user corrections   |
| **Easy-to-Use Interface**       | Simple GUI for reviewing predictions and training the model |
| **Fast Processing**        | Quick email classification with performance tracking   |
| **Pattern Analysis**         | Looks at both text content and email structure         |

## Installation Guide

This project requires Python 3.8+ and several essential libraries.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BerndHagen/Phishing-Classifier.git
   cd Phishing-Classifier
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python PhishingClassifier.py
   ```

## **Getting Started Guide**

Here's how to start using the classifier:

### **Step 1: Set Up Your Environment**

1. **Run the Application**
   - Execute **PhishingClassifier.py** to start the program.

2. **Load Training Data**
   - The system loads a dataset of 19,980 email samples automatically.
   - Training takes about 2-3 minutes while it finds the best model settings.

### **Step 2: Train and Use**

1. **Model Training**
   - The program trains using RandomForestClassifier with automatic parameter optimization.
   - You'll see a confusion matrix showing how well the model performed.

2. **Start Classifying Emails**
   - **Try the Examples:** The program includes sample emails to test with.
   - **See Results:** The system shows either "Phishing Detected!" or "No Threat Detected".

3. **Give Feedback**
   - Use the GUI to review each prediction.
   - Click "Correct" if the prediction looks right, or "Incorrect" if it's wrong.
   - Wrong predictions get saved to improve the model later.

4. **Track Performance**
   - Watch accuracy and other stats update in real-time.

## **AI Comparison Study**

I tested the classifier against popular AI systems using 100 email samples in May 2024. Note that AI systems are constantly improving with new features like thinking modes, so they may perform better now.

| **System** | **Correctly Identified** | **Incorrectly Identified** | **False Positives** | **False Negatives** |
|------------|--------------------------|----------------------------|--------------------|--------------------|
| **Phishing Classifier** | **98** | **2** | **0** | **2** |
| Microsoft Copilot | 94 | 6 | 1 | 5 |
| Claude | 81 | 19 | 7 | 12 |
| ChatGPT 4 | 80 | 20 | 4 | 16 |
| Gemini | 64 | 36 | 8 | 28 |
| ChatGPT 3.5 | 61 | 39 | 14 | 25 |

The specialized approach worked better than general-purpose AI systems at the time of testing, achieving the best results with zero false positives.

## **Confusion Matrix**

The confusion matrix shows how well the classifier performs in detail when tested on the dataset that's included in the releases. This evaluation shows the model can accurately tell the difference between legitimate and malicious emails using the same data I used for development and testing.

The system correctly identified **2,054 legitimate emails** as safe (True Negatives) and caught **1,848 phishing attempts** (True Positives), showing good accuracy in email classification. Mistakes are rare with low error rates that confirm the system works reliably.

**Key Performance Results:**
- **True Negatives:** 2,054 legitimate emails correctly identified
- **True Positives:** 1,848 phishing attempts successfully detected
- **False Positives:** 1 legitimate email incorrectly flagged (0.05% error rate)
- **False Negatives:** 97 phishing attempts missed (5% miss rate)

## **Classification Report**

The classification report shows **individual message analysis** with interactive feedback, demonstrating how the system works in practice. This interface lets users review classification results and help improve the model through feedback.

Each message gets analyzed with the model's prediction clearly shown alongside options for user validation. The system shows classification results with confidence levels, so users can verify accuracy and provide corrections when needed.

**Interactive Feedback Example:** The message *"Just landed in New York. I'll give you a call once I'm settled in at the hotel. Maybe we can meet up for dinner if you would like? Let me know if you're around."* is classified as **"No Threat Detected"** with user feedback confirming this is correct.

**Learning from Mistakes:** Wrong classifications are automatically saved and used to retrain the model, so it keeps getting better at detecting new phishing techniques.

## **License & Disclaimer**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/BerndHagen/Phishing-Classifier/blob/main/LICENSE) file for details. This is a completed project that demonstrates phishing email classification techniques. The project is provided as-is for educational purposes and reference.

- This tool is for educational and research purposes only
- Always implement multiple layers of security in production environments
- The model was trained on data available up to the training date and may not catch newer phishing techniques
- Regular updates and retraining would be needed for production use
- No warranty or guarantee is provided for the accuracy of classifications

## **Screenshots**

The screenshots below provide an overview of how the application looks when you run the Python script with the included dataset. You can see the training process, the GUI interface, and how the model learns through user feedback.

| Classifier - Confusion Matrix       | Classifier - Classification Report     | Classifier - User Feedback        |
|------------------------------|-----------------------------|-----------------------------|
| <img src="https://github.com/BerndHagen/Phishing-Classifier/raw/main/img/img_v1.0.0-matrix.png" width="300px"> | <img src="https://github.com/BerndHagen/Phishing-Classifier/raw/main/img/img_v1.0.0-report.png" width="300px"> | <img src="https://github.com/BerndHagen/Phishing-Classifier/raw/main/img/img_v1.0.0-feedback.png" width="300px"> |
