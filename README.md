# Voice Authentication Mechanism for Secure Voice Calls
=====================================================

This project, developed as a senior project for California Polytechnic State University, aims to create a robust voice authentication mechanism for secure voice calls. The primary objective is to design a system that can accurately differentiate between genuine and spoofed audio clips, as well as identify individual speakers. This project serves as a proof of concept of a multi-model method of authenticating an individual through passive liveness detecion.

## Background
------------

Voice authentication is a critical aspect of secure communication, particularly in voice calls. Spoofed audio can compromise the security and integrity of voice calls, making it essential to develop a reliable authentication mechanism. The ASV Spoof Challenge, a widely recognized benchmark for voice authentication, provides a comprehensive dataset for training and evaluating voice authentication models.

## Methodology
-------------

To achieve the project's objectives, I employed a machine learning-based approach, leveraging the Wav2Vec model, a state-of-the-art speech processing model developed by Facebook. I fine-tuned two pre-trained Wav2Vec models on the ASV Spoof Challenge dataset, which contains a diverse range of genuine and spoofed audio clips.

### Model 1: Wav2Vec2 Spoof Detection

* Fine-tuned to differentiate between genuine and spoofed audio clips
* Trained using a binary classification approach

### Model 2: Wav2Vec2 Speaker Identification

* Fine-tuned to differentiate target and non-target speakers
* Trained using a binary classification approach

## Implementation
--------------

The implementation of the project involved the following steps:

1. **Data Preprocessing**: The ASV Spoof Challenge dataset was preprocessed to extract relevant features, such as mel-frequency cepstral coefficients (MFCCs) and spectrograms.
2. **Model Fine-tuning**: The pre-trained Wav2Vec models were fine-tuned on the preprocessed dataset using a combination of transfer learning and domain adaptation techniques.
3. **Model Evaluation**: The fine-tuned models were evaluated using metrics such as accuracy, precision, and recall.

## Results
--------

The fine-tuned Wav2Vec models demonstrated excellent performance in differentiating between genuine and spoofed audio clips, as well as identifying individual speakers. The results showed that the models can achieve high accuracy, precision, and recall, indicating their potential for real-world applications.

## Future Work
-------------

While the project has achieved promising results, there are several avenues for future work:

1. **Model Optimization**: Further optimization of the fine-tuned models can be explored to improve their performance.
2. **Multi-Modal Fusion**: Fusing multiple modalities, such as audio and visual features, can enhance the robustness of the voice authentication mechanism.
3. **Real-World Deployment**: The developed models can be integrated into real-world applications, such as voice assistants or secure communication systems.

## Conclusion
----------

This project demonstrates the potential of fine-tuned Wav2Vec models for voice authentication. The developed models can accurately differentiate between genuine and spoofed audio clips, as well as identify the target speaker. The project's results have implications for secure voice calls and can be extended to various applications, including voice assistants and secure communication systems.
