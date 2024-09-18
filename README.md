# Enhancing Real-Time Voice Authentication
=====================================================

This project, developed as a senior project for California Polytechnic State University, aims to create a robust voice authentication mechanism for secure voice calls. The primary objective is to design a system that can accurately differentiate between genuine and spoofed audio clips, as well as identify individual speakers. This project serves as a proof of concept of a multi-model method of authenticating an individual through passive liveness detecion.

## Background
------------

Voice spoofing poses a major risk to systems that rely on voice authentication. Attackers can manipulate or fabricate voices to impersonate individuals, bypassing traditional security checks. This project utilizes deep learning techniques to mitigate this threat by detecting spoofed speech and verifying a speaker's identity in real-time during voice calls.

Objectives:
1. Develop a spoofed speech detection model to differentiate between genuine and spoofed audio.
2. Build a speaker verification model to authenticate the identity of a target speaker.

## Methodology
-------------

The ASVSpoof2019 dataset was selected for its relevance in detecting spoofed speech, offering a comprehensive range of labeled audio files that supported efficient model training and evaluation. Initial attempts with a custom Siamese network proved ineffective, leading to a pivot towards the Wav2Vec2 model, which excels in extracting detailed audio features for tasks like spoof detection. Wav2Vec2's pre-trained architecture enabled effective fine-tuning for distinguishing between genuine and spoofed speech, as well as target versus non-target speakers. Key tools like AWS SageMaker, Hugging Face, and Amazon S3 were leveraged to streamline model training, deployment, and feature extraction.

### Model 1: Wav2Vec2 Spoof Detection

* Fine-tuned to differentiate between genuine and spoofed audio clips
* Trained using a binary classification approach

### Model 2: Wav2Vec2 Speaker Identification

* Fine-tuned to differentiate target and non-target speakers
* Trained using a binary classification approach


## Results
--------

This project aimed to develop a real-time voice authentication system to combat the rising threat of spoofing scams in telecommunications. The primary objective was to leverage machine learning to provide speaker verification, particularly for identifying a target speaker in live voice calls, with a strong emphasis on detecting spoofed speech.

Within the project's timeframe, I was able to create two distinct but related models trained to perform binary classification tasks. The first model focused on detecting spoofed speech versus genuine speech, while the second model was designed to differentiate between a specific target speaker and non-target speakers. In evaluating the performance of the models developed for speaker authentication, significant insights emerge regarding their alignment with the initial project goal and their real-world applicability.

For the spoofed versus genuine speech model, the training results show a high degree of effectiveness in distinguishing between authentic and spoofed audio. The model demonstrated a substantial improvement in key metrics over time, reflecting its capability to accurately classify speech and detect spoofing attempts. The decrease in evaluation loss and the substantial increase in accuracy indicate that the model successfully refined its predictions and achieved near-perfect classification. Furthermore, high precision and recall values suggest that the model not only correctly identified spoofed speech with minimal false positives but also captured nearly all instances of spoofed speech, making it highly effective for the intended authentication task.

On the other hand, the target versus non-target speaker model exhibited notable improvements during training, though it faced some challenges. While there was a clear enhancement in performance metrics such as loss, accuracy, precision, and recall, the model did not reach the same level of effectiveness as the spoofed versus genuine model. The final performance, although better than initial results, indicates that the model still struggled with accurately identifying target speakers and may have missed some instances. This suggests that further refinement and additional training data could enhance its reliability.

