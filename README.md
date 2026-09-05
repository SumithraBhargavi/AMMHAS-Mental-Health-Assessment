# 🧠 AMMHAS — Mental Health Assessment System

An AI-powered multimodal framework for mental health assessment that combines text, audio, and visual emotion signals with machine learning and LLM-based interpretation.

## 🎥 Demo

▶️ **[Watch the 90-Second Demo Video](https://amritacampuschennai-my.sharepoint.com/personal/ch_sc_u4aie23029_ch_students_amrita_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fch%5Fsc%5Fu4aie23029%5Fch%5Fstudents%5Famrita%5Fedu%2FDocuments%2Fsoftware%20demo%20video%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E032ed771%2D918b%2D452f%2Db19f%2D5bc1878dcfe6)**

The demo showcases the application workflow and the system's machine-learning-based analysis and result generation.

---

## 📌 Overview

Mental health assessment can be challenging because emotional states are complex, contextual, and difficult to capture using a single source of information.

**AMMHAS (Adaptive Multimodal Mental Health Assessment System)** addresses this challenge by combining **text, audio, and visual information** to obtain richer emotional representations for session-level mental health assessment.

The framework separates **emotion prediction from interpretation**, allowing quantitative model outputs to be converted into human-readable, non-diagnostic explanations.

---

## ✨ Key Features

- 📝 Text-based emotion analysis
- 🎙️ Audio-based emotion analysis
- 🎥 Visual emotion analysis
- 🔗 Multimodal emotion fusion
- 📊 Emotion-distribution-based analysis
- 📈 Session-level assessment metrics
- 🤖 LLM-based human-readable interpretation
- ⚠️ Non-diagnostic AI-assisted assessment

---

## 🏗️ System Architecture

```text
                    MULTIMODAL INPUT
                  ┌───────┼────────┐
                  ▼       ▼        ▼
                Text    Audio     Video
                  │       │        │
                  ▼       ▼        ▼
               Bi-RNN   Bi-GRU   Bi-GRU
                  │       │        │
                  └───────┼────────┘
                          ▼
                 MLP Fusion Network
                          │
                          ▼
                7-D Emotion Distribution
                          │
                          ▼
                 Session-Level Analysis
                          │
                          ▼
                    LLaMA-3.1
                          │
                          ▼
              Human-Readable Explanation
