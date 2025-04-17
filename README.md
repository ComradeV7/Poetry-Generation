# AI Poetry Generation System

A theme-aware poetry generation system that combines a BERT-based multi-label theme classifier with a fine-tuned GPT-2 language model to generate semantically relevant, stylistically coherent poetry conditioned on user-selected or inferred themes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Challenges Faced](#challenges-faced)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Poetry is a powerful medium for human expression, often revolving around deep themes such as love, loss, nature, and freedom. Traditional AI models struggle to generate poetry that consistently aligns with a specific theme or emotional core. This project aims to bridge that gap by:

- Automatically classifying poems into multiple themes using a BERT-based classifier.
- Fine-tuning GPT-2 to generate poetry conditioned on these themes.
- Preserving poetic structure and evaluating outputs using both quantitative and qualitative metrics.

**Real-world applications include:**
- Creative writing tools for poets and authors.
- Expressive writing for mental health and therapy.
- Educational aids for teaching poetic form and theme.
- Literature apps for personalized poetry generation.

---

## Features

- **Multi-Label Theme Classification:** Automatically tags poems with one or more semantic themes using BERT.
- **Theme-Conditioned Generation:** Fine-tuned GPT-2 generates poetry aligned with selected themes.
- **Poetic Structure Preservation:** Post-processing ensures readable, well-structured verse.
- **Evaluation Metrics:** Uses Perplexity, BERTScore, and thematic coherence for robust evaluation.
- **User Interface:** Gradio-based web UI for easy interaction and poem generation.

---

**Core Components:**
- Input Layer (user or dataset)
- Theme Classifier (BERT)
- Poem Generator (GPT-2)
- Post-Processing Module
- User Interface (Gradio)

---

## Dataset

- **Poetry Foundation Dataset:** [Hugging Face](https://huggingface.co/datasets/suayptalha/Poetry-Foundation-Poems)
- **Gutenberg Poetry Corpus:** [GitHub](https://github.com/aparrish/gutenberg-poetry-corpus)
- **Kaggle Poems Dataset:** [Kaggle](https://www.kaggle.com/datasets/michaelarman/poemsdataset)
- **All Modified Datasets:** [Google Drive](https://drive.google.com/drive/folders/1NbCQJdy23gxLhnDklWxrcZOLnRhR3TNx?usp=sharing)

---

## Installation

1. **Clone the repository:**
    ```
    git clone <your-repo-url>
    cd poetry-generation
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

    *Key packages used:*
    - transformers
    - torch
    - pandas
    - scikit-learn
    - wandb
    - gradio
    - bert-score
    - nltk
    - regex

3. **Download pre-trained models and datasets** as described in the Dataset section.

---

## Usage

1. **Training:**
   - Run the Jupyter notebook `NLPP3.ipynb` to preprocess data, train the BERT classifier, and fine-tune GPT-2.

2. **Generating Poems:**
   - Use the Gradio UI or the provided script to input a theme and generate a poem:
     ```
     import gradio as gr

     def generate_poem(theme):
         # Your theme-conditioned generation logic
         return poem

     gr.Interface(fn=generate_poem, inputs="text", outputs="text").launch()
     ```

3. **Evaluation:**
   - Evaluate generated poems using BERTScore, perplexity, and thematic coherence (see notebook cells for examples).

---

## Results

- **Theme Classifier F1-score:** ~0.89
- **GPT-2 Perplexity:** ~16.44 (Validation Loss: ~2.85)
- **BERTScore:** ~0.81 (semantic similarity)
- **Thematic Coherence:** ~83% (classifier accuracy on generated poems)

**Sample Output:**

![image](https://github.com/user-attachments/assets/d603afb2-b52e-44ba-bb99-f371cc10d917)
![image](https://github.com/user-attachments/assets/fbf5cb18-d09a-448d-835f-0fa50ed58069)


