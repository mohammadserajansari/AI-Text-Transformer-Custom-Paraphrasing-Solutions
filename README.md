# Paraphrase Generator

This repository contains a Python application for generating paraphrases using T5-base and GPT-2 models. The application provides a web interface for interacting with the models and generating paraphrased text.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp.streamlit.app/)

## Overview

The application performs the following steps:

1. Loads pre-trained T5-base and GPT-2 models.
2. Uses these models to generate paraphrases of input text.
3. Provides a web interface via Streamlit for users to interact with the paraphrase generator.

## Installation

To set up the environment and install the required libraries, follow these steps:

1. Create a virtual environment and activate it:
    ```bash
    conda create -n pyt python=3.10
    ```
    ```bash
    conda activate pyt
    ```

2. Install PyTorch:
   - **For Linux:**
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - **For Other Operating Systems:**
     Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install PyTorch for your specific operating system and hardware.

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit script:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open the web interface to interact with the paraphrase generator:
    - Enter your text into the provided input field.
    - The application will generate and display paraphrased versions of the input text.

   Example:
