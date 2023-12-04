# Quote Generation with GPT-2

## Overview

This program uses the GPT-2 language model to generate quotes based on the provided input. It is designed to run in Google Colab.

## Getting Started

### 1. Prerequisites

- Google Colab account

### 2. Setup

#### 2.1. Google Drive

Upload the files model.py and generate.py into the MyDrive location of your Google Drive(don't place it in a folder)

#### 2.2. Google Colab

- Open the [Quote Generation Colab Notebook](#) in Google Colab.
- Make sure the Google Drive in which you run the Colab Notebook is the same drive in which model.py and generate.py are stored
- For training, make sure to have your runtime type set to "T4 GPU"

### 3. Customize

- Adjust the number of quotes used to train the model by modifying the nrows variable in the pandas.read_csv cell
- Change k and length to play around with the quality of quotes generated

### 4. Run

Run the notebook cells sequentially to train the GPT-2 model and generate quotes.
When asked for user input, enter a couple words or a topic that will be the start off the generated quote.

## Tips

- Feel free to experiment with different prompts, hyperparameters, or GPT-2 model sizes for varied results.
- Larger datasets may improve model performance.

## Note

This README assumes basic familiarity with Google Colab.

Happy quoting!
