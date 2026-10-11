# SentimentScope: Transformer Sentiment Analysis

This project was completed as part of the **Udacity x AWS AI Programming with Python coursework**.

The goal of the project is to build and train a transformer-based model for binary sentiment classification on the IMDB movie review dataset. The model classifies movie reviews as either **positive** or **negative** and demonstrates how transformer architectures can be adapted from sequence modelling tasks to text classification.


## Project Objectives

* Load, explore, and prepare the IMDB movie review dataset.
* Analyse sentiment-label distribution and review-length characteristics.
* Tokenize text using the `bert-base-uncased` tokenizer.
* Implement a custom PyTorch `Dataset` and `DataLoader`.
* Adapt a transformer architecture for binary text classification.
* Implement model training, validation, and accuracy evaluation.
* Train a model that achieves more than **75% test accuracy**.
* Save the best-performing model checkpoint.


## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook
* CUDA / GPU training


## Dataset

The project uses the **IMDB Large Movie Review Dataset**, containing labelled movie reviews for binary sentiment classification.

Reviews are labelled as:

* `0` — Negative
* `1` — Positive

The dataset contains:

* 25,000 training reviews
* 25,000 test reviews

The original training dataset was further divided into:

* 22,500 training samples
* 2,500 validation samples


## Model Architecture

The project uses a custom transformer-based architecture implemented in PyTorch.

Key components include:

* Token embeddings
* Positional embeddings
* Multi-head self-attention
* Feed-forward neural networks
* Residual connections
* Layer normalization
* Mean pooling across token representations
* Linear classification head for binary sentiment prediction

Text is tokenized using the Hugging Face `bert-base-uncased` tokenizer with a maximum sequence length of 128 tokens.


## Training

The model was initially trained for three epochs using:

* Cross-entropy loss
* AdamW optimizer
* Batch size: 32
* Initial learning rate: `3e-4`

After the initial training stage, the best validation checkpoint was trained for two additional epochs using a lower learning rate of `1e-4`.

This improved validation performance while maintaining good generalization to the test dataset.


## Results

| Stage | Validation Accuracy |
| --- | ---: |
| Epoch 1 | 72.60% |
| Epoch 2 | 75.60% |
| Epoch 3 | 76.76% |
| Additional Epoch 1 | 79.36% |
| Additional Epoch 2 | **79.52%** |

Final model performance:

| Metric | Result |
| --- | ---: |
| Best Validation Accuracy | **79.52%** |
| Test Accuracy | **77.17%** |
| Project Requirement | >75% |

The final model successfully exceeded the required test accuracy threshold.


## Key Takeaways

1. Transformer architectures can be adapted for text classification by aggregating token-level representations into a sequence-level representation and passing it through a classification head.

2. Monitoring validation accuracy and saving the best-performing checkpoint provides a more reliable method for selecting a model that generalizes to unseen data.

3. Continuing training with a smaller learning rate improved validation performance after the initial training stage.


## Project Structure

Key files include:

* `SentimentScope_starter.ipynb` — completed project notebook containing data exploration, model implementation, training, evaluation, and conclusions.
* `best_sentimentscope_model.pt` — PyTorch model checkpoint corresponding to the best-performing validation model.
* `README.md` — project documentation.


## Coursework Attribution

This project was completed as part of the **Udacity x AWS AI Programming with Python coursework**.

The project brief, starter Jupyter notebook, project structure, instructions, rubric, selected transformer component definitions, dataset-loading framework, and instructional/hint code were provided as part of the course materials.

The original project framework includes starter implementations and scaffolding for components such as the transformer architecture and training workflow.

My work focused on completing, integrating, testing, and evaluating the required components, including:

* implementing dataset-loading functionality;
* exploring and visualizing the IMDB dataset;
* preparing training and validation datasets;
* implementing the custom PyTorch `IMDBDataset` class;
* configuring tokenization and data loading;
* completing the transformer model for binary classification;
* implementing mean pooling and the classification output layer;
* implementing model-accuracy calculation;
* completing the training and optimization loop;
* monitoring validation performance;
* implementing model checkpointing;
* continuing training with a reduced learning rate to improve performance;
* evaluating the final model on the test dataset; and
* summarizing and interpreting the project results.

This repository is shared as a record of my coursework and learning. Credit for the original project design, starter code, instructional materials, and provided project resources belongs to **Udacity, AWS, and the respective original authors**.
