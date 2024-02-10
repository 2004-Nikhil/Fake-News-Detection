# Fake News Detection using ML (with Doc2Vec)

This project is a machine learning-based approach to detect fake news using the Doc2Vec model. The project is implemented in Python and uses various machine learning models for the detection of fake news.

## Project Structure

- [`app.py`]: This is the main application file where the machine learning models are trained and tested. It includes models like [`modelAttention`], [`modelBiA`], [`modelBiLSTM`], [`modelBiLSTMCNNa`], [`modelBiLSTM_CNN`], [`modelCNN`], and [`modelGru`].

- [`web.py`]: This file is used for the web interface of the application.

- [`Demonstration.ipynb`]: This Jupyter notebook contains a demonstration of how the models are trained and tested.

- [`datasets/`]: This directory contains the datasets used for training and testing the models. It includes `Fake.csv`, `True.csv`, and `Preprocessed_data.csv`.

- [`models/`]: This directory contains the trained models which are saved for future use. It includes models like `Attention.h5`, `BiA.h5`, `BiLSTM.h5`, `BiLSTMCNN.h5`, `CNN.h5`, `doc2vec.model`, `GRU.h5`, `LSTM.h5`, and others.

- [`templates/`]: This directory contains the HTML templates used for the web interface. It includes `index.html`.

## How to Run

To run the project, you need to execute the [`app.py`] or [`web.py`] file. Make sure you have all the necessary Python libraries installed.

```sh
python app.py
```

or

```sh
python web.py
```

## Contributing

Contributions are welcome. Please make sure to update tests as appropriate.
