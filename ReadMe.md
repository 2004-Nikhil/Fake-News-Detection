# Fake News Detection using ML (with Doc2Vec)

This project is a machine learning-based approach to detect fake news using the Doc2Vec model. The project is implemented in Python and uses various machine learning models for the detection of fake news.

## Project Structure

- [`app.py`](command:_github.copilot.openRelativePath?%5B%22app.py%22%5D "app.py"): This is the main application file where the machine learning models are trained and tested. It includes models like [`modelAttention`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelAttention%22%5D "app.py"), [`modelBiA`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelBiA%22%5D "app.py"), [`modelBiLSTM`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelBiLSTM%22%5D "app.py"), [`modelBiLSTMCNNa`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelBiLSTMCNNa%22%5D "app.py"), [`modelBiLSTM_CNN`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelBiLSTM_CNN%22%5D "app.py"), [`modelCNN`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelCNN%22%5D "app.py"), and [`modelGru`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22modelGru%22%5D "app.py").

- [`web.py`](command:_github.copilot.openRelativePath?%5B%22web.py%22%5D "web.py"): This file is used for the web interface of the application.

- [`Demonstration.ipynb`](command:_github.copilot.openRelativePath?%5B%22Demonstration.ipynb%22%5D "Demonstration.ipynb"): This Jupyter notebook contains a demonstration of how the models are trained and tested.

- [`datasets/`](command:_github.copilot.openRelativePath?%5B%22datasets%2F%22%5D "datasets/"): This directory contains the datasets used for training and testing the models. It includes `Fake.csv`, `True.csv`, and `Preprocessed_data.csv`.

- [`models/`](command:_github.copilot.openRelativePath?%5B%22models%2F%22%5D "models/"): This directory contains the trained models which are saved for future use. It includes models like `Attention.h5`, `BiA.h5`, `BiLSTM.h5`, `BiLSTMCNN.h5`, `CNN.h5`, `doc2vec.model`, `GRU.h5`, `LSTM.h5`, and others.

- [`templates/`](command:_github.copilot.openRelativePath?%5B%22templates%2F%22%5D "templates/"): This directory contains the HTML templates used for the web interface. It includes `index.html`.

## How to Run

To run the project, you need to execute the [`app.py`](command:_github.copilot.openRelativePath?%5B%22app.py%22%5D "app.py") or [`web.py`](command:_github.copilot.openRelativePath?%5B%22web.py%22%5D "web.py") file. Make sure you have all the necessary Python libraries installed.

```sh
python app.py
```

or

```sh
python web.py
```

## Contributing

Contributions are welcome. Please make sure to update tests as appropriate.