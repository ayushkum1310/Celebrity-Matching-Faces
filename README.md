# Celebrity Matching Faces

This project aims to develop a machine learning system capable of matching celebrity faces with high accuracy. The system processes raw image data, extracts features, trains various models, and evaluates their performance. The goal is to create a reliable and efficient tool for face recognition and matching in diverse datasets.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ayushkum1310/Celebrity-Matching-Faces.git
    cd Celebrity-Matching-Faces
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Prepare the data:
    ```bash
    make data
    ```

2. Train the model:
    ```bash
    make train
    ```

3. Run the application:
    ```bash
    python app.py
    ```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all the contributors and developers who made this project possible.

Made with ❤️ by Ayush Kumar  
Data Scientist

