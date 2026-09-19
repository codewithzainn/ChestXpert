# ChestXpert

**AI-Powered Chest CT Classification**

---

### Author
**M.Zain Ul Abideen**  
GitHub: [https://github.com/codewithzainn/](https://github.com/codewithzainn/)

---

## Overview

**ChestXpert** is an end-to-end Machine Learning web application designed to classify chest CT scan images into diagnosis categories (**Normal** vs. **Adenocarcinoma Cancer**). Built with a modern Flask web interface and leveraging transfer learning via a fine-tuned VGG16 Convolutional Neural Network, ChestXpert demonstrates a complete ML software engineering lifecycle—from automated data ingestion and DVC-managed pipeline execution to interactive local web inferencing and containerized deployment.

> **Disclaimer:** ChestXpert is an educational and portfolio demonstration project. It is **not** a clinical diagnostic tool and should not be used for medical diagnosis or clinical decision-making.

---

## Key Features

- **Interactive Web UI**: Single-page web application featuring custom drag-and-drop / file upload controls, instant Base64 preview, and async prediction execution.
- **Transfer Learning Architecture**: Utilizes pre-trained VGG16 weights (ImageNet) with custom dense classification layers optimized for chest CT scan analysis.
- **Reproducible ML Pipeline**: Managed with Data Version Control (DVC) to execute data ingestion, base model preparation, model training, and evaluation deterministically.
- **Modular Codebase**: Clean object-oriented package structure (`Respire`) separating components, configurations, entities, utilities, and training workflows.
- **Containerization Support**: Fully dockerized environment with `Dockerfile` and `docker-compose.yml` for isolated deployment across systems.

---

## How It Works

1. **Scan Input**: The user selects or uploads a chest CT scan image through the web interface.
2. **Preprocessing**: The client-side application encodes the image as a Base64 string and POSTs it to the Flask `/predict` endpoint.
3. **Image Decoding & Formatting**: The backend decodes the payload into `inputImage.jpg`, resizes the image to `224x224x3`, normalizes pixel tensors (`1./255`), and feeds it into the prediction pipeline.
4. **Model Inference**: The loaded VGG16 neural network evaluates the input tensor and outputs class probabilities.
5. **Result Display**: The backend returns the classification label (`Adenocarcinoma Cancer` or `Normal`), which is dynamically rendered in the web UI.

---

## Machine Learning Approach

ChestXpert employs **Transfer Learning** using the pre-trained **VGG16** Convolutional Neural Network architecture. Transfer learning enables high feature extraction performance on specialized medical imaging tasks by leveraging spatial representations learned from ImageNet, while training custom top classification layers on chest CT scan data.

---

## Model

- **Base Network**: VGG16 (ImageNet pre-trained weights, top classification layers frozen/excluded)
- **Input Shape**: `[224, 224, 3]`
- **Custom Classification Top**:
  - `Flatten()` layer
  - `Dense()` classification head with Softmax activation
- **Loss Function**: `Categorical Crossentropy`
- **Optimizer**: `SGD` (Stochastic Gradient Descent) with `learning_rate = 0.01`
- **Trained Model Location**: `Artifacts/Model_Training/Trained_Model.h5`

---

## Classification Classes

ChestXpert classifies chest CT scan cross-sections into two distinct classes:

1. **Adenocarcinoma Cancer**: CT scan images indicating presence of adenocarcinoma pulmonary tissue.
2. **Normal**: Healthy chest CT scan images exhibiting normal lung anatomy.

---

## Dataset

The model is trained on a Chest CT Scan image dataset organized into structured subdirectories for training and validation splits. Data ingestion is handled automatically via remote zip download configured in `Config/config.yaml`.

---

## Technology Stack

- **Core Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow 2.x / Keras
- **Web Backend**: Flask, Flask-CORS
- **ML Pipeline & Versioning**: DVC (Data Version Control), PyYAML
- **Data & Numeric Processing**: NumPy, Pandas, Pillow
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (Fetch API, FileReader)
- **Containerization**: Docker, Docker Compose
- **Experiment Tracking Integration**: MLflow / DagsHub (supported in evaluation pipeline)

---

## Project Architecture

```text
ChestXpert/
├── app.py                     # Flask application routes & server startup
├── main.py                    # Complete training pipeline runner script
├── dvc.yaml                   # DVC pipeline stages & dependencies definition
├── dvc.lock                   # DVC state lock file
├── params.yaml                # Hyperparameter specifications
├── setup.py                   # Package setup & dependency resolution
├── Dockerfile                 # Docker container image definition
├── docker-compose.yml         # Multi-container orchestration config
├── Config/
│   └── config.yaml            # Pipeline artifact & directory path configuration
├── Respire/                   # Main project Python package
│   ├── Components/            # Core ML stage handlers
│   │   ├── Data_Ingestion.py
│   │   ├── Base_Model.py
│   │   ├── Model_Trainer.py
│   │   └── Model_Evaluation.py
│   ├── Config/                # Configuration management logic
│   ├── Entity/                # Data classes & dataclass schemas
│   ├── Pipeline/              # Pipeline stage wrappers & PredictionPipeline
│   │   ├── Prediction_Pipeline.py
│   │   └── Training_Pipeline/
│   ├── Utils/                 # Utility helpers (image decoding, YAML parsing)
│   ├── Constants/             # Global constant definitions
│   └── Logger/                # Centralized logging configuration
├── templates/
│   └── index.html             # ChestXpert Web UI
├── Artifacts/                 # Pipeline output directory (models & unzipped data)
└── Notebook_Experiments/      # Jupyter notebook research files
```

---

## ML Pipeline

The machine learning workflow is divided into four distinct stages:

1. **Data Ingestion** (`Data_Ingestion.py`): Downloads the raw CT scan dataset archive, extracts files to `Artifacts/Data_Ingestion`, and prepares directory trees.
2. **Base Model Preparation** (`Base_Model.py`): Downloads VGG16 base weights, constructs the updated architecture with custom dense layers, and saves `Base_Model.h5` and `Updated_Model.h5`.
3. **Model Training** (`Model_Trainer.py`): Fits the updated model on CT scan training images using `ImageDataGenerator` data augmentation for the configured number of epochs.
4. **Model Evaluation** (`Model_Evaluation.py`): Computes loss and accuracy metrics on the validation dataset split and writes performance scores to `scores.json`.

---

## DVC (Data Version Control)

ChestXpert uses **DVC** to coordinate stage dependencies, outputs, and parameters across the pipeline.

To execute the entire end-to-end pipeline using DVC:

```bash
dvc repro
```

Each stage in `dvc.yaml` is evaluated against tracked file hashes in `dvc.lock` to avoid redundant re-computation.

---

## Model Training

Model hyperparameter configuration is specified centrally in `params.yaml`:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 2
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

To run training directly via Python:

```bash
python main.py
```

---

## Model Evaluation

After training execution, evaluation metrics are saved to `scores.json`:

```json
{
    "loss": 49.472511291503906,
    "accuracy": 0.0
}
```

The evaluation component can also log metrics directly to MLflow and DagsHub tracking platforms when configured.

---

## Web Application

The web interface is built with Flask and Vanilla CSS/JS. It provides an intuitive portal for CT image classification:

- **Endpoint `/` (GET)**: Serves the main single-page interface (`index.html`).
- **Endpoint `/predict` (POST)**: Receives a JSON payload `{"image": "<base64_string>"}`, decodes the image, runs model inference, and returns JSON `[{"image": "<class_label>"}]`.

---

## Project Structure

```text
ChestXpert
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── docker-compose.yml
├── dvc.lock
├── dvc.yaml
├── main.py
├── params.yaml
├── requirements.txt
├── scores.json
├── setup.py
├── template.py
├── Artifacts/
├── Config/
│   └── config.yaml
├── Notebook_Experiments/
├── Respire/
│   ├── Components/
│   ├── Config/
│   ├── Constants/
│   ├── Entity/
│   ├── Logger/
│   ├── Pipeline/
│   └── Utils/
├── docs/
└── templates/
    └── index.html
```

---

## Installation

### Prerequisites

- Python 3.8 or 3.9
- Git

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/codewithzainn/ChestXpert.git
   cd ChestXpert
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Local Package**:
   ```bash
   pip install -e .
   ```

---

## Running the Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:8080
```

Select a CT scan image and click **Analyze CT Scan** to view the classification result.

---

## Docker Usage

ChestXpert can be built and run inside a Docker container.

### Building the Docker Image

```bash
docker build -t chestxpert:latest .
```

### Running the Container

```bash
docker run -p 8080:8080 chestxpert:latest
```

### Using Docker Compose

```bash
docker compose up
```

Access the running application at `http://localhost:8080`.

---

## Example Prediction Flow

1. User uploads a chest CT image `scan01.jpg` in the UI.
2. Web UI converts image to Base64 data URI format.
3. Fetch request POSTs payload to `http://127.0.0.1:8080/predict`.
4. `app.py` receives request, calls `decodeImage(image_data, "inputImage.jpg")`.
5. `PredictionPipeline` loads `inputImage.jpg`, resizes tensor to `(1, 224, 224, 3)`.
6. Model `Trained_Model.h5` outputs argmax prediction index (`0` for Adenocarcinoma Cancer, `1` for Normal).
7. Flask responds with JSON result: `[{"image": "Adenocarcinoma Cancer"}]`.
8. UI updates result badge dynamically.

---

## Limitations

- **Educational Purpose Only**: This project is developed as a computer vision portfolio demonstration and software engineering reference implementation. It is **not** certified, validated, or intended for clinical diagnostics or medical use.
- **Domain Specificity**: The model is trained specifically on chest CT scan slice images. Inputting general photographs, X-rays, or non-CT medical scans will yield inaccurate predictions.
- **Binary Scope**: Current implementation is limited to binary classification (Adenocarcinoma Cancer vs. Normal).

---

## Future Improvements

- **Multi-Class CT Classification**: Expand dataset and classification head to cover additional pulmonary pathologies (e.g., Squamous Cell Carcinoma, Large Cell Carcinoma).
- **Explainable AI (Grad-CAM)**: Implement visual heatmap overlays to highlight feature regions driving neural network predictions.
- **Model Diversity**: Benchmark alternative backbones such as ResNet50, EfficientNet, and Vision Transformers (ViT).
- **Automated CI/CD**: Integrate GitHub Actions for continuous model evaluation and deployment.

---

## Attribution

**ChestXpert** is a redesigned, extended portfolio presentation built upon the open-source base project **[End-to-End-Chest-Disease-Classification](https://github.com/KalyanM45/End-to-End-Chest-Disease-Classification)** by **Hema Kalyan Murapaka (KalyanM45)**.

The underlying machine learning architecture, dataset structure, DVC stage framework, and core `Respire` package design originate from the original source. ChestXpert maintains full legal open-source credit while presenting a refreshed user interface, updated branding, clean project structure, and comprehensive documentation for educational and portfolio presentation.

---

## License

This project inherited its codebase from an open-source source licensed under the **GNU General Public License v3.0 (GPL-3.0)**. Refer to the original open-source license terms for complete details.
