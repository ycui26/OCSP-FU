# Online Client Selection and Pricing for AoI-aware Federated Unlearning with Private Information

Code repository for the paper "Online Client Selection and Pricing for AoI-aware Federated Unlearning with Private Information". 

---

## 📂 Repository Structure

The codebase is modularized to handle entirely different data modalities (Image and IoT tabular data) seamlessly.

```text
OCSP_FU/
├── data_utils/              # Data processing and loading modules
│   ├── preprocess_iot.py    # Offline preprocessing pipeline for raw CICIoT data
│   ├── cv_loader.py         # Partitioning & loading for MNIST/CIFAR
│   └── iot_loader.py        # Partitioning & loading for tabular IoT data
├── models/                  # Neural network architectures & local training logic
│   ├── cv_network.py        # CNN models for MNIST and CIFAR-100
│   └── iot_network.py       # Multi-Layer LSTM for tabular IoT data
├── utils/                   
│   └── helpers.py           # Core algorithmic functions (Client selection, aggregation)
├── main_cv.py               # Main execution script for Image datasets
├── main_iot.py              # Main execution script for IoT dataset
├── requirements.txt         # Python environment dependencies
└── README.md                # This file
```

## 🛠 Environment Setup

Install all the packages from requirments.txt

```
python3 -m venv fu_env
source fu_env/bin/activate  # On Windows use `fu_env\Scripts\activate`
pip install -r requirements.txt
```
## 💽 Dataset Preparation

### 1. Image Datasets (MNIST & CIFAR-100)

**No manual download is required.** The `torchvision` library will automatically download MNIST and CIFAR-100 into a local `./data/` folder the first time you run `main_cv.py`.

### 2. IoT Dataset (CICIoT)

Due to the massive size of the CICIoT dataset, we do not host the processed `.csv` files in this repository. Please follow these steps to prepare the data:

1. **Download the raw data:** Download the raw CICIoT dataset from its official source: [Kaggle CIC IoMT 2024 WiFi MQTT](https://www.kaggle.com/datasets/limamateus/cic-iomt-2024-wifi-mqtt).
2. **Place the data:** Create a `data/raw/` directory in this project root and place the downloaded files there.
3. **Run Preprocessing:** Execute the preprocessing script. This will clean the data, encode the features, and generate `processed_train.csv` and `processed_test.csv` inside the `./data/` directory.


```
python data_utils/preprocess_iot.py --raw_data_path ./data/raw/ --output_dir ./data/
```

## 🚀 Running Experiments

Once the environment and datasets are ready, you can start the training process.


```
python main_cv.py \
    --exp_name cv_cifar_baseline \
    --dataset cifar \
    --num_users 100 \
    --num_selections 200 \
    --epochs 5 \
    --perturb mislabel \
    --gpu 1
```
