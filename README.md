**Traffic Sign Detection using CNN and MobileNet**

This project implements traffic sign classification using deep learning, featuring two approaches:

Custom Convolutional Neural Network (CNN) trained from scratch
Transfer Learning with MobileNet, a lightweight and efficient pretrained model
Both models are trained and evaluated on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

**Overview:**

Implemented using TensorFlow and Keras
Full training and testing done in a single script for each model
Achieves high accuracy on real-world traffic sign data
Clean, modular structure with separate folders for each approach

### Project Structure

```
traffic-sign-detection/
├── cnn_model/
│   └── cnn_traffic_sign.py
├── mobilenet_model/
│   └── mobilenet_traffic_sign.py
├── README.md
├── requirements.txt
```


**Dataset:**

This project uses the GTSRB dataset available on Kaggle:
Download here: https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed

**After downloading:**
Unzip the dataset
Place it in your project folder (e.g., data/ directory)
How to Run

**Install dependencies:**
pip install -r requirements.txt

**Run the CNN model:**
python cnn_model/cnn_traffic_sign.py

**Run the MobileNet model:**
python mobilenet_model/mobilenet_traffic_sign.py

**Results (Example)**
Model	Description	Accuracy
CNN	Built from scratch	~99%
MobileNet	Transfer learning (ImageNet)	~93%
