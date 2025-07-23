**Traffic Sign Detection**

This project presents a deep learning-based image classification system to detect and classify traffic signs using camera-captured images. Leveraging both a custom Convolutional Neural Network (CNN) and the MobileNet architecture, this project aims to support autonomous driving systems by accurately recognizing road signs.

**🌐 Project Overview**

The classification models are trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset containing thousands of traffic sign images across 43 categories. The pipeline includes image preprocessing, model training using both CNN and MobileNet, and performance evaluation based on key metrics.

**📃 Dataset**

Source: GTSRB - German Traffic Sign Recognition Benchmark

**Description:**
Over 50,000 labeled images of traffic signs
43 distinct classes (e.g., Speed limits, Stop, Yield, No Entry)
Includes variations in lighting, angle, and occlusion to simulate real driving scenarios

**📊 Features**

- **🔧 CNN Model Architecture:**
       - Layers: Conv2D, MaxPooling2D, Dropout, Flatten, Dense
       - Trained from scratch for classification

- **🔄 MobileNet Transfer Learning:**
       - Uses pre-trained MobileNet (on ImageNet)
       - Fine-tuned on traffic sign images
       - Faster training and higher accuracy with fewer parameters

**🛠️ Data Preprocessing:**
- Image resizing and normalization
- One-hot encoding of labels
- Train-test split using sklearn
  
**📈 Evaluation Metrics:**
- Confusion matrix
- Accuracy and loss plots
- Classification report with precision, recall, and F1-score

**🎓 Technologies Used**

- Python
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn
**🚀 How to Run the Project**

**Clone the repository:**
```
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection
```
**Install dependencies:**
```
pip install -r requirements.txt
Download the dataset from Kaggle and place it inside a data/ folder.
```
**Run CNN model:**
```
python cnn_model/cnn_traffic_sign.py
```
**Run MobileNet model:**
```
python mobilenet_model/mobilenet_traffic_sign.py
```

**🖼️ Sample Results**

- Visualizations of accuracy and loss over epochs
- Confusion matrix for classification analysis
- MobileNet model achieves higher accuracy with lower training time

**🙌 Acknowledgments**

- Dataset courtesy of Kaggle (GTSRB)
- Inspired by real-time traffic sign recognition needs in autonomous vehicles
