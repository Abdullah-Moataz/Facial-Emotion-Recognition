# Facial Emotion Recognition Using Neural Networks

## Project Overview
This project uses a neural network to classify facial expressions into seven emotions using the FER2013 dataset.

The model was developed using Python and Keras, and a web application was created using Streamlit.

## Dataset
The FER2013 dataset contains grayscale facial images with a resolution of 48 × 48 pixels.

The seven emotion classes are:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset contains:
- **Training images:** 28,709
- **Testing images:** 3,589

## Data Preprocessing
- Resized images to 48 × 48 pixels.
- Converted images into NumPy arrays.
- Normalized pixel values between 0 and 1.
- Split training data into 80% training and 20% validation.

## Neural Network Architecture
The model uses a fully connected neural network:

- **Input:** 48 × 48 × 1
- **Flatten layer**
- **Dense layer:** 500 neurons (ReLU)
- **Dense layer:** 400 neurons (ReLU)
- **Dense layer:** 300 neurons (ReLU)
- **Output layer:** 7 neurons (Softmax)

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Epochs:** 10

## Model Evaluation
The initial model achieved approximately:
- **Training Accuracy:** 25%
- **Validation Accuracy:** 24%

The model currently has limited performance and tends to predict the Happy class more frequently.

## Streamlit Application
The application allows users to:
- Upload a facial image.
- Take a photo using their camera.
- Predict the facial expression.
- View prediction probabilities for all seven classes.

### Live Application
**Try the app here:**

https://facial-emotion-recognition-omkay9nccd8xapzfrfricm.streamlit.app/

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Google Colab

## How to Run Locally

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Run the Streamlit application:

   `streamlit run emotion.py`

## Limitations
- The model has relatively low prediction accuracy.
- Some emotion classes have fewer training images than others.
- Predictions may be affected by lighting, image quality, and facial angles.
- Facial expressions do not always reflect a person's actual emotions.

## Future Improvements
- Experiment with Convolutional Neural Networks (CNNs).
- Improve performance on underrepresented emotion classes.
- Add face detection before classification.
- Improve accuracy through additional training and evaluation.
