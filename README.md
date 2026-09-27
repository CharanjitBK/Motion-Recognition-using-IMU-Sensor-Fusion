# Motion Recognition using IMU Sensor Fusion

Real-time human gesture recognition on a Raspberry Pi 4, using IMU data from the Sense HAT and a lightweight neural network deployed on-device with TensorFlow Lite.

## Overview

This project captures accelerometer and gyroscope data from a Raspberry Pi Sense HAT, trains a neural network to classify four distinct motion gestures, and deploys the trained model back onto the Raspberry Pi for real-time inference, with live visual feedback on the Sense HAT's LED matrix.

**Gestures classified:**
- `move_none` — no movement
- `move_circle` — circular motion
- `move_shake` — shaking motion
- `move_twist` — twisting motion

## Results

- **Training accuracy:** 100%
- **Validation accuracy:** 97.92%
- **Validation loss:** stabilized around 0.0795

The model converged quickly, reaching high training accuracy by the second epoch, with validation accuracy remaining stable throughout training, indicating good generalization without overfitting.

| Gesture | LED Feedback Color |
|---|---|
| Circular motion | Red |
| Shake | Green |
| Twist | Blue |
| No movement | Off / Colourless |

## Pipeline

```
IMU Data Collection → Preprocessing → Model Training → TFLite Conversion → Real-Time Inference (Raspberry Pi)
```

1. **Data Collection** — Motion data captured from the Sense HAT's 3-axis accelerometer and 3-axis gyroscope at 50 Hz. Each gesture instance consists of 50 timesteps × 6 sensor channels (`acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`), flattened into a 300-value feature vector.
2. **Preprocessing** — Raw sensor values are used directly with no additional normalization or filtering, sufficient given the consistency of data collected in a controlled environment.
3. **Model Training** — A feedforward neural network trained in TensorFlow (Google Colab).
4. **TFLite Conversion** — The trained Keras model is converted to a `.tflite` file for efficient, low-latency inference on embedded hardware.
5. **Real-Time Inference** — The Raspberry Pi loads the TFLite model, classifies incoming live sensor data, and displays the predicted gesture on the Sense HAT LED matrix.

## Model Architecture

```
Input(shape=(300,))
Dense(128, activation='relu')
Dropout(0.2)
Dense(64, activation='relu')
Dense(4, activation='softmax')
```

- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy
- **Epochs:** 15
- **Batch size:** 32
- **Train/validation split:** 80:20

## Tech Stack

- **Language:** Python
- **Libraries:** TensorFlow, NumPy, scikit-learn, Sense HAT API, `tflite_runtime`
- **Hardware:** Raspberry Pi 4 with Sense HAT (deployment); training performed on a PC (AMD Ryzen 7 5700U, Radeon Graphics)

## Repository Structure

```
├── data_collection.py       # Captures IMU samples from the Sense HAT
├── train_model.py           # Trains the neural network in TensorFlow
├── convert_to_tflite.py     # Converts the trained model to TensorFlow Lite
├── real_time_inference.py   # Runs live inference + LED feedback on the Pi
├── gesture_model.tflite     # Trained, converted model
├── datasets/                # Collected motion datasets
└── README.md
```

*(Adjust file names above to match your actual repository if they differ.)*

## Usage

```bash
# 1. Collect gesture data (run on Raspberry Pi)
python data_collection.py

# 2. Train the model (run in Google Colab or locally)
python train_model.py

# 3. Convert the trained model to TensorFlow Lite
python convert_to_tflite.py

# 4. Deploy and run real-time inference (run on Raspberry Pi)
python real_time_inference.py
```

## Challenges & Limitations

- Synchronizing real-time data capture and classification on the Raspberry Pi.
- Distinguishing between similar motion patterns, particularly *shake* vs. *twist*.
- Occasional misclassification caused by IMU sensor drift and inconsistent gesture speed across trials.
- Limited to four predefined gestures; the model does not generalize to new users without retraining.
- Real-time inference performance is bounded by the Raspberry Pi's CPU capabilities.

## Future Improvements

- Explore recurrent architectures (e.g., LSTM) to better capture temporal dependencies in the IMU time-series data, rather than treating each gesture as a flattened static vector.
- Add preprocessing (normalization/filtering) to improve robustness across different users and conditions.
- Expand the gesture set beyond the current four classes.

## References

- [TensorFlow Documentation](https://www.tensorflow.org)
- [Sense HAT API Documentation](https://pythonhosted.org/sense-hat/)

## Author

**Charanjit Bangalore Kumar**
Completed as part of the Embedded Systems course (Lab 05), Technische Hochschule Deggendorf.
