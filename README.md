# CIFAR-10 Image Classification with CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for multi-class image classification on the CIFAR-10 dataset. Achieves **75.46% test accuracy** using data augmentation and modern architectural techniques.

## Project Overview

This project demonstrates practical application of deep learning for computer vision tasks. The implemented CNN architecture features batch normalization, dropout regularization, and comprehensive data augmentation to achieve strong performance on the CIFAR-10 benchmark dataset.

### Key Features

- **Efficient Architecture**: Only 357K parameters (1.36 MB model size)
- **Data Augmentation**: Rotation, shifts, and horizontal flips for improved generalization
- **Modern Techniques**: Batch normalization, dropout, and Adam optimization
- **Comprehensive Analysis**: Training visualizations and performance metrics
- **Production-Ready**: Clean code structure with utility functions

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 75.46% |
| Total Parameters | 357,706 |
| Model Size | 1.36 MB |
| Training Time/Epoch | ~45 seconds |
| Improvement with Augmentation | +22.89% validation accuracy |

## Project Structure

```
cifar10-image-classification/
├── cifar10_classifier.py          # Main training script
├── utils.py                        # Utility functions (plot_samples, plot_history, etc.)
├── requirements.txt                # Python dependencies
├── cifar10_model.h5               # Saved trained model
├── cifar10_report.html            # Comprehensive project report
├── README.md                       # This file
└── screenshots/                    # Training visualizations
    ├── model_architecture.png
    ├── training_accuracy.png
    ├── training_loss.png
    └── predictions.png
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- 4GB RAM minimum (8GB recommended)
- GPU optional but recommended for faster training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cifar10-classification.git
cd cifar10-classification
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python cifar10_classifier.py
```

This will:
1. Download the CIFAR-10 dataset automatically
2. Train the model for 5 epochs (baseline)
3. Apply data augmentation and train for 5 more epochs
4. Save the trained model as `cifar10_model.h5`
5. Display training plots and predictions

### Custom Configuration

Modify hyperparameters in the script:

```python
# In cifar10_classifier.py

# Training parameters
BATCH_SIZE = 64
EPOCHS_BASELINE = 5
EPOCHS_AUGMENTED = 5
LEARNING_RATE = 0.001

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

### Load Trained Model

```python
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('cifar10_model.h5')

# Make predictions
predictions = model.predict(test_images)
```

## Architecture

### CNN Architecture (functional_1)

```
Input (32x32x3)
    ↓
[Block 1]
Conv2D(32, 3x3, ReLU, padding='same')
BatchNormalization
MaxPooling2D(2x2)
    ↓ (16x16x32)
[Block 2]
Conv2D(64, 3x3, ReLU, padding='same')
BatchNormalization
MaxPooling2D(2x2)
    ↓ (8x8x64)
[Block 3]
Conv2D(128, 3x3, ReLU, padding='same')
BatchNormalization
MaxPooling2D(2x2)
    ↓ (4x4x128)
[Classifier]
Flatten
Dense(128, ReLU)
Dropout(0.5)
Dense(10, Softmax)
    ↓
Output (10 classes)
```

**Total Parameters**: 357,706 (1.36 MB)

## Dataset

### CIFAR-10 Details

- **Total Images**: 60,000 (32x32 RGB)
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 (balanced distribution)

### Class Labels

| Index | Class |
|-------|-------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

## Performance Analysis

### Training Progression

**Original Model (without augmentation):**
- Training Accuracy: 70.00%
- Validation Accuracy: 52.57%
- Overfitting Gap: 17.43%

**With Data Augmentation:**
- Training Accuracy: 70.65%
- Validation Accuracy: 75.46%
- Overfitting Gap: -4.81% (better generalization)
- Loss Reduction: 56%

### Key Findings

1. **Data augmentation** provided a 42% relative improvement in validation accuracy
2. **Batch normalization** stabilized training across all epochs
3. Model shows **balanced predictions** across all 10 classes
4. **Efficient architecture** achieves competitive results with minimal parameters
5. Training speed: ~40-50ms per step on CPU

## Visualizations

The project generates the following plots:

- **Model Architecture Summary**: Layer-by-layer breakdown
- **Training/Validation Accuracy**: Epoch progression comparison
- **Training/Validation Loss**: Loss curves analysis
- **Sample Predictions**: Visual verification of model outputs
- **Class Distribution**: True vs predicted counts

## Dependencies

```txt
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Challenges & Solutions

### 1. Overfitting
**Challenge**: 17% gap between training and validation accuracy  
**Solution**: Implemented data augmentation and 0.5 dropout

### 2. Small Image Resolution
**Challenge**: 32x32 images provide limited spatial information  
**Solution**: Used `padding='same'` to preserve dimensions and multiple pooling layers

### 3. Training Stability
**Challenge**: Deep networks suffer from internal covariate shift  
**Solution**: Batch normalization after each convolutional layer

### 4. Computational Resources
**Challenge**: CPU training slower than GPU  
**Solution**: Optimized batch size (64) and efficient architecture

## Future Enhancements

- [ ] Implement ResNet or EfficientNet architectures
- [ ] Add learning rate scheduling (ReduceLROnPlateau)
- [ ] Extend training to 50-100 epochs with early stopping
- [ ] Ensemble multiple models for improved accuracy
- [ ] Advanced augmentation (Cutout, MixUp, AutoAugment)
- [ ] Transfer learning with pre-trained models
- [ ] Grad-CAM visualizations for model interpretation
- [ ] Web deployment with Flask/Streamlit interface

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Project Report

A comprehensive project report is available in `cifar10_report.html`. Open it in any browser to view:

- Detailed methodology
- Literature survey
- System architecture diagrams
- Complete results and analysis
- Challenges and solutions
- References and appendix

To generate PDF from the report:
1. Open `cifar10_report.html` in a browser
2. Click "Download PDF" or press `Ctrl+P`
3. Select "Save as PDF" as destination

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: CIFAR-10 by Alex Krizhevsky, Geoffrey Hinton (University of Toronto)
- **Framework**: TensorFlow/Keras team
- **Guide**: Mr. Akshay Bhalerao
- **Organization**: EmpowerYou Technologies
- **Institution**: G.H Raisoni College of Engineering and Management, Pune







## References

1. Krizhevsky, A., & Hinton, G. (2009). Learning Multiple Layers of Features from Tiny Images.
2. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training.
3. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
4. TensorFlow Documentation: https://www.tensorflow.org/

---

