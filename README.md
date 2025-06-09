# 🎨 MNIST Handwritten Digit Generation using GAN

<div align="center">

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*Generating realistic handwritten digits using Generative Adversarial Networks*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-model-architecture) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 🚀 Project Overview

This project implements a **Generative Adversarial Network (GAN)** to generate realistic handwritten digit images using the famous MNIST dataset. The system consists of two competing neural networks:

- 🎭 **Generator**: Creates fake digit images from random noise
- 🕵️ **Discriminator**: Distinguishes between real and generated images

Through adversarial training, the generator learns to create increasingly realistic digits while the discriminator becomes better at detection, resulting in high-quality synthetic handwritten digits.

## ✨ Features

### 🧠 **Advanced Neural Architecture**
- **Generator**: Dense → Reshape → Conv2DTranspose layers with LeakyReLU
- **Discriminator**: Conv2D layers with dropout regularization
- **Optimized Training**: Adam optimizer with fine-tuned hyperparameters

### 📊 **Comprehensive Monitoring**
- Real-time loss tracking for both networks
- Discriminator accuracy metrics
- Periodic image generation and model checkpointing
- Visual progress tracking with generated digit grids

### 🎯 **Production Ready**
- Clean, modular code structure
- Configurable hyperparameters
- Automatic dataset handling
- Model persistence and loading

## 📦 Quick Start

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd mnist-gan
```

2. **Set up virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

**Run the training script:**
```bash
python mnist_gan.py
```

The model will:
- 📥 Automatically download MNIST dataset
- 🏋️ Start training with real-time progress updates
- 💾 Save generated images every 10 epochs
- 🔄 Create model checkpoints periodically

## 🏗️ Model Architecture

### Generator Network
```
Input: Random Noise (100D) 
    ↓
Dense Layer (7×7×128)
    ↓
LeakyReLU + Reshape
    ↓
Conv2DTranspose (14×14×128)
    ↓
Conv2DTranspose (28×28×128)
    ↓
Conv2D + Sigmoid → Output: (28×28×1)
```

### Discriminator Network
```
Input: Image (28×28×1)
    ↓
Conv2D + LeakyReLU + Dropout
    ↓
Conv2D + LeakyReLU + Dropout
    ↓
Flatten + Dense
    ↓
Sigmoid → Output: Real/Fake Probability
```

## 🎛️ Configuration

Customize training parameters:

```python
# Training Configuration
latent_dim = 100      # Noise vector dimension
n_epochs = 100        # Training epochs
n_batch = 256         # Batch size
learning_rate = 0.0002 # Adam optimizer learning rate
beta_1 = 0.5          # Adam optimizer momentum
```

## 📈 Results

### Training Progress
- **Loss Curves**: Monitor generator and discriminator losses
- **Accuracy Metrics**: Track discriminator performance on real vs fake images
- **Visual Evolution**: See generated digits improve over training epochs

### Output Files
```
📁 Project Directory
├── 🖼️ generated_plot_e10.png    # Generated digit grids
├── 🖼️ generated_plot_e20.png
├── 💾 generator_model_10.h5     # Saved generator models
├── 💾 generator_model_20.h5
└── 📊 Training logs in console
```

## 🔧 Customization Options

### Architecture Modifications
- Add **Batch Normalization** for training stability
- Experiment with **different activation functions**
- Modify **filter sizes and layer depths**
- Implement **spectral normalization**

### Advanced Features
- **Conditional GAN**: Generate specific digits on demand
- **Progressive Growing**: Start with low resolution and increase
- **Evaluation Metrics**: Implement FID, IS scores
- **Different Datasets**: Adapt for CIFAR-10, CelebA, etc.

### Hyperparameter Tuning
```python
# Experiment with these parameters
latent_dimensions = [50, 100, 200]
learning_rates = [0.0001, 0.0002, 0.0005]
batch_sizes = [128, 256, 512]
```

## 📊 Performance Monitoring

Track training progress with built-in metrics:

- **Generator Loss**: How well generator fools discriminator
- **Discriminator Loss**: How well discriminator detects fakes
- **Discriminator Accuracy**: Percentage of correct classifications
- **Visual Quality**: Inspect generated image grids

## 🛠️ Troubleshooting

### Common Issues

**Training Instability**
- Reduce learning rate
- Add noise to discriminator inputs
- Balance generator/discriminator training frequency

**Poor Image Quality**
- Increase training epochs
- Adjust network architecture
- Experiment with different loss functions

**Memory Issues**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

## 📁 Project Structure

```
mnist-gan/
├── 📄 mnist_gan.py           # Main training script
├── 📄 requirements.txt       # Dependencies
├── 📄 README.md             # This file
├── 📁 outputs/              # Generated images
│   ├── 🖼️ generated_plot_*.png
│   └── 💾 generator_model_*.h5
└── 📁 logs/                 # Training logs
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black mnist_gan.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Behrooz Filzadeh**
- 📧 Email: behrooz.filzadeh@gmail.com
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/behrooz-filzadeh)
- 🐙 GitHub: [Follow my work](https://github.com/behrooz-filzadeh)

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and CJ Burges
- **GAN Architecture**: Ian Goodfellow et al.
- **TensorFlow/Keras**: Google Brain Team
- **Community**: All contributors and users

## 📚 References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - Original GAN Paper
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Dataset Information
- [TensorFlow Documentation](https://tensorflow.org/tutorials/generative/dcgan) - DCGAN Tutorial

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ by Behrooz Filzadeh*

</div>
