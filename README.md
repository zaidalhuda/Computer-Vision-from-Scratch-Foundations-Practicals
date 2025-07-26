# 🔬 Computer Vision from Scratch: Foundations & Practicals

> *"The field of machine vision is swiftly evolving"* - From traditional rule-based approaches to cutting-edge deep learning architectures

[![Course Status](https://img.shields.io/badge/Status-Active-brightgreen)]() [![License](https://img.shields.io/badge/License-MIT-blue)]() [![Python](https://img.shields.io/badge/Python-3.8+-orange)]() [![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-red)]()

## 🎯 Course Overview

This comprehensive computer vision course takes you from the fundamentals of image processing to advanced generative models and real-world deployment. Born from a decade of research experience, starting with traditional rice grain detection algorithms and evolving into modern deep learning approaches.

### 🌟 What Makes This Course Special?

- **Hands-on Evolution Journey**: Witness the transformation from rule-based to ML-based computer vision
- **Production-Ready Skills**: Learn not just theory, but practical deployment and optimization
- **Complete Pipeline**: From data collection to model serving in production
- **Modern Architectures**: Cover everything from CNNs to Vision Transformers
- **Real Applications**: Object detection, segmentation, generative models, and more

---

## 📚 Course Structure

The course is designed as a **two-part journey** that builds from foundations to advanced applications:

### 🔥 **PART 1: Computer Vision Foundations** 
*Modules 1-6 | Building Your Core Knowledge*

### ⚡ **PART 2: Computer Vision Practicals** 
*Modules 7-12 | Real-World Applications & Deployment*

---

## 📋 Quick Navigation

**Jump to Module:** [1](#module-1) | [2](#module-2) | [3](#module-3) | [4](#module-4) | [5](#module-5) | [6](#module-6) | [7](#module-7) | [8](#module-8) | [9](#module-9) | [10](#module-10) | [11](#module-11) | [12](#module-12)

---

## 🗺️ Detailed Curriculum

### <a id="module-1"></a>**[📂 Module 1: Machine Learning for Computer Vision](./module-01-ml-for-cv/)**
**🎯 [Lecture 1: Rule-based vs ML-based approaches](./module-01-ml-for-cv/lecture-01-rule-vs-ml.ipynb)**
- Introduction to computer vision landscape
- Evolution from rule-based to ML approaches  
- Inception of AlexNet and the deep learning revolution
- Differences between ML and traditional programming paradigms
- Deep learning use cases in computer vision

📖 **Additional Resources:** [📄 Reading Materials](./module-01-ml-for-cv/readings/) | [💻 Code Examples](./module-01-ml-for-cv/code/) | [📊 Slides](./module-01-ml-for-cv/slides/)

---

### <a id="module-2"></a>**[📂 Module 2: Let's Build Some Basic Models](./module-02-basic-models/)**

**🔧 [Lecture 2: Building a simple linear model (no activation function)](./module-02-basic-models/lecture-02-linear-model.ipynb)**
- Defining and preparing datasets
- Reading images, scaling, and resizing techniques
- Introduction to TensorFlow ecosystem
- Building your first linear classification model

**🧠 [Lecture 3: Building a simple Neural network (no convolution)](./module-02-basic-models/lecture-03-neural-network.ipynb)**
- Understanding linear vs non-linear models
- Hidden layers and activation functions
- Gradient descent, backpropagation and optimizers
- Hyperparameter tuning strategies
- Defining and training neural networks
- Testing performance on custom datasets

**⚠️ [Lecture 4: Overfitting](./module-02-basic-models/lecture-04-overfitting.ipynb)**
- Understanding the overfitting phenomenon
- L1 and L2 regularization, dropout, early stopping
- Using validation datasets for monitoring
- Balancing model complexity with dataset size

📖 **Additional Resources:** [📄 Reading Materials](./module-02-basic-models/readings/) | [💻 Code Examples](./module-02-basic-models/code/) | [📊 Slides](./module-02-basic-models/slides/) | [🎯 Assignments](./module-02-basic-models/assignments/)

---

### <a id="module-3"></a>**[📂 Module 3: Convolutional Neural Networks](./module-03-cnns/)**

**🔍 [Lecture 5: What is a Convolutional Neural Network?](./module-03-cnns/lecture-05-cnn-intro.ipynb)**
- Convolutional filters and local receptive fields
- Parameter sharing for efficient image feature extraction
- Historical development of CNN theory
- Kernel size, stride, and padding mechanics
- Max pooling vs average pooling strategies

**🏗️ [Lecture 6: Historical CNN architectures](./module-03-cnns/lecture-06-historical-cnns.ipynb)**
- AlexNet breakthrough and its impact
- Key innovations in VGG and Inception architectures
- Role of competition datasets like ImageNet
- Architecture evolution timeline

**🔬 [Lecture 7: Deeper networks — ResNet and DenseNet](./module-03-cnns/lecture-07-deep-networks.ipynb)**
- Skip connections and residual block concepts
- Trade-offs between network depth and width
- Dense connectivity patterns and efficiency
- Performance vs accuracy considerations

**🎯 [Lecture 8: Transfer learning and fine-tuning](./module-03-cnns/lecture-08-transfer-learning.ipynb)**
- Loading pre-trained models for feature extraction
- Full fine-tuning vs partial layer freezing
- Learning rate scheduling and differential learning rates

📖 **Additional Resources:** [📄 Reading Materials](./module-03-cnns/readings/) | [💻 Code Examples](./module-03-cnns/code/) | [📊 Slides](./module-03-cnns/slides/) | [🎯 Assignments](./module-03-cnns/assignments/)

---

### <a id="module-4"></a>**[📂 Module 4: Vision Transformer](./module-04-vision-transformer/)**

**🤖 [Lecture 9: Vision transformer theory](./module-04-vision-transformer/lecture-09-vit-theory.ipynb)**
- Why transformers can replace convolutions in some cases
- Attention mechanisms adapted for computer vision
- Strengths and limitations of transformer-based architectures

**⚙️ [Lecture 10: Vision transformer application](./module-04-vision-transformer/lecture-10-vit-application.ipynb)**
- Practical implementation steps for training ViT models
- Adapting existing transformer libraries and checkpoints
- Performance comparison with CNN approaches

📖 **Additional Resources:** [📄 Reading Materials](./module-04-vision-transformer/readings/) | [💻 Code Examples](./module-04-vision-transformer/code/) | [📊 Slides](./module-04-vision-transformer/slides/) | [🎯 Assignments](./module-04-vision-transformer/assignments/)

---

### <a id="module-5"></a>**[📂 Module 5: Object Detection](./module-05-object-detection/)**

**📍 [Lecture 11: Intro to object detection](./module-05-object-detection/lecture-11-detection-intro.ipynb)**
- Bounding boxes and intersection over union metrics
- Classical detection vs deep learning methods
- Overview of relevant benchmark datasets

**⚡ [Lecture 12: YOLO architecture and training](./module-05-object-detection/lecture-12-yolo.ipynb)**
- Real-time detection principles and optimizations
- Anchor boxes and label format considerations
- Scaling YOLO for different application domains

**🎯 [Lecture 13: RetinaNet and focal loss](./module-05-object-detection/lecture-13-retinanet.ipynb)**
- Addressing class imbalance in detection tasks
- Single-stage detection improvements and innovations
- Understanding and implementing focal loss functions

📖 **Additional Resources:** [📄 Reading Materials](./module-05-object-detection/readings/) | [💻 Code Examples](./module-05-object-detection/code/) | [📊 Slides](./module-05-object-detection/slides/) | [🎯 Assignments](./module-05-object-detection/assignments/)

---

### <a id="module-6"></a>**[📂 Module 6: Image Segmentation](./module-06-image-segmentation/)**

**🧩 [Lecture 14: Fundamentals of image segmentation](./module-06-image-segmentation/lecture-14-segmentation-intro.ipynb)**
- Semantic segmentation vs instance segmentation approaches
- Evaluation metrics: mIoU and Dice coefficient
- Classical segmentation vs neural network approaches

**🏥 [Lecture 15: U-Net and Mask R-CNN](./module-06-image-segmentation/lecture-15-unet-maskrcnn.ipynb)**
- Encoder-decoder design patterns in U-Net architecture
- Instance segmentation with Mask R-CNN framework
- Practical applications and labeling challenges

📖 **Additional Resources:** [📄 Reading Materials](./module-06-image-segmentation/readings/) | [💻 Code Examples](./module-06-image-segmentation/code/) | [📊 Slides](./module-06-image-segmentation/slides/) | [🎯 Assignments](./module-06-image-segmentation/assignments/)

---

### <a id="module-7"></a>**[📂 Module 7: Creating Vision Datasets](./module-07-creating-datasets/)**

**📊 [Lecture 16: Dataset collection and labeling](./module-07-creating-datasets/lecture-16-data-collection.ipynb)**
- Collecting images from various sources and platforms
- Manual labeling strategies for classification and detection
- Multilabel tasks and bounding box considerations
- Crowdsourcing and large-scale labeling services

**🤖 [Lecture 17: Automated labeling and addressing bias](./module-07-creating-datasets/lecture-17-automated-labeling.ipynb)**
- Generating labels from related data and self-supervised learning
- Noisy student approach and semi-supervised techniques
- Recognizing selection bias and measurement bias
- Dataset splitting strategies and minimizing data leakage

📖 **Additional Resources:** [📄 Reading Materials](./module-07-creating-datasets/readings/) | [💻 Code Examples](./module-07-creating-datasets/code/) | [📊 Slides](./module-07-creating-datasets/slides/) | [🎯 Assignments](./module-07-creating-datasets/assignments/)

---

### <a id="module-8"></a>**[📂 Module 8: Data Preprocessing](./module-08-data-preprocessing/)**

**🔧 [Lecture 18: Data quality and transformations](./module-08-data-preprocessing/lecture-18-data-quality.ipynb)**
- Image resizing, cropping, and color space conversions
- Ensuring consistent aspect ratios across datasets
- Common pitfalls in preprocessing pipelines

**📈 [Lecture 19: Data augmentation and training-serving consistency](./module-08-data-preprocessing/lecture-19-data-augmentation.ipynb)**
- Random flips, rotations, and color distortion techniques
- Information dropping strategies (cutout, mixup)
- Avoiding training-serving skew in production
- Integrating preprocessing in models vs external scripts

📖 **Additional Resources:** [📄 Reading Materials](./module-08-data-preprocessing/readings/) | [💻 Code Examples](./module-08-data-preprocessing/code/) | [📊 Slides](./module-08-data-preprocessing/slides/) | [🎯 Assignments](./module-08-data-preprocessing/assignments/)

---

### <a id="module-9"></a>**[📂 Module 9: Training Pipeline](./module-09-training-pipeline/)**

**⚡ [Lecture 20: Efficient data ingestion](./module-09-training-pipeline/lecture-20-data-ingestion.ipynb)**
- Storing data in optimized tfrecord format
- Parallel reads, caching, and sharding strategies
- Maximizing GPU utilization during training

**🌐 [Lecture 21: Distributing training](./module-09-training-pipeline/lecture-21-distributed-training.ipynb)**
- Data parallelism with multiple GPUs
- Mirrored and multiworker training strategies
- Introduction to TPUs and their advantages

**🔄 [Lecture 22: Checkpoints and automated workflows](./module-09-training-pipeline/lecture-22-checkpoints-workflows.ipynb)**
- Checkpointing best practices for training resilience
- Model export using SavedModel and deployment formats
- Hyperparameter tuning with serverless pipeline automation

📖 **Additional Resources:** [📄 Reading Materials](./module-09-training-pipeline/readings/) | [💻 Code Examples](./module-09-training-pipeline/code/) | [📊 Slides](./module-09-training-pipeline/slides/) | [🎯 Assignments](./module-09-training-pipeline/assignments/)

---

### <a id="module-10"></a>**[📂 Module 10: Model Quality and Continuous Evaluation](./module-10-model-quality/)**

**📊 [Lecture 23: Monitoring training and debugging](./module-10-model-quality/lecture-23-training-monitoring.ipynb)**
- Using TensorBoard for metrics and visualizations
- Detecting anomalies in gradients and loss curves
- Interpreting weight histograms for debugging

**📏 [Lecture 24: Metrics for classification, detection, and segmentation](./module-10-model-quality/lecture-24-evaluation-metrics.ipynb)**
- Accuracy, precision, recall, and F1 score analysis
- ROC curves, AUC, and confusion matrix interpretation
- Intersection over union and mean IoU calculations

**⚖️ [Lecture 25: Ongoing evaluation, bias, and fairness](./module-10-model-quality/lecture-25-bias-fairness.ipynb)**
- Sliced evaluations for different subpopulations
- Measuring bias in model outcomes and predictions
- Setting up continuous evaluation in production environments

📖 **Additional Resources:** [📄 Reading Materials](./module-10-model-quality/readings/) | [💻 Code Examples](./module-10-model-quality/code/) | [📊 Slides](./module-10-model-quality/slides/) | [🎯 Assignments](./module-10-model-quality/assignments/)

---

### <a id="module-11"></a>**[📂 Module 11: Model Predictions and Deployment](./module-11-model-deployment/)**

**🚀 [Lecture 26: Prediction workflows](./module-11-model-deployment/lecture-26-prediction-workflows.ipynb)**
- Batch prediction using Apache Beam for large datasets
- Real-time serving with TensorFlow Serving and REST APIs
- Handling pre- and post-processing at inference time

**📱 [Lecture 27: Edge deployment](./module-11-model-deployment/lecture-27-edge-deployment.ipynb)**
- Model compression and quantization strategies
- TensorFlow Lite for mobile and embedded devices
- Overview of federated learning and privacy considerations

**🔮 [Lecture 28: Trends in production ML](./module-11-model-deployment/lecture-28-production-ml-trends.ipynb)**
- Pipeline orchestration and automation with Kubeflow
- Explainability methods: Grad-CAM and saliency maps
- Comparing no-code solutions to custom development approaches

📖 **Additional Resources:** [📄 Reading Materials](./module-11-model-deployment/readings/) | [💻 Code Examples](./module-11-model-deployment/code/) | [📊 Slides](./module-11-model-deployment/slides/) | [🎯 Assignments](./module-11-model-deployment/assignments/)

---

### <a id="module-12"></a>**[📂 Module 12: Advanced Vision Problems and Generative Models](./module-12-advanced-topics/)**

**📐 [Lecture 29: Advanced object measurement and pose estimation](./module-12-advanced-topics/lecture-29-measurement-pose.ipynb)**
- Ratio-based measurement using reference objects
- Counting objects via density estimation techniques
- Keypoint detection and multi-person pose setups

**🔍 [Lecture 30: Image retrieval and search](./module-12-advanced-topics/lecture-30-image-retrieval.ipynb)**
- Building image embeddings for similarity search
- Large-scale indexing and retrieval methods
- Practical considerations for dimensionality reduction

**🎨 [Lecture 31: Autoencoders and Generative Adversarial Networks](./module-12-advanced-topics/lecture-31-autoencoders-gans.ipynb)**
- Autoencoder architectures for reconstruction and anomaly detection
- Introduction to GANs for image-to-image translation
- Super-resolution and image inpainting applications

**🌊 [Lecture 32: Image captioning and multimodal learning](./module-12-advanced-topics/lecture-32-image-captioning.ipynb)**
- Image-to-text pipeline fundamentals and architecture
- Combining CNN and transformer-based language models
- Future directions in multimodal AI research

**🎓 [Lecture 33: Course summary](./module-12-advanced-topics/lecture-33-course-summary.ipynb)**
- Comprehensive recap of entire course modules
- Major concepts and practical takeaways from each module
- Common pitfalls and strategies for overcoming them
- Recommendations for further learning and advanced reading
- How to continue building real-world computer vision applications

📖 **Additional Resources:** [📄 Reading Materials](./module-12-advanced-topics/readings/) | [💻 Code Examples](./module-12-advanced-topics/code/) | [📊 Slides](./module-12-advanced-topics/slides/) | [🎯 Assignments](./module-12-advanced-topics/assignments/)

---

## 🛠️ Prerequisites

- **Programming**: Intermediate Python knowledge
- **Mathematics**: Basic linear algebra and calculus
- **Machine Learning**: Understanding of basic ML concepts (optional but helpful)
- **Hardware**: GPU access recommended for practical exercises

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/computer-vision-from-scratch.git
   cd computer-vision-from-scratch
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start with Module 1**
   ```bash
   cd module-1-ml-for-cv
   jupyter notebook
   ```

## 🎯 Learning Outcomes

By completing this course, you will:

- ✅ **Master the fundamentals** of computer vision and deep learning
- ✅ **Build production-ready** CV models from scratch
- ✅ **Deploy models** to cloud, mobile, and edge devices
- ✅ **Handle real-world challenges** like bias, data quality, and scaling
- ✅ **Stay current** with latest architectures like Vision Transformers
- ✅ **Apply CV** to diverse domains: detection, segmentation, generation


## 📈 Progress Tracking

Track your learning progress through the course:

- [ ] **Module 1-2**: ML Foundations & Basic Models
- [ ] **Module 3**: Convolutional Neural Networks  
- [ ] **Module 4**: Vision Transformers
- [ ] **Module 5**: Object Detection
- [ ] **Module 6**: Image Segmentation
- [ ] **Module 7-8**: Dataset Creation & Preprocessing
- [ ] **Module 9**: Training Pipeline
- [ ] **Module 10**: Model Quality & Evaluation
- [ ] **Module 11**: Model Deployment
- [ ] **Module 12**: Advanced Topics & Generative Models

## 📚 Citations & References

If you use this course in your research or work, please cite:

```bibtex
@misc{computer_vision_from_scratch_2025,
  title={Computer Vision from Scratch: Foundations and Practicals},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-username/computer-vision-from-scratch}
}
```

**Key References:**
- LeCun, Y., et al. "Deep learning." Nature 521.7553 (2015): 436-444.
- Krizhevsky, A., et al. "ImageNet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.
- Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).





## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

### 📋 **License Summary**
- ✅ **Commercial use** - Use in commercial projects
- ✅ **Modification** - Modify and adapt the content  
- ✅ **Distribution** - Share and redistribute
- ✅ **Private use** - Use for personal learning
- ⚠️ **Attribution required** - Credit the original course

---

<div align="center">

### 🚀 Ready to Start Your Computer Vision Journey?

**[📂 Browse All Modules](#quick-navigation)** | **[🎯 Choose Learning Path](./learning-paths/)** | **[💻 Set Up Environment](./docs/setup-guide.md)**

---

**⭐ Star this repository if you find it helpful!** 

*Let's build the future of computer vision together.*

---

**Made with ❤️ by the Computer Vision Community**

</div>
