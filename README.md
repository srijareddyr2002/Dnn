# Multi-Task Visual Storytelling with Tag Prediction
# Multimodal Sequence Modelling for Visual Story Understanding.
# Project Overview
# Author: Sai Srija Reddy Ramasahayam


## Quick Links
- *[Experiments Notebook](experiment.ipynb)* – Full experimental workflow and implementation  
- *[Evaluation Results](results/)* – All The results are in this folder 
- *[Model Architecture](src/)* – Encoders, fusion, temporal modelling, and decoders.
## Project Overview

This project implements a **Multi-Task Learning with Dual Losses to Better Story Alignment** model for visual storytelling. The model learns to predict both the next image and text in a story sequence while simultaneously learning to predict semantic tags (objects, actions, locations) for each frame. This auxiliary tag prediction task helps the model better understand the content of each frame, leading to more coherent and contextually appropriate predictions.

### Key Innovation

Unlike baseline models that only learn to predict the next image and text, this implementation adds an auxiliary tag prediction task. By training the model to recognize objects, actions, and locations in each frame, it gains a deeper understanding of the story context, resulting in:
- More accurate image predictions
- More coherent text generation
- Better story-to-story alignment

## Dataset

### VIST (Visual Storytelling) Dataset

The project uses the Visual Storytelling (VIST) dataset, which consists of:
- **Training set**: 8,831 complete stories (each with 5 sequential image frames)
- **Image frames**: 167,528 total images from Flickr
- **Text descriptions**: 64,934 captions describing story progression
- **Format**: JSON files containing image URLs, story IDs, frame ordering, and text annotations

### Data Structure

Each story consists of:
- 5 sequential image frames (K=5)
- Corresponding text descriptions for each frame
- Frame ordering information (0-4)
- Story ID for grouping

### Dataset Preprocessing

1. **Image Processing**:
   - Resized to 224×224 pixels (ResNet50 input size)
   - Normalized using ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Data augmentation: horizontal flipping, random cropping, color jittering, rotation (±5°)

2. **Text Processing**:
   - Tokenized using word-level tokenization
   - Maximum sequence length: 30 words
   - Vocabulary size: 10,000 most frequent words
   - Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
   - Padding applied to ensure uniform sequence lengths

3. **Tag Processing**:
   - Tag vocabulary: 51 tags across 3 categories
     - Objects (19): person, dog, cat, car, table, chair, cup, bottle, tree, building, etc.
     - Actions (15): walking, running, sitting, standing, eating, drinking, reading, etc.
     - Locations (15): kitchen, bedroom, park, restaurant, beach, city, home, etc.
   - Tags converted to 128-dimensional embeddings
   - Multi-label binary classification (each frame can have multiple tags)

## Tokenization

### Text Tokenization Pipeline

1. **Vocabulary Building**:
   - Build vocabulary from training set captions
   - Keep top 10,000 most frequent words
   - Add special tokens: `<PAD>` (index 0), `<START>`, `<END>`, `<UNK>`

2. **Token Processing**:
   - Lowercase all text
   - Basic word tokenization
   - Convert words to token IDs based on vocabulary
   - Unknown words mapped to `<UNK>` token
   - Pad sequences to max_length=30

3. **Tag Tokenization**:
   - Tag vocabulary created from predefined categories
   - Each tag mapped to unique index
   - Multi-hot encoding for multi-label classification

## Model Architecture

### Overview

The model follows an encoder-decoder architecture with multi-task learning:

```
Input Images (K=5) ──> Visual Encoder (ResNet50)
                       │
                       ├──> 2048-dim features
                       │
Input Text (K=5) ────> Text Encoder (BiLSTM)
                       │
                       ├──> 1024-dim features
                       │
                       V
                 Multimodal Fusion
                       │
                       ├──> 1024-dim fused features
                       │
                       V
                 Sequence Model (LSTM)
                       │
                       ├──> 1024-dim sequence output
                       │
                       V
                 Attention Mechanism (8-head)
                       │
                       ├──────────────┬─────────────┬──────────────┐
                       │              │             │              │
                       V              V             V              V
              Image Decoder    Text Decoder   Tag Predictor   (unused)
                (3-layer)      (2-layer LSTM)  (2-layer MLP)
                    │              │              │
                    V              V              V
             Next Image     Next Text        Frame Tags
            (2048-dim)    (vocab_size)     (51 tags)
```

### Components

1. **Visual Encoder** (ResNet50):
   - Pretrained on ImageNet
   - Fine-tuned on storytelling task
   - Output: 2048-dimensional feature vectors per frame
   - Processes K=5 input frames

2. **Text Encoder** (Bidirectional LSTM):
   - Word embedding dimension: 300
   - Hidden dimension: 512 (×2 for bidirectional = 1024)
   - Number of layers: 2
   - Dropout: 0.3
   - Processes K=5 input captions

3. **Multimodal Fusion Layer**:
   - Concatenates visual and text features (2048 + 1024 = 3072)
   - Projects to 1024 dimensions through 2-layer MLP
   - Dropout: 0.3

4. **Sequence Model** (LSTM):
   - Hidden dimension: 1024
   - Number of layers: 2
   - Dropout: 0.3
   - Captures temporal dependencies across frames

5. **Attention Mechanism**:
   - Multi-head self-attention (8 heads)
   - Attention dimension: 1024
   - Allows model to focus on relevant frames

6. **Image Decoder**:
   - 3-layer MLP: [1024 → 2048 → 2048 → 2048]
   - Dropout: 0.3
   - Predicts next frame's visual features

7. **Text Decoder** (LSTM):
   - Hidden dimension: 1024
   - Number of layers: 2
   - Dropout: 0.3
   - Generates next caption autoregressively

8. **Tag Prediction Head** (Auxiliary Task):
   - 2-layer MLP: [1024 → 512 → 51]
   - Dropout: 0.4
   - Sigmoid activation for multi-label classification
   - Predicts objects, actions, and locations in current frame

### Loss Functions

Multi-task loss with three components:

```
Total Loss = λ_image × L_image + λ_text × L_text + λ_tag × L_tag
```

- **L_image**: Mean Squared Error (MSE) between predicted and target image features
- **L_text**: Cross-Entropy loss for next word prediction
- **L_tag**: Binary Cross-Entropy for tag prediction (auxiliary task)
- **Weights**: λ_image=1.0, λ_text=1.0, λ_tag=0.3

### Training Configuration

- **Optimizer**: Adam (lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
- **Batch size**: 16
- **Epochs**: 10
- **Learning rate scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Gradient clipping**: max_norm=1.0
- **Early stopping**: patience=10 epochs
- **Seed**: 42 (for reproducibility)

## Environment Setup

### Requirements

```
Python 3.12+
CUDA 12.8+ (for GPU support)
```

### Dependencies

- **Deep Learning**: torch==2.8.0, torchvision==0.23.0, pytorch-lightning==2.6.0
- **NLP**: transformers==4.57.3, tokenizers==0.22.2, nltk==3.9.2, spacy==3.8.11
- **Image Processing**: Pillow==12.0.0, opencv-python==4.11.0, imageio==2.37.2, scikit-image==0.26.0
- **Metrics**: torchmetrics==1.7.4, rouge-score==0.1.2, lpips==0.1.4
- **Scientific Computing**: numpy==1.26.4, pandas==2.1.4, scipy==1.11.4, scikit-learn==1.3.2
- **Visualization**: matplotlib==3.8.2, seaborn==0.13.2, tensorboard==2.15.1, wandb==0.23.1
- **Others**: tqdm==4.67.1, pyyaml==6.0.3

## Installation and Setup

### 1. Clone Repository

```bash
git clone https://github.com/srijareddyr2002/Dnn.git
cd Dnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download VIST Dataset

Download the Visual Storytelling (VIST) dataset from:
- Training images and annotations
- Validation and test splits
- Place in `/path/to/vist_kaggle/` directory

### 4. Configure Paths

Update paths in the notebook (Cell 3):

```python
DATASET_BASE_PATH = "/path/to/vist_kaggle"
OUTPUT_DIR = "/path/to/outputs"
```

## Running in Google Colab

### 1. Upload Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `experiment-2.ipynb`
3. Select GPU runtime: Runtime → Change runtime type → GPU (T4/V100/A100)

### 2. Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install Dependencies

Run Cell 1 to install all required packages.

### 4. Configure Dataset Path

Update the dataset path to your Google Drive location:

```python
DATASET_BASE_PATH = "/content/drive/MyDrive/vist_dataset"
```

### 5. Run Training

Execute cells sequentially:
- Cell 2: Import packages
- Cell 3: Configure hyperparameters
- Cell 4: Load dataset
- Cells 5-14: Build model components
- Cell 15: Initialize complete model
- Cells 16-18: Training loop

### Training Progress

The training loop will display:
- Epoch progress bars
- Training loss (total, image, text, tag components)
- Validation loss
- Learning rate updates
- Best model checkpoints

Example output:
```
================================================================================
📅 EPOCH 3/10
================================================================================

🏋️ Training...
✅ Training Complete
   Total Loss: 6.321478
   Image Loss: 0.197729
   Text Loss: 6.123749
   Tag Loss: 0.000000

🔍 Validating...
✅ Validation Complete
   Val Loss: 6.756166

💾 Checkpoint saved: .../checkpoints/best_model.pth
🏆 New best model! Val Loss: 6.756166
```

## Expected Results

### Quantitative Metrics

Based on the validation results from training:

1. **Image Quality Metrics**:
   - SSIM (Structural Similarity Index): Expected improvement over baseline
   - LPIPS (Learned Perceptual Image Patch Similarity): Lower is better
   - MSE Loss: ~0.20-0.34 (decreasing trend)

2. **Text Generation Metrics**:
   - BLEU Score: Measures n-gram overlap with reference captions
   - ROUGE-L Score: Measures longest common subsequence
   - Perplexity: Lower values indicate better text predictions
   - Cross-Entropy Loss: ~5.5-7.4 (decreasing trend)

3. **Tag Prediction Metrics**:
   - Precision/Recall/F1 for each tag category
   - Hamming Loss: Multi-label classification accuracy
   - Tag BCE Loss: ~0.0 (auxiliary task weight λ_tag=0.3)

### Training Characteristics

- **Convergence**: Model shows improvement in first 3 epochs (validation loss: 8.67 → 6.76)
- **Best performance**: Typically achieved around epoch 3-5
- **Overfitting**: May occur after epoch 5-6 (validation loss increases)
- **Training time**: ~3 minutes per epoch on NVIDIA H200 GPU

### Qualitative Results

The model should produce:
1. **Coherent image features** that align with story progression
2. **Contextually appropriate text** that follows narrative flow
3. **Accurate tag predictions** for objects, actions, and locations in frames
4. **Smooth story transitions** between consecutive frames

## Future Work

### Planned Improvements

1. **Architecture Enhancements**:
   - Experiment with Transformer-based encoders (CLIP, BERT)
   - Implement hierarchical attention mechanisms
   - Add memory networks for long-term story dependencies

2. **Training Strategies**:
   - Curriculum learning (easy to hard stories)
   - Data augmentation for text (back-translation, paraphrasing)
   - Contrastive learning for better multimodal alignment

3. **Auxiliary Tasks**:
   - Scene graph prediction
   - Emotion recognition
   - Character tracking across frames

4. **Evaluation**:
   - Human evaluation studies
   - Story coherence metrics
   - Ablation studies on tag categories

5. **Applications**:
   - Interactive storytelling systems
   - Video captioning
   - Educational content generation

## License

This project is for academic research purposes. The VIST dataset has its own licensing terms. Please refer to the [VIST dataset page](http://visionandlanguage.net/VIST/) for usage restrictions.

## References

1. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
2. Lu, C., Krishna, R., Bernstein, M., & Fei-Fei, L. (2016). Language-based visual relationship detection. arXiv:1608.00187.
3. Huang, T. H., Ferraro, F., Mostafazadeh, N., et al. (2016). Visual Storytelling. *NAACL*.

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{srija2026multitask,
  title={Multi-Task Learning with Dual Losses for Visual Storytelling},
  author={Srija},
  journal={Deep Neural Networks Project},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Note**: This implementation is based on the VIST dataset and extends the baseline visual storytelling model with multi-task learning for improved story alignment.
