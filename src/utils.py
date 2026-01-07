# ============================================
# CELL 5 (COMPLETE): Custom Dataset Class + Subset
# ============================================

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests


# =========================================================
# PART A: SIMPLE TOKENIZER
# =========================================================

class SimpleTokenizer:
    """Basic word-level tokenizer with vocabulary"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = {}

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        print("🔨 Building vocabulary...")

        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        for idx, (word, freq) in enumerate(sorted_words[:self.vocab_size-4], 4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"✅ Vocabulary built: {len(self.word2idx)} tokens")
        return self

    def encode(self, text, max_length=30, pad=True):
        """Convert text to token indices"""
        words = text.lower().split()[:max_length]
        tokens = [self.word2idx.get(w, 1) for w in words]

        if pad:
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]

        return tokens

    def decode(self, tokens):
        """Convert token indices back to text"""
        words = [self.idx2word.get(int(t), "<UNK>") for t in tokens if t > 0]
        return " ".join(words)


# =========================================================
# PART B: IMAGE LOADING UTILITIES
# =========================================================

def load_image_from_url(url, image_size=224, timeout=5):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        return None


def create_dummy_image(image_size=224):
    """Create dummy image if loading fails"""
    return Image.new("RGB", (image_size, image_size), color=(128, 128, 128))


# =========================================================
# PART C: CUSTOM DATASET CLASS
# =========================================================

class StorySequenceDataset(Dataset):
    """
    Dataset for visual storytelling.

    Strategy: For each story with K frames,
    - Input: frames[0:K] (all K frames)
    - Target: frames[K-1] (last frame as target, which is frame K)

    This works because VIST stories have exactly K=5 frames each.
    """

    def __init__(
        self,
        stories_dict,
        tokenizer,
        image_size=224,
        max_text_length=30,
        num_frames=5,
        augment=True
    ):
        self.stories = stories_dict
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.num_frames = num_frames

        # Build list of story_ids with correct number of frames
        self.story_ids = [
            sid for sid, frames in stories_dict.items()
            if len(frames) >= num_frames
        ]

        print(f"   Found {len(self.story_ids)} stories with ≥{num_frames} frames")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Data augmentation (optional)
        self.augment = augment
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(5),
            ])

    def __len__(self):
        return len(self.story_ids)

    def __getitem__(self, idx):
        """Get one training sample"""
        story_id = self.story_ids[idx]
        frames = self.stories[story_id]

        # Use first K frames as input
        input_frames = frames[:self.num_frames]
        # Use the LAST of the K frames as target
        target_frame = frames[self.num_frames - 1]

        # Load and process images
        input_images = []
        for frame in input_frames:
            img = load_image_from_url(frame["url_o"], self.image_size)
            if img is None:
                img = create_dummy_image(self.image_size)

            # Apply augmentation to input images
            if self.augment:
                img = self.augment_transform(img)

            img = self.transform(img)
            input_images.append(img)

        # Load target image
        target_img = load_image_from_url(target_frame["url_o"], self.image_size)
        if target_img is None:
            target_img = create_dummy_image(self.image_size)

        target_img = self.transform(target_img)

        # Stack images: shape (K, 3, H, W)
        input_images = torch.stack(input_images, dim=0)

        # Tokenize texts
        input_texts = [frame["text"] for frame in input_frames]
        target_text = target_frame["text"]

        input_tokens = [self.tokenizer.encode(text, self.max_text_length) for text in input_texts]
        input_tokens = torch.tensor(input_tokens, dtype=torch.long)

        target_tokens = torch.tensor(
            self.tokenizer.encode(target_text, self.max_text_length),
            dtype=torch.long
        )

        return {
            "story_id": story_id,
            "input_images": input_images,
            "input_texts": input_texts,
            "input_tokens": input_tokens,
            "target_image": target_img,
            "target_text": target_text,
            "target_tokens": target_tokens
        }


# =========================================================
# PART D: BUILD VOCABULARY & TOKENIZER
# =========================================================

all_texts = []
for story_id, frames in STORIES.items():
    for frame in frames:
        all_texts.append(frame["text"])

print(f"\n📝 Building tokenizer from {len(all_texts)} texts...")
tokenizer = SimpleTokenizer(vocab_size=config["data"]["vocab_size"])
tokenizer.build_vocab(all_texts)

import pickle
tokenizer_path = f"{OUTPUT_DIR}/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"💾 Tokenizer saved to: {tokenizer_path}")


# =========================================================
# PART E: CREATE DATASETS & DATALOADERS
# =========================================================

# Split stories into train/val/test
story_ids = list(STORIES.keys())
n_stories = len(story_ids)
n_train = int(n_stories * config["data"]["train_split"])
n_val = int(n_stories * config["data"]["val_split"])

set_seed(SEED)
random.shuffle(story_ids)

train_story_ids = story_ids[:n_train]
val_story_ids = story_ids[n_train:n_train + n_val]
test_story_ids = story_ids[n_train + n_val:]

train_stories = {sid: STORIES[sid] for sid in train_story_ids}
val_stories = {sid: STORIES[sid] for sid in val_story_ids}
test_stories = {sid: STORIES[sid] for sid in test_story_ids}

print(f"\n📊 Data split (BEFORE subset):")
print(f"   Train stories: {len(train_stories)}")
print(f"   Val stories: {len(val_stories)}")
print(f"   Test stories: {len(test_stories)}")

# Create full datasets
print(f"\n🔧 Creating full datasets...")
train_dataset_full = StorySequenceDataset(
    train_stories,
    tokenizer,
    image_size=config["data"]["image_size"],
    max_text_length=config["data"]["max_text_length"],
    num_frames=config["data"]["num_frames"],
    augment=True
)

val_dataset_full = StorySequenceDataset(
    val_stories,
    tokenizer,
    image_size=config["data"]["image_size"],
    max_text_length=config["data"]["max_text_length"],
    num_frames=config["data"]["num_frames"],
    augment=False
)

test_dataset_full = StorySequenceDataset(
    test_stories,
    tokenizer,
    image_size=config["data"]["image_size"],
    max_text_length=config["data"]["max_text_length"],
    num_frames=config["data"]["num_frames"],
    augment=False
)

# ========================================
# ⚠️  SUBSET FOR TESTING (MODIFY HERE)
# ========================================

SUBSET_SIZE = 200  # ← CHANGE THIS VALUE
USE_SUBSET = True  # Set to False to use full dataset

if USE_SUBSET:
    print(f"\n⚠️  USING SUBSET FOR TESTING")
    print(f"   Original train: {len(train_dataset_full)}")
    print(f"   Subset size: {SUBSET_SIZE}")

    # Create subsets
    train_indices = list(range(min(SUBSET_SIZE, len(train_dataset_full))))
    val_indices = list(range(min(SUBSET_SIZE // 5, len(val_dataset_full))))
    test_indices = list(range(min(SUBSET_SIZE // 5, len(test_dataset_full))))

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    print(f"   New train: {len(train_dataset)}")
    print(f"   New val: {len(val_dataset)}")
    print(f"   New test: {len(test_dataset)}")
else:
    print(f"\n✅ USING FULL DATASET")
    train_dataset = train_dataset_full
    val_dataset = val_dataset_full
    test_dataset = test_dataset_full

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"\n✅ DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")
print(f"   Batch size: {config['training']['batch_size']}")

# Estimate training time
est_time_per_epoch = len(train_loader) * 5 / 60
est_total_time = est_time_per_epoch * config['training']['num_epochs']
print(f"\n⏱️  Estimated training time:")
print(f"   Per epoch: ~{est_time_per_epoch:.1f} minutes")
print(f"   Total ({config['training']['num_epochs']} epochs): ~{est_total_time:.1f} minutes")

# =========================================================
# PART F: TEST ONE BATCH
# =========================================================

print("\n" + "="*80)
print("🧪 Testing one batch from train_loader")
print("="*80)

try:
    sample_batch = next(iter(train_loader))
    print(f"\n✅ Successfully loaded batch!")
    print(f"📦 Batch keys: {list(sample_batch.keys())}")
    print(f"   input_images shape: {sample_batch['input_images'].shape}")
    print(f"   input_tokens shape: {sample_batch['input_tokens'].shape}")
    print(f"   target_image shape: {sample_batch['target_image'].shape}")
    print(f"   target_tokens shape: {sample_batch['target_tokens'].shape}")

    print(f"\n📝 Sample input texts (first sequence):")
    for i in range(config["data"]["num_frames"]):
        text = sample_batch['input_texts'][i]
        print(f"   Frame {i+1}: {text[:70]}...")

    print(f"\n🎯 Sample target text:")
    print(f"   {sample_batch['target_text'][0][:70]}...")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Dataset loading successful!")
print("="*80)
