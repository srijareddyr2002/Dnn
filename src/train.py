# ============================================
# CELL 14: Training Loop
# ============================================

# =========================================================
# PART A: LOSS FUNCTIONS
# =========================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    1. Image prediction loss (MSE)
    2. Text prediction loss (CrossEntropy)
    3. Tag prediction loss (BCEWithLogits) - YOUR INNOVATION
    """

    def __init__(self, config):
        super(MultiTaskLoss, self).__init__()

        self.lambda_image = config['training']['lambda_image']
        self.lambda_text = config['training']['lambda_text']
        self.lambda_tag = config['training']['lambda_tag']

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> token
        self.bce_loss = nn.BCELoss()  # For tag prediction

        print(f"✅ MultiTaskLoss initialized")
        print(f"   λ_image: {self.lambda_image}")
        print(f"   λ_text: {self.lambda_text}")
        print(f"   λ_tag: {self.lambda_tag} (AUXILIARY)")

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from model with predictions
            targets: dict with ground truth
                - 'image_features': (B, 2048)
                - 'text_tokens': (B, max_length)
                - 'tag_labels': (B, K, tag_vocab) [optional]

        Returns:
            loss_dict: dict with individual and total losses
        """

        # ========== MAIN TASK 1: IMAGE PREDICTION ==========
        image_loss = self.mse_loss(
            outputs['predicted_image_features'],
            targets['image_features']
        )

        # ========== MAIN TASK 2: TEXT PREDICTION ==========
        # Reshape for CrossEntropyLoss
        pred_text = outputs['predicted_text_logits']  # (B, max_length, vocab_size)
        true_text = targets['text_tokens']  # (B, max_length)

        # Flatten for loss computation
        pred_text_flat = pred_text.view(-1, pred_text.shape[-1])
        true_text_flat = true_text.view(-1)

        text_loss = self.ce_loss(pred_text_flat, true_text_flat)

        # ========== AUXILIARY TASK: TAG PREDICTION (YOUR INNOVATION) ==========
        tag_loss = 0.0
        if 'tag_labels' in targets and targets['tag_labels'] is not None:
            # Tag labels: (B, K, tag_vocab) - multi-label
            tag_preds = outputs['tag_predictions']  # (B, K, tag_vocab)
            tag_labels = targets['tag_labels']  # (B, K, tag_vocab)

            # Flatten for loss computation
            tag_preds_flat = tag_preds.view(-1, tag_preds.shape[-1])
            tag_labels_flat = tag_labels.view(-1, tag_labels.shape[-1])

            tag_loss = self.bce_loss(tag_preds_flat, tag_labels_flat)

        # ========== COMBINED LOSS ==========
        total_loss = (
            self.lambda_image * image_loss +
            self.lambda_text * text_loss +
            self.lambda_tag * tag_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'tag_loss': tag_loss,
        }

        return loss_dict


# =========================================================
# PART B: TRAINING UTILITIES
# =========================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


# =========================================================
# PART C: TRAINING LOOP
# =========================================================

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train one epoch"""
    model.train()

    total_loss = 0.0
    losses_log = {'image': 0.0, 'text': 0.0, 'tag': 0.0}
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_images = batch['input_images'].to(device)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        target_image = batch['target_image'].to(device)

        # Forward pass
        outputs = model(input_images, input_tokens, target_tokens=target_tokens)

        # Prepare targets
        targets = {
            'image_features': target_image,
            'text_tokens': target_tokens,
            'tag_labels': None,  # We'll add real labels in future
        }

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['grad_clip_norm']
        )

        optimizer.step()

        # Logging
        total_loss += loss.item()
        losses_log['image'] += loss_dict['image_loss'].item()
        losses_log['text'] += loss_dict['text_loss'].item()
        losses_log['tag'] += loss_dict['tag_loss'].item() if isinstance(loss_dict['tag_loss'], torch.Tensor) else 0.0
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}"
        })

        # Log every N steps
        if (batch_idx + 1) % config['logging']['log_every_n_steps'] == 0:
            print(f"\n   Batch {batch_idx + 1}/{len(train_loader)}")
            print(f"   Loss: {loss.item():.6f}")
            print(f"   Image Loss: {loss_dict['image_loss'].item():.6f}")
            print(f"   Text Loss: {loss_dict['text_loss'].item():.6f}")
            print(f"   Tag Loss: {loss_dict['tag_loss'].item():.6f}")

    avg_loss = total_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in losses_log.items()}

    return avg_loss, avg_losses


def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for batch in progress_bar:
            # Move batch to device
            input_images = batch['input_images'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_image = batch['target_image'].to(device)

            # Forward pass
            outputs = model(input_images, input_tokens, target_tokens=target_tokens)

            # Prepare targets
            targets = {
                'image_features': target_image,
                'text_tokens': target_tokens,
                'tag_labels': None,
            }

            # Compute loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


# =========================================================
# PART D: INITIALIZE TRAINING
# =========================================================

print("\n" + "="*80)
print("🚀 INITIALIZING TRAINING")
print("="*80)

# Initialize loss function
criterion = MultiTaskLoss(config).to(device)

# Initialize optimizer
if config['training']['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        eps=config['training']['eps'],
        weight_decay=config['training']['weight_decay']
    )
elif config['training']['optimizer'] == 'adamw':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
else:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

print(f"✅ Optimizer: {config['training']['optimizer'].upper()}")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   Weight decay: {config['training']['weight_decay']}")

# Initialize scheduler (FIXED)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['training']['scheduler_factor'],
    patience=config['training']['scheduler_patience'],
    min_lr=config['training']['scheduler_min_lr']
)

print(f"✅ Scheduler: ReduceLROnPlateau")
print(f"   Factor: {config['training']['scheduler_factor']}")
print(f"   Patience: {config['training']['scheduler_patience']}")

# Initialize early stopping
early_stopping = EarlyStopping(
    patience=config['training']['early_stopping_patience'],
    min_delta=1e-4
)

print(f"✅ Early Stopping")
print(f"   Patience: {config['training']['early_stopping_patience']}")

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_losses': {'image': [], 'text': [], 'tag': []},
}

print(f"\n✅ Training setup complete!")
print(f"   Epochs: {config['training']['num_epochs']}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
