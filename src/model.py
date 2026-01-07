# ============================================
# CELL 10: Sequence Model (LSTM)
# ============================================

class SequenceModel(nn.Module):
    """
    LSTM-based sequence model for temporal modeling.
    Processes the K fused features to capture story progression.
    """

    def __init__(self, config):
        super(SequenceModel, self).__init__()

        self.input_dim = config["model"]["fusion_dim"]
        self.hidden_dim = config["model"]["sequence_hidden_dim"]
        self.num_layers = config["model"]["sequence_num_layers"]
        self.dropout = config["model"]["sequence_dropout"]

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        print(f"✅ Sequence Model initialized")
        print(f"   Input dim: {self.input_dim}")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   Num layers: {self.num_layers}")
        print(f"   Dropout: {self.dropout}")

    def forward(self, fused_features):
        """
        Args:
            fused_features: (batch_size, K, fusion_dim)

        Returns:
            sequence_output: (batch_size, K, hidden_dim)
            final_hidden: (num_layers, batch_size, hidden_dim)
            final_cell: (num_layers, batch_size, hidden_dim)
        """
        # LSTM forward pass
        # sequence_output: (batch_size, K, hidden_dim)
        # (hidden, cell): each (num_layers, batch_size, hidden_dim)
        sequence_output, (final_hidden, final_cell) = self.lstm(fused_features)

        return sequence_output, final_hidden, final_cell


# =========================================================
# TEST THE SEQUENCE MODEL
# =========================================================

print("\n" + "="*80)
print("🧪 Testing Sequence Model")
print("="*80)

# Initialize sequence model
sequence_model = SequenceModel(config).to(device)

print(f"\n📦 Input features info:")
print(f"   Fused features shape: {fused_features.shape}")

# Forward pass
print(f"\n🔄 Running forward pass...")
with torch.no_grad():
    sequence_output, final_hidden, final_cell = sequence_model(fused_features)

print(f"\n✅ Forward pass successful!")
print(f"   Sequence output shape: {sequence_output.shape}")
print(f"   Expected: (batch_size=16, K=5, hidden_dim=1024)")
print(f"   Final hidden shape: {final_hidden.shape}")
print(f"   Expected: (num_layers=2, batch_size=16, hidden_dim=1024)")
print(f"   Final cell shape: {final_cell.shape}")

# Check output statistics
print(f"\n📊 Sequence output statistics:")
print(f"   Mean: {sequence_output.mean().item():.4f}")
print(f"   Std: {sequence_output.std().item():.4f}")
print(f"   Min: {sequence_output.min().item():.4f}")
print(f"   Max: {sequence_output.max().item():.4f}")

# Check for NaN or Inf
has_nan = torch.isnan(sequence_output).any()
has_inf = torch.isinf(sequence_output).any()
print(f"   Contains NaN: {has_nan}")
print(f"   Contains Inf: {has_inf}")

# Check temporal progression (output should evolve across time)
print(f"\n📈 Temporal progression (L2 distance between consecutive time steps):")
for i in range(config["data"]["num_frames"] - 1):
    dist = torch.norm(sequence_output[0, i] - sequence_output[0, i+1]).item()
    print(f"   Time {i} → Time {i+1}: {dist:.4f}")

# Compare first and last time step
first_last_dist = torch.norm(sequence_output[0, 0] - sequence_output[0, -1]).item()
print(f"\n   First → Last time step distance: {first_last_dist:.4f}")
print(f"   (Should be larger, indicating temporal evolution)")

# Visualize final hidden state
print(f"\n🔍 Final hidden state (last layer, first batch, first 10 dims):")
print(f"   {final_hidden[-1, 0, :10]}")

# Count parameters
total_params = sum(p.numel() for p in sequence_model.parameters())
trainable_params = sum(p.numel() for p in sequence_model.parameters() if p.requires_grad)
print(f"\n📊 Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

print("\n✅ Sequence Model validation complete!")
print("   All checks passed!" if not (has_nan or has_inf) else "   ⚠️ Warning: NaN/Inf detected!")

# ============================================
# CELL 13: Complete Model Architecture (CORRECTED)
# ============================================

class SrijaMultiTaskModel(nn.Module):
    """
    Complete multi-task learning model for visual storytelling.

    Main tasks:
    1. Image prediction (next frame visual features)
    2. Text prediction (next frame caption)

    Auxiliary task:
    3. Tag prediction (semantic tags for each frame) - YOUR INNOVATION
    """

    def __init__(self, config, vocab_size):
        super(SrijaMultiTaskModel, self).__init__()

        # Component 1: Visual Encoder
        self.visual_encoder = VisualEncoder(config)

        # Component 2: Text Encoder
        self.text_encoder = TextEncoder(config, vocab_size)

        # Component 3: Multimodal Fusion
        self.fusion = MultimodalFusion(config)

        # Component 4: Tag Prediction Head (INNOVATION)
        self.tag_predictor = TagPredictionHead(config)

        # Component 5: Sequence Model
        self.sequence_model = SequenceModel(config)

        # Component 6: Attention Mechanism
        self.attention = AttentionMechanism(config)

        # Component 7: Dual Decoders
        self.image_decoder = ImageDecoder(config)
        self.text_decoder = TextDecoder(config, vocab_size)

        print(f"\n✅ SrijaMultiTaskModel initialized!")
        print(f"   All 7 components integrated")

    def forward(self, input_images, input_tokens, target_tokens=None):
        """
        Complete forward pass through entire architecture.

        Args:
            input_images: (batch_size, K, 3, H, W)
            input_tokens: (batch_size, K, max_length)
            target_tokens: (batch_size, max_length) - for teacher forcing

        Returns:
            outputs: dict with predictions and auxiliary outputs
        """
        batch_size = input_images.shape[0]

        # ============ ENCODING PHASE ============

        # 1. Visual Encoding
        visual_features = self.visual_encoder(input_images)  # (B, K, 2048)

        # 2. Text Encoding
        text_features = self.text_encoder(input_tokens)  # (B, K, 1024)

        # 3. Multimodal Fusion
        fused_features = self.fusion(visual_features, text_features)  # (B, K, 1024)

        # ============ AUXILIARY TASK: TAG PREDICTION ============

        # 4. Tag Prediction (on all frames)
        tag_predictions = self.tag_predictor(fused_features)  # (B, K, tag_vocab)

        # ============ SEQUENCE MODELING PHASE ============

        # 5. Sequence Model (temporal modeling)
        sequence_output, final_hidden, final_cell = self.sequence_model(fused_features)
        # sequence_output: (B, K, 1024)

        # 6. Attention Mechanism
        attended_output, attention_weights = self.attention(sequence_output)
        # attended_output: (B, K, 1024)
        # attention_weights: (B, K, K)

        # ============ DECODING PHASE ============

        # Get context from final frame's attended output
        context = attended_output[:, -1, :]  # (B, 1024)

        # 7a. Image Decoding
        predicted_image_features = self.image_decoder(context)  # (B, 2048)

        # 7b. Text Decoding
        predicted_text_logits = self.text_decoder(context, target_tokens=target_tokens)
        # (B, max_length, vocab_size)

        # ============ RETURN OUTPUTS ============

        outputs = {
            # Main task outputs
            'predicted_image_features': predicted_image_features,
            'predicted_text_logits': predicted_text_logits,

            # Auxiliary task output (YOUR INNOVATION)
            'tag_predictions': tag_predictions,

            # Intermediate representations (for analysis/visualization)
            'visual_features': visual_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'sequence_output': sequence_output,
            'attended_output': attended_output,
            'attention_weights': attention_weights,
            'context': context,
        }

        return outputs


# =========================================================
# TEST THE COMPLETE MODEL
# =========================================================

print("\n" + "="*80)
print("🧪 Testing Complete SrijaMultiTaskModel")
print("="*80)

# Initialize complete model
model = SrijaMultiTaskModel(config, vocab_size=len(tokenizer.word2idx)).to(device)

print(f"\n📦 Input info:")
print(f"   Input images shape: {real_images.shape}")
print(f"   Input tokens shape: {real_text_tokens.shape}")
print(f"   Target tokens shape: {target_tokens_real.shape}")

# Forward pass
print(f"\n🔄 Running complete forward pass...")
with torch.no_grad():
    outputs = model(real_images, real_text_tokens, target_tokens=target_tokens_real)

print(f"\n✅ Forward pass successful!")

# Check all outputs
print(f"\n📊 Output shapes:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"   {key}: {value.shape}")
    else:
        print(f"   {key}: {type(value).__name__}")

# Validate main outputs
print(f"\n🎯 Main Task Outputs:")
print(f"   Predicted image features:")
print(f"      Shape: {outputs['predicted_image_features'].shape}")
print(f"      Expected: (16, 2048)")
print(f"      Mean: {outputs['predicted_image_features'].mean().item():.4f}")
print(f"      Std: {outputs['predicted_image_features'].std().item():.4f}")

print(f"   Predicted text logits:")
print(f"      Shape: {outputs['predicted_text_logits'].shape}")
print(f"      Expected: (16, 30, 10000)")

# Validate auxiliary output
print(f"\n🏷️  Auxiliary Task Output (YOUR INNOVATION):")
print(f"   Tag predictions:")
print(f"      Shape: {outputs['tag_predictions'].shape}")
print(f"      Expected: (16, 5, 51)")
print(f"      Mean: {outputs['tag_predictions'].mean().item():.4f}")
print(f"      Std: {outputs['tag_predictions'].std().item():.4f}")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Model Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Breakdown by component
components = {
    'Visual Encoder': model.visual_encoder,
    'Text Encoder': model.text_encoder,
    'Fusion': model.fusion,
    'Tag Predictor': model.tag_predictor,
    'Sequence Model': model.sequence_model,
    'Attention': model.attention,
    'Image Decoder': model.image_decoder,
    'Text Decoder': model.text_decoder,
}

print(f"\n🔧 Parameters by component:")
for name, component in components.items():
    comp_params = sum(p.numel() for p in component.parameters())
    print(f"   {name}: {comp_params:,}")

# Check for NaN/Inf in all outputs
has_nan = any(torch.isnan(v).any() for v in outputs.values() if isinstance(v, torch.Tensor))
has_inf = any(torch.isinf(v).any() for v in outputs.values() if isinstance(v, torch.Tensor))

print(f"\n✅ Output validation:")
print(f"   Contains NaN: {has_nan}")
print(f"   Contains Inf: {has_inf}")

# ============================================
# FIX: Testing backward pass (CORRECTED)
# ============================================

print(f"\n🔄 Testing backward pass...")
# ============================================
# Testing backward pass (CORRECTED)
# ============================================

print(f"\n🔄 Testing backward pass...")
model.train()

# Re-run forward with gradient tracking
outputs = model(real_images, real_text_tokens, target_tokens=target_tokens_real)

# Get actual batch size from outputs
batch_size = outputs['predicted_image_features'].shape[0]

# Create dummy target with CORRECT batch size
image_features_target = torch.randn(batch_size, 2048).to(device)

# MSE Loss for image prediction
loss = torch.nn.functional.mse_loss(outputs['predicted_image_features'], image_features_target)

print(f"   Loss value: {loss.item():.6f}")
print(f"   Loss requires grad: {loss.requires_grad}")

# Backward
loss.backward()

# Check if gradients are computed
has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"✅ Gradients computed: {has_gradients}")

if has_gradients:
    all_grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    print(f"   Gradient mean: {all_grads.mean().item():.6f}")
    print(f"   Gradient std: {all_grads.std().item():.6f}")
    print(f"   Gradient max: {all_grads.max().item():.6f}")
    print(f"   Gradient min: {all_grads.min().item():.6f}")

print("\n" + "="*80)
print("🎉 COMPLETE MODEL VALIDATED!")
print("="*80)
