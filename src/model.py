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
