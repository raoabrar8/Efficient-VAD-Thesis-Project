import torch
import torch.nn as nn
import math

class MemoryModule(nn.Module):
    """
    The 'Memory Bank' restricts the model to only using 'Normal' patterns it has learned.
    This forces the model to fail when reconstructing Anomalies, increasing accuracy.
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        
        # The Memory Matrix (Slots x Features)
        self.weight = nn.Parameter(torch.Tensor(mem_dim, fea_dim))  
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly
        stdv = 1. / math.sqrt(self.mem_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input shape: (Batch * Seq, Feature_Dim)
        
        # 1. Calculate Attention (How similar is input to Memory?)
        att_weight = torch.mm(input, self.weight.t())  
        att_weight = torch.softmax(att_weight, dim=1) 
        
        # 2. Read from Memory (Reconstruct using ONLY memory items)
        output = torch.mm(att_weight, self.weight)
        
        return output, att_weight

class EfficientVAD_Pro(nn.Module):
    """
    Efficient-VAD Pro: 
    Combines MobileNetV3 features with an LSTM-Autoencoder + Memory Module.
    """
    def __init__(self, input_dim=960, hidden_dim=256, latent_dim=64, num_memories=50):
        super(EfficientVAD_Pro, self).__init__()
        
        # ==========================
        # ENCODER
        # ==========================
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )
        
        # ==========================
        # MEMORY MODULE
        # ==========================
        self.memory = MemoryModule(mem_dim=num_memories, fea_dim=latent_dim)
        
        # ==========================
        # DECODER
        # ==========================
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 960)
        batch, seq, dim = x.size()
        
        # 1. Encode
        x_enc = self.encoder_fc(x) # (Batch, Seq, 256)
        z, (h, c) = self.encoder_lstm(x_enc) # (Batch, Seq, 64)
        
        # 2. Memory Access
        # Flatten time to treat every moment as a query
        z_flat = z.contiguous().view(-1, 64)
        z_hat_flat, att = self.memory(z_flat)
        
        # Reshape back to sequence
        z_hat = z_hat_flat.view(batch, seq, 64)
        
        # 3. Decode
        dec_out, _ = self.decoder_lstm(z_hat) # (Batch, Seq, 256)
        reconstruction = self.decoder_fc(dec_out) # (Batch, Seq, 960)
        
        return reconstruction, att

if __name__ == "__main__":
    # Sanity Check
    model = EfficientVAD_Pro(input_dim=960) # MobileNet Size
    dummy = torch.randn(32, 10, 960)
    out, att = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {out.shape}") # Should match input
    print(f"Memory Attention: {att.shape}")