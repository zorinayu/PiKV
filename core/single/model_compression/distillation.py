import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any

class BaseCompressor(nn.Module):
    """
    Base compressor class for KV cache compression
    """
    def __init__(self, hidden_size: int):
        super(BaseCompressor, self).__init__()
        self.hidden_size = hidden_size
        
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Base implementation for compressing KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        # Base class doesn't implement compression, just returns originals
        return keys, values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            "hidden_size": self.hidden_size,
        }

class FastVideoCompressor(BaseCompressor):
    """
    FastVideo-inspired compressor for KV cache
    Uses techniques from FastVideo to compress KV cache:
    1. Temporal correlation for sequential tokens
    2. Motion-aware compression for changing context
    3. Adaptive keyframe selection
    """
    def __init__(
        self, 
        hidden_size: int, 
        keyframe_interval: int = 8,
        motion_threshold: float = 0.2,
        compression_ratio: float = 0.5
    ):
        super(FastVideoCompressor, self).__init__(hidden_size)
        self.keyframe_interval = keyframe_interval
        self.motion_threshold = motion_threshold
        self.compression_ratio = compression_ratio
        
        # Motion prediction networks
        self.key_motion_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.value_motion_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Residual compression network
        compressed_size = max(int(hidden_size * compression_ratio), 1)
        self.residual_encoder = nn.Sequential(
            nn.Linear(hidden_size, compressed_size),
            nn.ReLU()
        )
        
        self.residual_decoder = nn.Sequential(
            nn.Linear(compressed_size, hidden_size),
            nn.ReLU()
        )
        
        # Key-frame selector
        self.keyframe_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Cache for previous frames
        self.register_buffer('prev_key_frame', torch.zeros(1, hidden_size))
        self.register_buffer('prev_value_frame', torch.zeros(1, hidden_size))
        
        # Frame counter
        self.register_buffer('frame_count', torch.tensor(0))
        
        # Statistics
        self.register_buffer('keyframe_ratio', torch.tensor(0.0))
        self.register_buffer('motion_magnitude', torch.tensor(0.0))
        self.register_buffer('sample_count', torch.tensor(0.0))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _is_keyframe(self, x: torch.Tensor) -> torch.Tensor:
        """Determine if current frame is a keyframe"""
        # Every keyframe_interval frames is automatically a keyframe
        is_interval_keyframe = (self.frame_count % self.keyframe_interval) == 0
        
        # Also check content-based keyframe selection
        batch_size = x.size(0)
        keyframe_scores = self.keyframe_selector(x.mean(dim=1))  # [batch_size, 1]
        content_keyframe = keyframe_scores > self.motion_threshold
        
        # Combine both criteria
        return torch.logical_or(is_interval_keyframe, content_keyframe).float()
    
    def _compute_motion(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Compute motion between current and previous frames"""
        # Concatenate current and previous frames
        combined = torch.cat([current, previous.expand(current.size(0), -1)], dim=-1)
        return combined
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FastVideo-inspired compression to KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Process each position in the sequence
        compressed_keys = []
        compressed_values = []
        
        # Track keyframe ratio for statistics
        keyframe_count = 0
        
        for pos in range(seq_len):
            # Get current frame
            curr_keys = keys[:, pos]  # [batch_size, hidden_size]
            curr_values = values[:, pos]  # [batch_size, hidden_size]
            
            # Check if this should be a keyframe
            is_keyframe = self._is_keyframe(curr_keys)
            keyframe_count += is_keyframe.sum().item()
            
            if is_keyframe.item() > 0.5:
                # For keyframes, just use the original
                pos_keys = curr_keys
                pos_values = curr_values
                
                # Update previous keyframe
                with torch.no_grad():
                    self.prev_key_frame = curr_keys.mean(dim=0, keepdim=True)
                    self.prev_value_frame = curr_values.mean(dim=0, keepdim=True)
            else:
                # For non-keyframes, predict from previous
                # Compute motion between current and previous
                key_motion_input = self._compute_motion(curr_keys, self.prev_key_frame)
                value_motion_input = self._compute_motion(curr_values, self.prev_value_frame)
                
                # Predict motion
                key_motion = self.key_motion_predictor(key_motion_input)
                value_motion = self.value_motion_predictor(value_motion_input)
                
                # Apply motion to get prediction
                predicted_keys = self.prev_key_frame + key_motion
                predicted_values = self.prev_value_frame + value_motion
                
                # Compute residuals
                key_residual = curr_keys - predicted_keys
                value_residual = curr_values - predicted_values
                
                # Compress residuals
                compressed_key_residual = self.residual_decoder(self.residual_encoder(key_residual))
                compressed_value_residual = self.residual_decoder(self.residual_encoder(value_residual))
                
                # Final reconstruction
                pos_keys = predicted_keys + compressed_key_residual
                pos_values = predicted_values + compressed_value_residual
                
                # Update motion magnitude for statistics
                with torch.no_grad():
                    self.motion_magnitude = (key_motion.abs().mean() + value_motion.abs().mean()) / 2
            
            # Apply importance weighting if provided
            if importance is not None:
                curr_importance = importance[:, pos].unsqueeze(-1)  # [batch_size, 1]
                
                # High importance tokens get more original content
                blend_ratio = torch.sigmoid(curr_importance * 5)  # Scale for sharper transition
                pos_keys = blend_ratio * curr_keys + (1 - blend_ratio) * pos_keys
                pos_values = blend_ratio * curr_values + (1 - blend_ratio) * pos_values
            
            # Add to compressed sequence
            compressed_keys.append(pos_keys.unsqueeze(1))
            compressed_values.append(pos_values.unsqueeze(1))
        
        # Concatenate all positions
        compressed_keys = torch.cat(compressed_keys, dim=1)  # [batch_size, seq_len, hidden_size]
        compressed_values = torch.cat(compressed_values, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # Update frame counter and statistics
        with torch.no_grad():
            self.frame_count += seq_len
            if seq_len > 0:
                self.keyframe_ratio = keyframe_count / (batch_size * seq_len)
            self.sample_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        
        # Estimated compression ratio based on keyframe ratio
        effective_ratio = self.keyframe_ratio * 1.0 + (1 - self.keyframe_ratio) * self.compression_ratio
        
        stats.update({
            "keyframe_interval": self.keyframe_interval,
            "motion_threshold": self.motion_threshold,
            "compression_ratio": effective_ratio.item(),
            "memory_reduction": 1.0 - effective_ratio.item(),
            "keyframe_ratio": self.keyframe_ratio.item(),
            "motion_magnitude": self.motion_magnitude.item(),
            "frame_count": self.frame_count.item(),
            "sample_count": self.sample_count.item()
        })
        
        return stats

class MiniLLMCompressor(BaseCompressor):
    """
    MiniLLM-inspired compressor for KV cache
    Uses knowledge distillation to compress cache with small student model
    """
    def __init__(
        self, 
        hidden_size: int, 
        student_size: int = 64,
        num_layers: int = 2,
        use_attention: bool = True
    ):
        super(MiniLLMCompressor, self).__init__(hidden_size)
        self.student_size = student_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Student encoder (teacher → student)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, student_size)
        )
        
        # Student model layers
        self.student_layers = nn.ModuleList()
        for i in range(num_layers):
            # For first layer, input size is student_size
            # For subsequent layers, both input and output size are student_size
            input_size = student_size
            
            if use_attention and i == num_layers - 1:
                # Last layer uses attention if enabled
                self.student_layers.append(self._make_attention_layer(input_size, student_size))
            else:
                # Regular feedforward layer
                self.student_layers.append(self._make_ff_layer(input_size, student_size))
        
        # Student decoder (student → teacher)
        self.decoder = nn.Sequential(
            nn.Linear(student_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Teacher projections for distillation
        self.teacher_key_proj = nn.Linear(hidden_size, hidden_size)
        self.teacher_value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize weights
        self._init_weights()
        
        # Statistics tracking
        self.register_buffer('distillation_loss', torch.tensor(0.0))
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _make_ff_layer(self, input_size: int, output_size: int) -> nn.Module:
        """Create a feedforward layer with residual connection"""
        return nn.Sequential(
            nn.Linear(input_size, output_size * 4),
            nn.GELU(),
            nn.Linear(output_size * 4, output_size),
            nn.LayerNorm(output_size)
        )
    
    def _make_attention_layer(self, input_size: int, output_size: int) -> nn.Module:
        """Create a self-attention layer"""
        return nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=4,
            batch_first=True
        )
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _apply_student_model(self, x: torch.Tensor) -> torch.Tensor:
        """Apply student model to compressed representation"""
        # Encode to student representation
        student_repr = self.encoder(x)
        
        # Apply student layers
        for i, layer in enumerate(self.student_layers):
            if self.use_attention and i == len(self.student_layers) - 1:
                # For attention layer
                attn_output, _ = layer(
                    student_repr, student_repr, student_repr,
                    need_weights=False
                )
                student_repr = student_repr + attn_output
            else:
                # For feedforward layers
                student_repr = student_repr + layer(student_repr)
        
        # Decode back to teacher representation
        return self.decoder(student_repr)
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MiniLLM-inspired compression to KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Apply teacher projections
        projected_keys = self.teacher_key_proj(keys)
        projected_values = self.teacher_value_proj(values)
        
        # Flatten for processing
        keys_flat = projected_keys.reshape(-1, hidden_size)
        values_flat = projected_values.reshape(-1, hidden_size)
        
        # Process with student model
        processed_keys = self._apply_student_model(keys_flat)
        processed_values = self._apply_student_model(values_flat)
        
        # Reshape back to original dimensions
        compressed_keys = processed_keys.reshape(batch_size, seq_len, hidden_size)
        compressed_values = processed_values.reshape(batch_size, seq_len, hidden_size)
        
        # Apply importance-weighted residual if provided
        if importance is not None:
            if importance.dim() == 2:  # [batch_size, seq_len]
                importance = importance.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Linear scaling of importance for residual weighting
            residual_weight = torch.clamp(importance, 0, 1)
            
            # Add weighted residual from original
            compressed_keys = compressed_keys + residual_weight * (keys - compressed_keys)
            compressed_values = compressed_values + residual_weight * (values - compressed_values)
        
        # Calculate distillation loss for statistics
        with torch.no_grad():
            key_mse = F.mse_loss(compressed_keys, keys)
            value_mse = F.mse_loss(compressed_values, values)
            self.distillation_loss = (key_mse + value_mse) / 2
            self.sample_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        
        # Calculate compression ratio (student size / hidden size)
        compression_ratio = float(self.student_size) / self.hidden_size
        
        stats.update({
            "student_size": self.student_size,
            "num_layers": self.num_layers,
            "use_attention": self.use_attention,
            "compression_ratio": compression_ratio,
            "memory_reduction": 1.0 - compression_ratio,
            "distillation_loss": self.distillation_loss.item(),
            "sample_count": self.sample_count.item()
        })
        
        return stats 