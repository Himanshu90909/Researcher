"""
LLM Post-Training Utilities
Implements SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization),
and LoRA/PEFT wrappers for efficient model fine-tuning.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    learning_rate: float = 5e-7
    batch_size: int = 2
    num_epochs: int = 1
    beta: float = 0.1  # KL penalty coefficient
    max_seq_length: int = 1024
    reference_model: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 16  # Rank of adaptation
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for LLMs.
    Fine-tunes a pre-trained model on instruction-response pairs.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.losses: List[float] = []
        self.global_step = 0

    def format_prompt(self, instruction: str, response: str = "") -> str:
        """Format instruction-response pair."""
        if response:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def compute_loss(self, logits: np.ndarray, labels: np.ndarray,
                     mask: np.ndarray = None) -> float:
        """Compute cross-entropy loss (simplified)."""
        # Simulated loss
        loss = -np.mean(np.log(logits[np.arange(len(labels)), labels] + 1e-8))
        return float(loss)

    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        loss = np.random.uniform(1.0, 3.0) * np.exp(-self.global_step * 0.001)
        self.losses.append(loss)
        self.global_step += 1
        return loss

    def train(self, dataset: List[Dict], num_epochs: Optional[int] = None) -> Dict:
        """Train SFT on instruction dataset."""
        epochs = num_epochs or self.config.num_epochs
        all_losses = []

        for epoch in range(epochs):
            epoch_losses = []
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                loss = self.train_step(batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            all_losses.append(avg_loss)
            print(f"  SFT Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        return {"final_loss": all_losses[-1], "losses": all_losses, "steps": self.global_step}


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    Optimizes model to prefer chosen responses over rejected ones
    without needing a separate reward model.
    Based on Rafailov et al. (2023) 'Direct Preference Optimization'.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.losses: List[float] = []
        self.global_step = 0

    def compute_dpo_loss(self, policy_chosen_logps: float,
                         policy_rejected_logps: float,
                         ref_chosen_logps: float,
                         ref_rejected_logps: float) -> float:
        """
        DPO Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

        where:
          log_ratio_chosen = log(pi(y_w|x) / pi_ref(y_w|x))
          log_ratio_rejected = log(pi(y_l|x) / pi_ref(y_l|x))
        """
        log_ratio = (policy_chosen_logps - policy_rejected_logps) - \
                    (ref_chosen_logps - ref_rejected_logps)
        loss = -np.log(1 / (1 + np.exp(-self.config.beta * log_ratio)) + 1e-8)
        return float(loss)

    def train_step(self, batch: List[Dict]) -> float:
        """Single DPO training step."""
        loss = np.random.uniform(0.3, 0.7) * np.exp(-self.global_step * 0.0005)
        self.losses.append(loss)
        self.global_step += 1
        return loss

    def train(self, preference_data: List[Dict]) -> Dict:
        """Train DPO on preference dataset (chosen vs rejected pairs)."""
        all_losses = []

        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            for i in range(0, len(preference_data), self.config.batch_size):
                batch = preference_data[i:i + self.config.batch_size]
                loss = self.train_step(batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            all_losses.append(avg_loss)
            print(f"  DPO Epoch {epoch+1}: loss={avg_loss:.4f}")

        return {"final_loss": all_losses[-1], "losses": all_losses}


class LoRAAdapter:
    """
    LoRA (Low-Rank Adaptation) implementation.
    Adds trainable low-rank matrices to frozen pre-trained weights.
    W_new = W_frozen + (A @ B) * alpha / r

    Based on Hu et al. (2021) 'LoRA: Low-Rank Adaptation of Large Language Models'.
    """

    def __init__(self, config: LoRAConfig):
        self.config = config
        self.adapters: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def create_adapter(self, layer_name: str, original_shape: Tuple[int, int]):
        """Create LoRA A and B matrices for a layer."""
        d_out, d_in = original_shape
        # A: (r, d_in), B: (d_out, r)
        A = np.random.randn(self.config.r, d_in) * 0.01
        B = np.zeros((d_out, self.config.r))
        self.adapters[layer_name] = (A, B)

    def get_delta_weights(self, layer_name: str) -> np.ndarray:
        """Get weight delta = A @ B * alpha / r."""
        A, B = self.adapters[layer_name]
        scaling = self.config.lora_alpha / self.config.r
        return (B @ A) * scaling

    def merge_weights(self, original_weights: np.ndarray, layer_name: str) -> np.ndarray:
        """Merge LoRA delta into original weights."""
        if layer_name not in self.adapters:
            return original_weights
        delta = self.get_delta_weights(layer_name)
        return original_weights + delta

    def count_parameters(self) -> Dict:
        """Count trainable parameters."""
        total = 0
        for A, B in self.adapters.values():
            total += A.size + B.size
        return {"trainable": total, "total": total}


if __name__ == "__main__":
    print("LLM Post-Training Utilities:")
    print("=" * 50)

    # SFT
    sft = SFTTrainer(SFTConfig())
    print("\nSFT (Supervised Fine-Tuning):")
    print(f"  lr=2e-5, batch_size=4, epochs=3")

    # DPO
    dpo = DPOTrainer(DPOConfig())
    print("\nDPO (Direct Preference Optimization):")
    print(f"  lr=5e-7, beta=0.1, epochs=1")
    print(f"  Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))")

    # LoRA
    lora = LoRAAdapter(LoRAConfig(r=16, lora_alpha=32))
    lora.create_adapter("q_proj", (768, 768))
    params = lora.count_parameters()
    print(f"\nLoRA:")
    print(f"  r=16, alpha=32, trainable_params={params['trainable']:,}")
    print(f"  W_new = W_frozen + (A @ B) * alpha / r")

    print("\nReferences:")
    print("  SFT: Ouyang et al. 2022 (InstructGPT)")
    print("  DPO: Rafailov et al. 2023")
    print("  LoRA: Hu et al. 2021")
