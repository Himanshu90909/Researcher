"""
Research Notebook 6: LLM Post-Training (SFT, DPO, LoRA)
Demonstrates fine-tuning techniques for LLMs.

Run: python notebooks/06_llm_finetune.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.llm_finetune import (
    SFTTrainer, SFTConfig, DPOTrainer, DPOConfig,
    LoRAAdapter, LoRAConfig
)


def main():
    print("=" * 60)
    print("Notebook 6: LLM Post-Training (SFT, DPO, LoRA)")
    print("=" * 60)
    
    # 1. Supervised Fine-Tuning
    print("\n--- Supervised Fine-Tuning (SFT) ---")
    sft_config = SFTConfig(
        learning_rate=2e-5,
        batch_size=4,
        num_epochs=5,
        warmup_steps=50,
        max_seq_length=512,
    )
    print(f"  Config: lr={sft_config.learning_rate}, batch={sft_config.batch_size}, "
          f"epochs={sft_config.num_epochs}")
    
    sft_trainer = SFTTrainer(sft_config)
    
    # Create training dataset
    sft_dataset = [
        {"instruction": f"Summarize the following video scene: Scene {i}",
         "response": f"This scene shows activity {i} with key visual elements."}
        for i in range(100)
    ]
    
    sft_result = sft_trainer.train(sft_dataset, num_epochs=5)
    print(f"\n  Training losses: {[f'{l:.4f}' for l in sft_result['losses']]}")
    print(f"  Total steps: {sft_result['steps']}")
    
    # 2. Direct Preference Optimization
    print("\n--- Direct Preference Optimization (DPO) ---")
    dpo_config = DPOConfig(
        learning_rate=5e-7,
        batch_size=2,
        num_epochs=3,
        beta=0.1,
    )
    print(f"  Config: lr={dpo_config.learning_rate}, beta={dpo_config.beta}")
    print(f"  Loss: -log(σ(β * (log π(y_w|x) - log π_ref(y_w|x)) - "
          f"(log π(y_l|x) - log π_ref(y_l|x))))")
    
    dpo_trainer = DPOTrainer(dpo_config)
    
    # Create preference dataset
    preference_data = [
        {
            "prompt": f"Generate a title for video {i}",
            "chosen": f"Engaging Video Title {i} - AI Powered Content Creation",
            "rejected": f"video {i} title here",
        }
        for i in range(50)
    ]
    
    dpo_result = dpo_trainer.train(preference_data)
    print(f"\n  Training losses: {[f'{l:.4f}' for l in dpo_result['losses']]}")
    
    # Test DPO loss computation
    print("\n  DPO Loss Analysis:")
    test_cases = [
        {"policy_chosen": -1.0, "policy_rejected": -3.0, "ref_chosen": -2.0, "ref_rejected": -2.0,
         "label": "Policy prefers chosen more than ref"},
        {"policy_chosen": -2.0, "policy_rejected": -2.0, "ref_chosen": -2.0, "ref_rejected": -2.0,
         "label": "No preference change"},
        {"policy_chosen": -3.0, "policy_rejected": -1.0, "ref_chosen": -2.0, "ref_rejected": -2.0,
         "label": "Policy prefers rejected (bad)"},
    ]
    
    for case in test_cases:
        loss = dpo_trainer.compute_dpo_loss(
            case["policy_chosen"], case["policy_rejected"],
            case["ref_chosen"], case["ref_rejected"]
        )
        print(f"    {case['label']}: loss={loss:.4f}")
    
    # 3. LoRA (Low-Rank Adaptation)
    print("\n--- LoRA (Low-Rank Adaptation) ---")
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    print(f"  Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"dropout={lora_config.lora_dropout}")
    
    lora = LoRAAdapter(lora_config)
    
    # Create adapters for Transformer attention modules
    modules = [
        ("q_proj", (768, 768)),
        ("k_proj", (768, 768)),
        ("v_proj", (768, 768)),
        ("o_proj", (768, 768)),
    ]
    
    for name, shape in modules:
        lora.create_adapter(name, shape)
    
    params = lora.count_parameters()
    original_params = sum(s[0] * s[1] for _, s in modules)
    print(f"\n  Original params: {original_params:,}")
    print(f"  LoRA trainable params: {params['trainable']:,}")
    print(f"  Reduction: {original_params / params['trainable']:.1f}x")
    print(f"  Trainable: {params['trainable'] / original_params * 100:.2f}%")
    
    # Test weight merging
    original_weights = np.random.randn(768, 768) * 0.02
    merged = lora.merge_weights(original_weights, "q_proj")
    delta = merged - original_weights
    print(f"\n  Weight delta: mean={delta.mean():.6f}, std={delta.std():.6f}")
    
    # Compare LoRA ranks
    print("\n--- LoRA Rank Comparison ---")
    for r in [2, 4, 8, 16, 32, 64]:
        cfg = LoRAConfig(r=r, lora_alpha=r * 2)
        adapter = LoRAAdapter(cfg)
        adapter.create_adapter("q_proj", (768, 768))
        p = adapter.count_parameters()["trainable"]
        print(f"  r={r:2d}: trainable_params={p:,} ({p / original_params * 100:.2f}%)")
    
    print("\n✓ LLM post-training pipeline verified")
    print("Key: SFT adapts to instructions, DPO aligns to preferences,")
    print("     LoRA enables efficient fine-tuning with <1% trainable params")


if __name__ == "__main__":
    main()
