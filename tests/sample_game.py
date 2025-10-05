import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameConfig, GameSimulator

def main():
    print("=" * 60)
    print("SAMPLING ONE GAME WITH vLLM")
    print("=" * 60)
    
    config = GameConfig(
        num_firms=7,
        max_rounds=5,
        initial_capital_range=(100.0, 300.0),
        initial_mc_range=(10.0, 30.0),
        gpu_devices=[0, 5, 6, 7],
        market_size=300.0,
        collaboration_synergy=1.5,
        investment_efficiency=0.1,
        max_message_tokens=64,
        use_lora=False,
        lora_dir=""
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  GPU Devices: {config.gpu_devices}")
    print(f"  Number of Firms: {config.num_firms}")
    print(f"  Max Rounds: {config.max_rounds}")
    print(f"  Initial Capital Range: ${config.initial_capital_range[0]:.0f}-${config.initial_capital_range[1]:.0f}")
    print(f"  Initial MC Range: ${config.initial_mc_range[0]:.1f}-${config.initial_mc_range[1]:.1f}")
    print(f"  Market Size: {config.market_size}")
    print(f"  Max Message Tokens: {config.max_message_tokens}")
    print(f"  Use LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  LoRA Directory: {config.lora_dir}")
    
    output_path = os.path.join("tests", "game_trajectory.json")
    
    print(f"\nInitializing game simulator...")
    simulator = GameSimulator(config)
    
    print(f"\nFirms in this game:")
    for firm in simulator.firms:
        lora_status = f"LoRA: {firm.lora_adapter}" if firm.lora_adapter else "LoRA: None"
        print(f"  {firm.name}: MC=${firm.marginal_cost:.2f}, Capital=${firm.capital:.2f}, {lora_status}")
    
    print(f"\nRunning game simulation...")
    print(f"Output will be saved to: {output_path}")
    print("=" * 60)
    
    simulator.save_trajectory(output_path)
    
    print("\n" + "=" * 60)
    print("GAME COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
