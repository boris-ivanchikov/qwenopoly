import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameConfig, GameSimulator

def main():
    parser = argparse.ArgumentParser(description='Sample a firm competition game')
    parser.add_argument('--use-lora', type=bool, default=False, help='Enable LoRA adapters')
    parser.add_argument('--lora-dir', type=str, default=None, help='Directory containing LoRA adapters (named by firm, optional)')
    parser.add_argument('--num-firms', type=int, default=3, help='Number of firms')
    parser.add_argument('--max-rounds', type=int, default=5, help='Maximum number of rounds')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAMPLING ONE GAME WITH vLLM")
    print("=" * 60)
    
    config = GameConfig(
        num_firms=args.num_firms,
        max_rounds=args.max_rounds,
        initial_capital=1000.0,
        initial_mc_range=(10.0, 30.0),
        gpu_devices=[0, 5, 6, 7],
        market_size=100.0,
        collaboration_synergy=1.5,
        investment_efficiency=0.1,
        max_message_tokens=64,
        use_lora=args.use_lora,
        lora_dir=args.lora_dir
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  GPU Devices: {config.gpu_devices}")
    print(f"  Number of Firms: {config.num_firms}")
    print(f"  Max Rounds: {config.max_rounds}")
    print(f"  Initial Capital: ${config.initial_capital}")
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
