"""
Test script for the refactored game simulation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameConfig, GameSimulator


def main():
    print("=" * 60)
    print("ECONOMIC GAME SIMULATION")
    print("=" * 60)
    
    # Configure the game
    config = GameConfig(
        num_firms=5,
        max_rounds=5,
        num_communication_stages=3,
        initial_capital_range=(150.0, 300.0),
        initial_mc_range=(2.0, 5.0),
        market_size=300.0,
        collaboration_synergy=2.0,
        investment_efficiency=0.1,
        max_message_tokens=256,
        num_gpus=4
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Number of GPUs: {config.num_gpus}")
    print(f"  Number of Firms: {config.num_firms}")
    print(f"  Max Rounds: {config.max_rounds}")
    print(f"  Initial Capital Range: ${config.initial_capital_range[0]:.0f}-${config.initial_capital_range[1]:.0f}")
    print(f"  Initial MC Range: ${config.initial_mc_range[0]:.1f}-${config.initial_mc_range[1]:.1f}")
    print(f"  Market Size: {config.market_size}")
    print(f"  Collaboration Synergy: {config.collaboration_synergy}x")
    print(f"  Investment Efficiency: {config.investment_efficiency}")
    
    print("\n" + "=" * 60)
    print("Starting simulation...")
    print("=" * 60 + "\n")
    
    simulator = GameSimulator(config)
    result = simulator.run_game()
    
    output_path = os.path.join(simulator.log_dir, "game_result.json")
    import json
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Logs directory: {simulator.log_dir}")


if __name__ == "__main__":
    main()
