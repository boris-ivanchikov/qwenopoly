"""
Logger module for game simulation with proper conversation tracking.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any


class GameLogger:
    """Handles logging of agent conversations and game events."""
    
    def __init__(self, log_dir: str = None):
        """Initialize logger with a game-specific directory."""
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join("logs", f"game_{timestamp}")
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Agent log files
        self.agent_files = {}
        
        # Game trajectory file
        self.trajectory_path = os.path.join(self.log_dir, "game_trajectory.json")
        
        print(f"Game logs will be saved to: {self.log_dir}")
    
    def initialize_agent_log(self, agent_name: str, initial_data: Dict[str, Any]):
        """Initialize a log file for an agent."""
        filepath = os.path.join(self.log_dir, f"{agent_name}.json")
        self.agent_files[agent_name] = filepath
        
        # Create initial log structure
        log_data = {
            "agent": agent_name,
            "initial_state": initial_data,
            "conversation": [],
            "rounds": []
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_round_conversation(self, agent_name: str, round_num: int, 
                              conversation: List[Dict[str, str]]):
        """Log a round's conversation for an agent."""
        if agent_name not in self.agent_files:
            return
        
        filepath = self.agent_files[agent_name]
        
        # Read existing data
        with open(filepath, 'r') as f:
            log_data = json.load(f)
        
        # Add round conversation
        round_data = {
            "round": round_num,
            "conversation": conversation
        }
        log_data["rounds"].append(round_data)
        
        # Write back
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_game_trajectory(self, trajectory: List[Dict[str, Any]]):
        """Log the complete game trajectory."""
        with open(self.trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
    
    def create_human_readable_log(self, agent_name: str):
        """Create a human-readable text log from JSON conversation."""
        if agent_name not in self.agent_files:
            return
        
        json_path = self.agent_files[agent_name]
        text_path = os.path.join(self.log_dir, f"{agent_name}.txt")
        
        with open(json_path, 'r') as f:
            log_data = json.load(f)
        
        with open(text_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"GAME LOG FOR AGENT: {agent_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Initial state
            f.write("INITIAL STATE:\n")
            f.write("-" * 40 + "\n")
            for key, value in log_data["initial_state"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Rounds
            for round_data in log_data["rounds"]:
                f.write("=" * 80 + "\n")
                f.write(f"ROUND {round_data['round']}\n")
                f.write("=" * 80 + "\n\n")
                
                for message in round_data["conversation"]:
                    role = message["role"].upper()
                    content = message["content"]
                    
                    if role == "SYSTEM":
                        f.write("[SYSTEM MESSAGE]\n")
                        f.write("-" * 40 + "\n")
                        f.write(content + "\n\n")
                    
                    elif role == "USER":
                        f.write("[PROMPT]\n")
                        f.write("-" * 40 + "\n")
                        f.write(content + "\n\n")
                    
                    elif role == "ASSISTANT":
                        f.write("[AGENT RESPONSE]\n")
                        f.write("-" * 40 + "\n")
                        f.write(content + "\n\n")
                        
                        # Try to parse and display JSON if present
                        try:
                            json_response = json.loads(content)
                            f.write("Parsed JSON:\n")
                            f.write(json.dumps(json_response, indent=2) + "\n\n")
                        except:
                            pass
    
    def finalize_logs(self, agents: List[Any]):
        """Create human-readable logs for all agents at game end."""
        for agent in agents:
            self.create_human_readable_log(agent.name)
        print(f"Human-readable logs created in {self.log_dir}")
