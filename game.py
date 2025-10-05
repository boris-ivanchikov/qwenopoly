"""
Main game simulation module with clean architecture.
"""

import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from agent import Agent, LLMEngine
from logger import GameLogger
from prompts import PromptTemplates, JSONSchemas


@dataclass
class GameConfig:
    """Game configuration parameters."""
    num_firms: int = 3
    max_rounds: int = 10
    num_communication_stages: int = 2
    initial_capital_range: tuple = (300.0, 500.0)
    initial_mc_range: tuple = (10.0, 30.0)
    market_size: float = 100.0
    collaboration_synergy: float = 1.5
    investment_efficiency: float = 0.05
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    num_gpus: int = 4
    max_message_tokens: int = 256
    firm_names: List[str] = None
    use_lora: bool = False
    lora_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.firm_names is None:
            self.firm_names = [
                "Adam", "Bayes", "Cluster", "Data", "Epoch", 
                "Forward", "Gradient", "Heuristic", "Intelligence", 
                "Jacobi", "Kernel", "Lambda", "Matrix", "Neuron", "Pipeline"
            ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for prompt templates."""
        return asdict(self)


class GameSimulator:
    """Main game simulator with clean architecture."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.agents = []
        self.round_number = 0
        self.trajectory = []
        self.pending_cost_reductions = {}
        
        self.llm_engine = LLMEngine(config.to_dict())
        self.logger = GameLogger()
        
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize agent firms with random parameters."""
        selected_names = self.config.firm_names[:self.config.num_firms]
        
        for i, name in enumerate(selected_names):
            mc = random.uniform(*self.config.initial_mc_range)
            capital = random.uniform(*self.config.initial_capital_range)
            
            lora_adapter = None
            if self.config.use_lora and self.config.lora_dir:
                lora_path = os.path.join(self.config.lora_dir, name)
                if os.path.exists(lora_path):
                    lora_adapter = lora_path
            
            agent = Agent(
                id=i,
                name=name,
                marginal_cost=mc,
                capital=capital,
                lora_adapter=lora_adapter,
                initial_mc=mc,
                initial_capital=capital
            )
            
            # Set system prompt
            system_prompt = PromptTemplates.get_system_prompt(name, self.config.to_dict())
            agent.add_message("system", system_prompt)
            
            # Initialize logger for this agent
            self.logger.initialize_agent_log(name, {
                "initial_mc": mc,
                "initial_capital": capital
            })
            
            self.agents.append(agent)
    
    def get_active_agents(self) -> List[Agent]:
        """Get list of active (non-bankrupt) agents."""
        return [a for a in self.agents if a.is_active]
    
    def get_other_agent_names(self, agent: Agent) -> List[str]:
        """Get names of other active agents."""
        active = self.get_active_agents()
        return [a.name for a in active if a.id != agent.id]
    
    def phase1_private_messaging(self) -> Dict:
        """Execute private messaging phase with multiple stages."""
        active_agents = self.get_active_agents()
        all_stage_messages = []
        
        for agent in active_agents:
            round_context = PromptTemplates.get_round_context(
                agent, active_agents, self.round_number, self.config.max_rounds
            )
            agent.add_message("user", round_context)
        
        for stage in range(1, self.config.num_communication_stages + 1):
            stage_messages = []
            
            # Collect prompts for all agents
            prompts = []
            for agent in active_agents:
                # Get messages sent to this agent in previous stage
                received_messages = []
                if stage > 1 and all_stage_messages:
                    received_messages = [
                        {"from": msg["from"], "message": msg["message"]}
                        for msg in all_stage_messages[-1]
                        if msg["to"] == agent.name
                    ]
                
                other_firms = self.get_other_agent_names(agent)
                prompt = PromptTemplates.phase1_messaging_prompt(
                    other_firms, stage, received_messages
                )
                
                # Add to agent's conversation
                agent.add_message("user", prompt)
                prompts.append(self.llm_engine.build_conversation_prompt(agent, ""))
            
            # Generate responses
            schema = JSONSchemas.phase1_messaging(
                self.get_other_agent_names(active_agents[0])
            )
            responses = self.llm_engine.generate(
                active_agents, prompts, schema, self.config.max_message_tokens
            )
            
            # Process responses and update conversations
            for agent, response in zip(active_agents, responses):
                agent.add_message("assistant", response["text"])
                
                if response.get("json"):
                    json_data = response["json"]
                    if json_data["to"] != "None":
                        stage_messages.append({
                            "from": agent.name,
                            "to": json_data["to"],
                            "message": json_data["message"]
                        })
            
            all_stage_messages.append(stage_messages)
        
        # Prepare final messages per agent
        final_messages = {}
        for agent in active_agents:
            agent_messages = []
            for stage_msgs in all_stage_messages:
                agent_messages.extend([
                    {"from": msg["from"], "message": msg["message"]}
                    for msg in stage_msgs
                    if msg["to"] == agent.name
                ])
            final_messages[agent.name] = agent_messages
        
        return {
            "stages": all_stage_messages,
            "final_messages_per_agent": final_messages
        }
    
    def phase2_public_statements(self, private_messages: Dict) -> Dict:
        """Execute public statement phase."""
        active_agents = self.get_active_agents()
        prompts = []
        
        for agent in active_agents:
            received = private_messages["final_messages_per_agent"].get(agent.name, [])
            prompt = PromptTemplates.phase2_public_prompt(received)
            
            agent.add_message("user", prompt)
            prompts.append(self.llm_engine.build_conversation_prompt(agent, ""))
        
        # Generate responses
        schema = JSONSchemas.phase2_public()
        responses = self.llm_engine.generate(
            active_agents, prompts, schema, self.config.max_message_tokens
        )
        
        # Process responses
        statements = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                statements[agent.name] = response["json"]["message"]
        
        return {"public_statements": statements}
    
    def phase3_investments(self, public_statements: Dict) -> Dict:
        """Execute investment decision phase."""
        active_agents = self.get_active_agents()
        prompts = []
        
        for agent in active_agents:
            all_firm_names = [a.name for a in active_agents]
            prompt = PromptTemplates.phase3_investment_prompt(
                agent, public_statements["public_statements"], all_firm_names
            )
            
            agent.add_message("user", prompt)
            prompts.append(self.llm_engine.build_conversation_prompt(agent, ""))
        
        # Generate responses
        schema = JSONSchemas.phase3_investment([a.name for a in active_agents])
        responses = self.llm_engine.generate(
            active_agents, prompts, schema, self.config.max_message_tokens
        )
        
        # Process responses
        investments = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                invest_amount = min(response["json"]["invest"], int(agent.capital))
                investments[agent.name] = {
                    "target": response["json"]["to"],
                    "amount": invest_amount
                }
        
        return {"investments": investments}
    
    def phase4_quantities(self) -> Dict:
        """Execute quantity decision phase."""
        active_agents = self.get_active_agents()
        prompts = []
        
        for agent in active_agents:
            prompt = PromptTemplates.phase4_quantity_prompt(
                agent, self.config.market_size
            )
            
            agent.add_message("user", prompt)
            prompts.append(self.llm_engine.build_conversation_prompt(agent, ""))
        
        # Generate responses
        schema = JSONSchemas.phase4_quantity()
        responses = self.llm_engine.generate(
            active_agents, prompts, schema, self.config.max_message_tokens
        )
        
        # Process responses
        quantities = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                quantities[agent.name] = response["json"]["quantity"]
        
        return {"quantities": quantities}
    
    def phase5_resolution(self, investments: Dict, quantities: Dict) -> Dict:
        """Execute resolution phase - calculate profits, apply investments, etc."""
        active_agents = self.get_active_agents()
        inv_data = investments["investments"]
        qty_data = quantities["quantities"]
        
        self.pending_cost_reductions = {}
        
        collaborations = []
        for agent1 in active_agents:
            for agent2 in active_agents:
                if agent1.id < agent2.id:
                    inv1 = inv_data.get(agent1.name, {})
                    inv2 = inv_data.get(agent2.name, {})
                    
                    if (inv1.get("target") == agent2.name and 
                        inv2.get("target") == agent1.name):
                        total_inv = inv1["amount"] + inv2["amount"]
                        synergy = total_inv * self.config.collaboration_synergy * self.config.investment_efficiency
                        
                        self.pending_cost_reductions[agent1.name] = self.pending_cost_reductions.get(agent1.name, 0) + synergy
                        self.pending_cost_reductions[agent2.name] = self.pending_cost_reductions.get(agent2.name, 0) + synergy
                        
                        collaborations.append({
                            "firms": [agent1.name, agent2.name],
                            "investments": [inv1["amount"], inv2["amount"]],
                            "cost_reduction": synergy
                        })
        
        collab_firms = set()
        for collab in collaborations:
            collab_firms.update(collab["firms"])
        
        solo_investments = []
        for agent in active_agents:
            if agent.name not in collab_firms:
                inv = inv_data.get(agent.name, {})
                if inv.get("amount", 0) > 0:
                    reduction = inv["amount"] * self.config.investment_efficiency
                    self.pending_cost_reductions[agent.name] = self.pending_cost_reductions.get(agent.name, 0) + reduction
                    solo_investments.append({
                        "firm": agent.name,
                        "amount": inv["amount"],
                        "cost_reduction": reduction
                    })
        
        # Deduct investments from capital
        for agent in active_agents:
            inv = inv_data.get(agent.name, {})
            agent.capital -= inv.get("amount", 0)
        
        # Calculate market results
        total_quantity = sum(qty_data.get(a.name, 0) for a in active_agents)
        market_price = max(0, self.config.market_size - total_quantity)
        
        profits = {}
        bankruptcies = []
        
        for agent in active_agents:
            quantity = qty_data.get(agent.name, 0)
            revenue = market_price * quantity
            cost = agent.marginal_cost * quantity
            profit = revenue - cost
            
            agent.capital += profit
            profits[agent.name] = profit
            
            if agent.capital < 0:
                agent.is_active = False
                bankruptcies.append(agent.name)
        
        # Generate news
        news = None
        if bankruptcies:
            news = {"type": "bankruptcy", "firms": bankruptcies}
        else:
            news_options = []
            for inv in solo_investments:
                if inv["amount"] > 0:
                    news_options.append({
                        "type": "solo_investment",
                        "firm": inv["firm"],
                        "amount": inv["amount"],
                        "cost_reduction": inv["cost_reduction"]
                    })
            for collab in collaborations:
                news_options.append({
                    "type": "collaboration",
                    "firms": collab["firms"],
                    "investments": collab["investments"],
                    "cost_reduction": collab["cost_reduction"]
                })
            
            if news_options:
                news = random.choice(news_options)
        
        # Add news to agents' context
        if news:
            news_text = PromptTemplates.news_update_prompt(news)
            for agent in active_agents:
                agent.add_message("user", news_text)
        
        return {
            "collaborations": collaborations,
            "solo_investments": solo_investments,
            "market_price": market_price,
            "total_quantity": total_quantity,
            "profits": profits,
            "bankruptcies": bankruptcies,
            "news": news,
            "firm_states": {
                a.name: {
                    "capital": a.capital,
                    "marginal_cost": a.marginal_cost,
                    "is_active": a.is_active
                }
                for a in self.agents
            }
        }
    
    def play_round(self) -> Dict:
        """Play one complete round of the game."""
        self.round_number += 1
        
        for agent in self.agents:
            agent.clear_round_context()
            if agent.name in self.pending_cost_reductions:
                agent.marginal_cost = max(0, agent.marginal_cost - self.pending_cost_reductions[agent.name])
        
        self.pending_cost_reductions = {}
        
        # Execute phases
        phase1_data = self.phase1_private_messaging()
        phase2_data = self.phase2_public_statements(phase1_data)
        phase3_data = self.phase3_investments(phase2_data)
        phase4_data = self.phase4_quantities()
        phase5_data = self.phase5_resolution(phase3_data, phase4_data)
        
        # Log conversations for each agent
        for agent in self.agents:
            self.logger.log_round_conversation(
                agent.name, 
                self.round_number,
                agent.get_conversation_for_logging()
            )
        
        # Build round data
        round_data = {
            "round": self.round_number,
            "phase1": phase1_data,
            "phase2": phase2_data,
            "phase3": phase3_data,
            "phase4": phase4_data,
            "phase5": phase5_data
        }
        
        self.trajectory.append(round_data)
        return round_data
    
    def is_game_over(self) -> bool:
        """Check if the game should end."""
        active_agents = self.get_active_agents()
        return len(active_agents) <= 1 or self.round_number >= self.config.max_rounds
    
    def run_game(self) -> Dict:
        """Run the complete game simulation."""
        print("Starting game simulation...")
        print(f"Initial agents: {[a.name for a in self.agents]}")
        
        while not self.is_game_over():
            print(f"\nRound {self.round_number + 1}/{self.config.max_rounds}")
            round_data = self.play_round()
            
            # Print round summary
            print(f"  Active agents: {[a.name for a in self.get_active_agents()]}")
            if round_data["phase5"]["bankruptcies"]:
                print(f"  Bankruptcies: {round_data['phase5']['bankruptcies']}")
        
        # Calculate final results
        winner = None
        active = self.get_active_agents()
        if len(active) == 1:
            winner = active[0].name
        
        # Log game trajectory
        self.logger.log_game_trajectory(self.trajectory)
        
        # Create human-readable logs
        self.logger.finalize_logs(self.agents)
        
        result = {
            "config": self.config.to_dict(),
            "initial_agents": [
                {
                    "name": a.name,
                    "initial_mc": a.initial_mc,
                    "initial_capital": a.initial_capital
                }
                for a in self.agents
            ],
            "trajectory": self.trajectory,
            "winner": winner,
            "total_rounds": self.round_number
        }
        
        print(f"\nGame complete!")
        print(f"Total rounds: {self.round_number}")
        print(f"Winner: {winner or 'No winner (max rounds reached)'}")
        print(f"Logs saved to: {self.logger.log_dir}")
        
        return result


def main():
    """Run a sample game."""
    config = GameConfig(
        num_firms=3,
        max_rounds=5,
        initial_capital_range=(200.0, 400.0),
        initial_mc_range=(10.0, 30.0),
        market_size=200.0
    )
    
    simulator = GameSimulator(config)
    result = simulator.run_game()
    
    # Save final trajectory
    with open(os.path.join(simulator.logger.log_dir, "final_result.json"), 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
