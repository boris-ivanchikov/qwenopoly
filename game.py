import os
import json
import random
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from agent import Agent, LLMEngine
from prompts import PromptsConfig


@dataclass
class GameConfig:
    """Game configuration parameters."""
    num_firms: int
    max_rounds: int
    num_communication_stages: int
    initial_capital_range: tuple
    initial_mc_range: tuple
    market_size: float
    collaboration_synergy: float
    investment_efficiency: float
    max_message_tokens: int
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    num_gpus: int = 4
    
    firm_names = [
        "Adam", "Bayes", "Cluster", "Data", "Epoch", 
        "Forward", "Grad", "Hash", "Instance", "Jacobi", 
        "Kernel", "Lambda", "Matrix", "Neuron", "Optim", 
        "Pipeline", "Queue", "ReLU", "Sequence", "Tensor", 
        "U-net", "Vector", "Weight", "Xavier", "Yandex", "ZeRO"
    ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for prompt templates."""
        return asdict(self)


@dataclass
class GameContext:
    """Container for game state that resets between games."""
    agents: List[Agent]
    round_number: int
    trajectory: List[Dict]
    pending_cost_reductions: Dict[str, float]
    log_dir: Optional[str]


class GameSimulator:
    """Main game simulator that can run multiple games with a shared LLM engine."""
    
    def __init__(self, config: GameConfig, lora_path: Optional[str] = None, run_name: Optional[str] = None):
        """
        Initialize the game simulator.
        
        Args:
            config: Game configuration
            lora_path: Optional path to base LoRA adapter
            run_name: Optional name for this run. If None, uses timestamp.
        """
        self.config = config
        self.lora_path = lora_path
        self.llm_engine = LLMEngine(config.to_dict())
        self.context: Optional[GameContext] = None
        
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"run_{timestamp}"
        
        self.run_dir = os.path.join("logs", run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.game_counter = 0
        
        print(f"GameSimulator initialized. Run directory: {self.run_dir}")
    
    def _initialize_context(self, log_dir: Optional[str] = None):
        """Initialize a new game context."""
        if log_dir is None:
            self.game_counter += 1
            log_dir = os.path.join(self.run_dir, f"game_{self.game_counter}")
        
        os.makedirs(log_dir, exist_ok=True)
        
        selected_names = random.sample(self.config.firm_names, k=self.config.num_firms)
        agents = []
        
        for i, name in enumerate(selected_names):
            mc = random.uniform(*self.config.initial_mc_range)
            capital = random.uniform(*self.config.initial_capital_range)
            
            agent = Agent(
                id=i,
                name=name,
                marginal_cost=mc,
                capital=capital,
                lora_path=self.lora_path
            )
            
            agent.initialize_log_file(log_dir)
            agents.append(agent)
        
        self.context = GameContext(
            agents=agents,
            round_number=0,
            trajectory=[],
            pending_cost_reductions={},
            log_dir=log_dir
        )
    
    def get_active_agents(self) -> List[Agent]:
        """Get list of active (non-bankrupt) agents."""
        return [a for a in self.context.agents if a.is_active]
    
    def get_other_agent_names(self, agent: Agent) -> List[str]:
        """Get names of other active agents."""
        active = self.get_active_agents()
        return [a.name for a in active if a.id != agent.id]
    
    def is_game_over(self) -> bool:
        """Check if the game should end."""
        active_agents = self.get_active_agents()
        return len(active_agents) <= 1 or self.context.round_number >= self.config.max_rounds
    
    def log_trajectory(self):
        """Save the current game trajectory to file."""
        trajectory_path = os.path.join(self.context.log_dir, "game_trajectory.json")
        with open(trajectory_path, 'w') as f:
            json.dump(self.context.trajectory, f, indent=2)
    
    def play_round(self) -> Dict:
        """Play one complete round of the game."""
        self.context.round_number += 1
        
        for agent in self.context.agents:
            if agent.name in self.context.pending_cost_reductions:
                agent.marginal_cost = max(0, agent.marginal_cost - self.context.pending_cost_reductions[agent.name])
        self.context.pending_cost_reductions = {}
        
        round_data = {"round": self.context.round_number}
        self.context.trajectory.append(round_data)
        
        active_agents = self.get_active_agents()
        
        # PHASE 1: PUBLIC STATEMENTS
        prompts_cfg = PromptsConfig()
        
        for agent in active_agents:
            public_info = "\n".join([f"- {a.name}: Capital=${a.capital:.2f}" for a in active_agents])
            round_context = prompts_cfg.round_context.create(
                round_num=self.context.round_number,
                max_rounds=self.config.max_rounds,
                public_info=public_info,
                marginal_cost=agent.marginal_cost,
                capital=agent.capital
            )
            agent.add_message("user", round_context)
            prompt = prompts_cfg.phase1_prompt.create()
            agent.add_message("user", prompt)
        
        schema = prompts_cfg.phase1_prompt.schema
        responses = self.llm_engine.generate(active_agents, schema, self.config.max_message_tokens)
        
        public_statements = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                public_statements[agent.name] = response["json"]["message"]
        
        for agent in self.context.agents:
            agent.log_conversation()
        
        # PHASE 2: PRIVATE MESSAGING
        all_stage_messages = []
        message_senders_per_agent = {agent.name: [] for agent in active_agents}
        
        for stage in range(1, self.config.num_communication_stages + 1):
            stage_messages = []
            
            for agent in active_agents:
                received_messages = []
                if stage > 1 and all_stage_messages:
                    received_messages = [
                        {"from": msg["from"], "message": msg["message"]}
                        for msg in all_stage_messages[-1]
                        if msg["to"] == agent.name
                    ]
                
                other_firms = self.get_other_agent_names(agent)
                firms_that_sent = message_senders_per_agent.get(agent.name, []) if stage > 1 else None
                
                template = prompts_cfg.phase2_prompt(
                    other_firms, stage, received_messages,
                    public_statements if stage == 1 else None, 
                    firms_that_sent
                )
                prompt = template.create(
                    stage=stage,
                    total_stages=self.config.num_communication_stages,
                    targets=json.dumps(other_firms + ["None"])
                )
                agent.add_message("user", prompt)
            
            schema = prompts_cfg.phase2_prompt(other_firms, stage).schema
            responses = self.llm_engine.generate(active_agents, schema, self.config.max_message_tokens)
            
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
                        if agent.name not in message_senders_per_agent[json_data["to"]]:
                            message_senders_per_agent[json_data["to"]].append(agent.name)
            
            all_stage_messages.append(stage_messages)
        
        for agent in self.context.agents:
            agent.log_conversation()
        
        # PHASE 3: INVESTMENT DECISIONS
        for agent in active_agents:
            all_firm_names = [a.name for a in active_agents]
            prompt = prompts_cfg.phase3_prompt.create(
                targets=json.dumps(all_firm_names),
                capital=agent.capital
            )
            agent.add_message("user", prompt)
        
        schema = prompts_cfg.phase3_prompt.schema
        responses = self.llm_engine.generate(active_agents, schema, self.config.max_message_tokens)
        
        investments = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                invest_amount = min(response["json"]["invest"], int(agent.capital))
                investments[agent.name] = {
                    "target": response["json"]["to"],
                    "amount": invest_amount
                }
        
        for agent in self.context.agents:
            agent.log_conversation()
        
        round_data["investments"] = investments
        self.log_trajectory()
        
        # PHASE 4: QUANTITY DECISIONS
        num_active = len(active_agents)
        for agent in active_agents:
            prompt = prompts_cfg.phase4_prompt.create(
                num_active_firms=num_active,
                market_size=self.config.market_size,
                marginal_cost=agent.marginal_cost
            )
            agent.add_message("user", prompt)
        
        schema = prompts_cfg.phase4_prompt.schema
        responses = self.llm_engine.generate(active_agents, schema, self.config.max_message_tokens)
        
        quantities = {}
        for agent, response in zip(active_agents, responses):
            agent.add_message("assistant", response["text"])
            if response.get("json"):
                quantities[agent.name] = response["json"]["quantity"]
        
        for agent in self.context.agents:
            agent.log_conversation()
        
        round_data["quantities"] = quantities
        self.log_trajectory()
        
        # PHASE 5: RESOLUTION
        collaborations = []
        for agent1 in active_agents:
            for agent2 in active_agents:
                if agent1.id < agent2.id:
                    inv1 = investments.get(agent1.name, {})
                    inv2 = investments.get(agent2.name, {})
                    
                    if (inv1.get("target") == agent2.name and 
                        inv2.get("target") == agent1.name):
                        total_inv = inv1["amount"] + inv2["amount"]
                        synergy = total_inv * self.config.collaboration_synergy * self.config.investment_efficiency
                        
                        self.context.pending_cost_reductions[agent1.name] = self.context.pending_cost_reductions.get(agent1.name, 0) + synergy
                        self.context.pending_cost_reductions[agent2.name] = self.context.pending_cost_reductions.get(agent2.name, 0) + synergy
                        
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
                inv = investments.get(agent.name, {})
                if inv.get("amount", 0) > 0:
                    reduction = inv["amount"] * self.config.investment_efficiency
                    self.context.pending_cost_reductions[agent.name] = self.context.pending_cost_reductions.get(agent.name, 0) + reduction
                    solo_investments.append({
                        "firm": agent.name,
                        "amount": inv["amount"],
                        "cost_reduction": reduction
                    })
        
        for agent in active_agents:
            inv = investments.get(agent.name, {})
            agent.capital -= inv.get("amount", 0)
        
        total_quantity = sum(quantities.get(a.name, 0) for a in active_agents)
        market_price = max(0, self.config.market_size - total_quantity)
        
        profits = {}
        bankruptcies = []
        
        for agent in active_agents:
            quantity = quantities.get(agent.name, 0)
            revenue = market_price * quantity
            cost = agent.marginal_cost * quantity
            profit = revenue - cost
            
            agent.capital += profit
            profits[agent.name] = profit
            
            if agent.capital < 0:
                agent.is_active = False
                bankruptcies.append(agent.name)
        
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
        
        if news:
            news_text = PromptsConfig.news_prompt(news)
            for agent in active_agents:
                agent.add_message("user", news_text)
        
        for agent in self.context.agents:
            agent.log_conversation()
        
        round_data.update({
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
                for a in self.context.agents
            }
        })
        self.log_trajectory()
        
        return round_data
    
    def play_game(self, log_dir: Optional[str] = None) -> Dict:
        """
        Run a complete game simulation.
        
        Args:
            log_dir: Optional directory for logs. If None, auto-generated.
            
        Returns:
            Dictionary with game results including trajectory, winner, etc.
        """
        self._initialize_context(log_dir)
        
        prompts_cfg = PromptsConfig()
        for agent in self.context.agents:
            system_prompt = prompts_cfg.system_prompt.create(
                agent_name=agent.name,
                **self.config.to_dict()
            )
            agent.add_message("system", system_prompt)
        
        print(f"Starting game simulation...")
        print(f"Logs will be saved to: {self.context.log_dir}")
        print(f"Initial agents: {[a.name for a in self.context.agents]}")
        
        while not self.is_game_over():
            print(f"\nRound {self.context.round_number + 1}/{self.config.max_rounds}")
            round_data = self.play_round()
            
            print(f"  Active agents: {[a.name for a in self.get_active_agents()]}")
            if round_data["bankruptcies"]:
                print(f"  Bankruptcies: {round_data['bankruptcies']}")
        
        winner = None
        active = self.get_active_agents()
        if len(active) == 1:
            winner = active[0].name
        
        result = {
            "config": self.config.to_dict(),
            "initial_agents": [
                {
                    "name": a.name,
                    "initial_mc": a.initial_mc,
                    "initial_capital": a.initial_capital,
                    "lora_seed": a.lora_seed
                }
                for a in self.context.agents
            ],
            "trajectory": self.context.trajectory,
            "winner": winner,
            "total_rounds": self.context.round_number,
            "final_states": {
                a.name: {
                    "capital": a.capital,
                    "marginal_cost": a.marginal_cost,
                    "is_active": a.is_active
                }
                for a in self.context.agents
            }
        }
        
        print(f"\nGame complete!")
        print(f"Total rounds: {self.context.round_number}")
        print(f"Winner: {winner or 'No winner (max rounds reached)'}")
        
        with open(os.path.join(self.context.log_dir, "final_result.json"), 'w') as f:
            json.dump(result, f, indent=2)
        
        return result


if __name__ == "__main__":
    config = GameConfig(
        num_firms=8,
        max_rounds=5,
        num_communication_stages=3,
        initial_capital_range=(200.0, 400.0),
        initial_mc_range=(10.0, 30.0),
        market_size=1000.0,
        collaboration_synergy=1.5,
        investment_efficiency=0.1,
        max_message_tokens=256
    )
    
    simulator = GameSimulator(config, lora_path="lora_adapters/base_lora/adapter_model.safetensors")
    result = simulator.play_game()
    
    print(f"\nGame results saved to: {simulator.context.log_dir}")
