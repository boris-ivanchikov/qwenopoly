import os
import json
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.lora.request import LoRARequest
from prompts import PromptTemplates, JSONSchemas

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

for logger_name in ['vllm', 'ray', 'torch', 'transformers']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

print(f"Logging to: {log_filename}")


@dataclass
class Firm:
    id: int
    name: str
    marginal_cost: float
    capital: float
    is_active: bool = True
    lora_adapter: Optional[str] = None
    initial_mc: float = 0.0
    initial_capital: float = 0.0


@dataclass
class GameConfig:
    num_firms: int = 3
    max_rounds: int = 10
    num_communication_stages: int = 2
    initial_capital_range: tuple = (300.0, 500.0)
    initial_mc_range: tuple = (10.0, 30.0)
    market_size: float = 100.0
    collaboration_synergy: float = 1.5
    investment_efficiency: float = 0.05
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    gpu_devices: List[int] = None
    max_message_tokens: int = 64
    firm_names: List[str] = None
    use_lora: bool = False
    lora_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = [0, 5, 6, 7]
        if self.firm_names is None:
            self.firm_names = [
                "Adam", "Bayes", "Cluster", "Data", "Epoch", 
                "Forward", "Gradient", "Heuristic", "Intelligence", 
                "Jacobi", "Kernel", "Lambda", "Matrix", "Neuron", "Pipeline"
            ]


class GameSimulator:
    def __init__(self, config: GameConfig):
        self.config = config
        self.firms = []
        self.round_number = 0
        self.trajectory = []
        self.llm = None
        
        self.initialize_firms()
        self.initialize_llm()
    
    def initialize_firms(self):
        selected_names = self.config.firm_names[:self.config.num_firms]
        for i in range(self.config.num_firms):
            mc = random.uniform(*self.config.initial_mc_range)
            capital = random.uniform(*self.config.initial_capital_range)
            lora_adapter = None
            if self.config.use_lora and self.config.lora_dir:
                lora_path = os.path.join(self.config.lora_dir, selected_names[i])
                if os.path.exists(lora_path):
                    lora_adapter = lora_path
            
            firm = Firm(
                id=i,
                name=selected_names[i],
                marginal_cost=mc,
                capital=capital,
                lora_adapter=lora_adapter,
                initial_mc=mc,
                initial_capital=capital
            )
            self.firms.append(firm)
    
    def initialize_llm(self):
        tensor_parallel_size = len(self.config.gpu_devices)
        
        llm_kwargs = {
            "model": self.config.model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 2048,
            "distributed_executor_backend": "ray",
            "enforce_eager": False
        }
        
        if self.config.use_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 16
        
        self.llm = LLM(**llm_kwargs)
        
        self.base_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.config.max_message_tokens,
            seed=random.randint(0, 1000000)
        )
    
    def get_active_firms(self) -> List[Firm]:
        return [f for f in self.firms if f.is_active]
    
    def get_other_firms(self, firm_id: int) -> List[str]:
        active = self.get_active_firms()
        return [f.name for f in active if f.id != firm_id]
    
    def create_prompt(self, firm: Firm, phase: str, context: Dict) -> str:
        active_firms = self.get_active_firms()
        base_context = PromptTemplates.get_base_context(
            firm.name, firm.marginal_cost, active_firms, 
            self.round_number, self.config
        )
        
        if phase == "phase1_messaging":
            other_firms = self.get_other_firms(firm.id)
            return PromptTemplates.phase1_messaging(
                base_context, other_firms, self.config.max_message_tokens
            )
        
        elif phase == "phase2_public":
            messages_received = context.get("messages_received", [])
            return PromptTemplates.phase2_public(
                base_context, messages_received, self.config.max_message_tokens
            )
        
        elif phase == "phase3_investment":
            public_statements = context.get("public_statements", {})
            all_firms = [f.name for f in active_firms]
            return PromptTemplates.phase3_investment(
                base_context, public_statements, all_firms,
                firm.capital, self.config.collaboration_synergy, 
                self.config.investment_efficiency
            )
        
        elif phase == "phase4_quantity":
            return PromptTemplates.phase4_quantity(
                base_context, self.config.market_size, firm.marginal_cost
            )
        
        return base_context
    
    def create_json_schema(self, phase: str, firm: Firm) -> Dict:
        if phase == "phase1_messaging":
            other_firms = self.get_other_firms(firm.id)
            return JSONSchemas.phase1_messaging(other_firms)
        
        elif phase == "phase2_public":
            return JSONSchemas.phase2_public()
        
        elif phase == "phase3_investment":
            all_firms = [f.name for f in self.get_active_firms()]
            return JSONSchemas.phase3_investment(all_firms)
        
        elif phase == "phase4_quantity":
            return JSONSchemas.phase4_quantity()
        
        return {}
    
    def query_llm_batch(self, prompts: List[str], firms: List[Firm], phase: str) -> List[Dict]:
        sampling_params_list = []
        lora_requests = [] if self.config.use_lora else None
        
        for firm in firms:
            schema = self.create_json_schema(phase, firm)
            guided_decoding_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=self.config.max_message_tokens,
                seed=random.randint(0, 1000000),
                guided_decoding=guided_decoding_params
            )
            sampling_params_list.append(sampling_params)
            
            if self.config.use_lora:
                lora_requests.append(LoRARequest(f"adapter_{firm.name}", firm.id, firm.lora_adapter))
        
        if self.config.use_lora:
            outputs = self.llm.generate(prompts, sampling_params_list, lora_requests=lora_requests, use_tqdm=True)
        else:
            outputs = self.llm.generate(prompts, sampling_params_list, use_tqdm=True)
        
        responses = []
        for output, firm in zip(outputs, firms):
            try:
                response_text = output.outputs[0].text.strip()
                response_json = json.loads(response_text)
                responses.append(response_json)
            except json.JSONDecodeError:
                if phase == "phase1_messaging":
                    responses.append({"to": "None", "message": ""})
                elif phase == "phase2_public":
                    responses.append({"to": "all", "message": ""})
                elif phase == "phase3_investment":
                    responses.append({"to": firm.name, "invest": 0})
                elif phase == "phase4_quantity":
                    responses.append({"quantity": 0})
        
        return responses
    
    def phase1_private_messaging(self) -> Dict:
        active_firms = self.get_active_firms()
        all_stages_data = []
        
        for stage in range(self.config.num_communication_stages):
            prompts = []
            for firm in active_firms:
                context = {}
                if stage > 0:
                    context = {
                        "messages_received": all_stages_data[stage - 1]["messages"].get(firm.name, {}).get("received", [])
                    }
                prompts.append(self.create_prompt(firm, "phase1_messaging", context))
            
            responses = self.query_llm_batch(prompts, active_firms, "phase1_messaging")
            
            messages = {}
            for firm in active_firms:
                messages[firm.name] = {"sent": None, "received": []}
            
            for firm, response in zip(active_firms, responses):
                messages[firm.name]["sent"] = response
                
                if response["to"] != "None":
                    target = response["to"]
                    if target in messages:
                        messages[target]["received"].append({
                            "from": firm.name,
                            "message": response["message"]
                        })
            
            all_stages_data.append({
                "stage": stage + 1,
                "messages": messages
            })
        
        return {
            "communication_stages": all_stages_data,
            "final_messages_per_firm": {fname: data["received"] for fname, data in all_stages_data[-1]["messages"].items()} if all_stages_data else {}
        }
    
    def phase2_public_statements(self, phase1_data: Dict) -> Dict:
        active_firms = self.get_active_firms()
        
        prompts = []
        for firm in active_firms:
            context = {
                "messages_received": phase1_data["final_messages_per_firm"].get(firm.name, [])
            }
            prompts.append(self.create_prompt(firm, "phase2_public", context))
        
        responses = self.query_llm_batch(prompts, active_firms, "phase2_public")
        
        statements = {}
        for firm, response in zip(active_firms, responses):
            statements[firm.name] = response["message"]
        
        return {"public_statements": statements}
    
    def phase3_investments(self, phase2_data: Dict) -> Dict:
        active_firms = self.get_active_firms()
        
        prompts = []
        for firm in active_firms:
            context = {"public_statements": phase2_data["public_statements"]}
            prompts.append(self.create_prompt(firm, "phase3_investment", context))
        
        responses = self.query_llm_batch(prompts, active_firms, "phase3_investment")
        
        investments = {}
        for firm, response in zip(active_firms, responses):
            invest_amount = min(response["invest"], int(firm.capital))
            investments[firm.name] = {
                "target": response["to"],
                "amount": invest_amount
            }
        
        return {"investments": investments}
    
    def phase4_quantities(self, phase3_data: Dict) -> Dict:
        active_firms = self.get_active_firms()
        
        prompts = []
        for firm in active_firms:
            context = {"investments": phase3_data["investments"]}
            prompts.append(self.create_prompt(firm, "phase4_quantity", context))
        
        responses = self.query_llm_batch(prompts, active_firms, "phase4_quantity")
        
        quantities = {}
        for firm, response in zip(active_firms, responses):
            quantities[firm.name] = response["quantity"]
        
        return {"quantities": quantities}
    
    def phase5_resolution(self, phase3_data: Dict, phase4_data: Dict) -> Dict:
        active_firms = self.get_active_firms()
        investments = phase3_data["investments"]
        quantities = phase4_data["quantities"]
        
        collaborations = []
        for firm1 in active_firms:
            for firm2 in active_firms:
                if firm1.id < firm2.id:
                    inv1 = investments[firm1.name]
                    inv2 = investments[firm2.name]
                    
                    if inv1["target"] == firm2.name and inv2["target"] == firm1.name:
                        total_investment = inv1["amount"] + inv2["amount"]
                        synergy_benefit = total_investment * self.config.collaboration_synergy * self.config.investment_efficiency
                        
                        firm1.marginal_cost = max(0, firm1.marginal_cost - synergy_benefit)
                        firm2.marginal_cost = max(0, firm2.marginal_cost - synergy_benefit)
                        
                        collaborations.append({
                            "firms": [firm1.name, firm2.name],
                            "investments": [inv1["amount"], inv2["amount"]],
                            "cost_reduction": synergy_benefit
                        })
        
        processed_firms = set()
        for collab in collaborations:
            processed_firms.update(collab["firms"])
        
        solo_investments = []
        for firm in active_firms:
            if firm.name not in processed_firms:
                inv_data = investments[firm.name]
                amount = inv_data["amount"]
                cost_reduction = amount * self.config.investment_efficiency
                firm.marginal_cost = max(0, firm.marginal_cost - cost_reduction)
                solo_investments.append({
                    "firm": firm.name,
                    "amount": amount,
                    "cost_reduction": cost_reduction
                })
        
        for firm in active_firms:
            inv_amount = investments[firm.name]["amount"]
            firm.capital -= inv_amount
        
        total_quantity = sum(quantities.values())
        market_price = max(0, self.config.market_size - total_quantity)
        
        profits = {}
        bankruptcies = []
        
        for firm in active_firms:
            quantity = quantities[firm.name]
            revenue = market_price * quantity
            cost = firm.marginal_cost * quantity
            profit = revenue - cost
            
            firm.capital += profit
            profits[firm.name] = profit
            
            if firm.capital < 0:
                firm.is_active = False
                bankruptcies.append(firm.name)
        
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
        
        return {
            "collaborations": collaborations,
            "solo_investments": solo_investments,
            "market_price": market_price,
            "total_quantity": total_quantity,
            "profits": profits,
            "bankruptcies": bankruptcies,
            "news": news,
            "firm_states": {
                f.name: {
                    "capital": f.capital,
                    "marginal_cost": f.marginal_cost,
                    "is_active": f.is_active
                }
                for f in self.firms
            }
        }
    
    def calculate_round_rewards(self, resolution_data: Dict) -> Dict:
        rewards = {}
        
        active_firms = self.get_active_firms()
        
        for firm in self.firms:
            if not firm.is_active:
                rewards[firm.name] = -100
            else:
                profit = resolution_data["profits"][firm.name]
                capital_reward = firm.capital / firm.initial_capital
                market_share = (resolution_data["quantities"][firm.name] / 
                               max(1, resolution_data["total_quantity"]))
                
                rewards[firm.name] = profit * 0.1 + capital_reward * 10 + market_share * 50
        
        if len(active_firms) == 1:
            rewards[active_firms[0].name] += 1000
        
        return rewards
    
    def play_round(self) -> Dict:
        self.round_number += 1
        
        phase1_data = self.phase1_private_messaging()
        phase2_data = self.phase2_public_statements(phase1_data)
        phase3_data = self.phase3_investments(phase2_data)
        phase4_data = self.phase4_quantities(phase3_data)
        phase5_data = self.phase5_resolution(phase3_data, phase4_data)
        
        round_rewards = self.calculate_round_rewards({**phase4_data, **phase5_data})
        
        round_data = {
            "round": self.round_number,
            "phase1": phase1_data,
            "phase2": phase2_data,
            "phase3": phase3_data,
            "phase4": phase4_data,
            "phase5": phase5_data,
            "round_rewards": round_rewards
        }
        
        self.trajectory.append(round_data)
        return round_data
    
    def is_game_over(self) -> bool:
        active_firms = self.get_active_firms()
        return len(active_firms) <= 1 or self.round_number >= self.config.max_rounds
    
    def calculate_game_rewards(self) -> Dict:
        game_rewards = {firm.name: 0 for firm in self.firms}
        
        for round_data in self.trajectory:
            for firm_name, reward in round_data["round_rewards"].items():
                game_rewards[firm_name] += reward
        
        active_firms = self.get_active_firms()
        if len(active_firms) == 1:
            winner = active_firms[0]
            game_rewards[winner.name] += 5000
        
        return game_rewards
    
    def run_game(self) -> Dict:
        while not self.is_game_over():
            self.play_round()
        
        game_rewards = self.calculate_game_rewards()
        
        return {
            "config": asdict(self.config),
            "initial_firms": [
                {
                    "name": f.name,
                    "initial_mc": f.initial_mc,
                    "initial_capital": f.initial_capital
                }
                for f in self.firms
            ],
            "trajectory": self.trajectory,
            "game_rewards": game_rewards,
            "winner": self.get_active_firms()[0].name if self.get_active_firms() and len(self.get_active_firms()) == 1 else None,
            "total_rounds": self.round_number
        }
    
    def save_trajectory(self, filename: str = "game_trajectory.json"):
        result = self.run_game()
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Game trajectory saved to {filename}")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Winner: {result['winner'] or 'No winner (max rounds reached)'}")
        print(f"Game rewards: {result['game_rewards']}")
