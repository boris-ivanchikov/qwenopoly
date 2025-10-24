import os
import json
import random
from typing import List, Dict, Optional, Any
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.lora.request import LoRARequest
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage


class Agent:
    """Represents a firm agent with its own conversation history and LoRA configuration."""
    
    def __init__(
        self,
        id: int,
        name: str,
        marginal_cost: float,
        capital: float,
        lora_path: Optional[str] = None,
    ):
        self.id = id
        self.name = name
        self.marginal_cost = marginal_cost
        self.capital = capital
        self.initial_mc = marginal_cost
        self.initial_capital = capital
        self.is_active = True
        self.conversation_history: List[BaseMessage] = []
        self.log_filepath: Optional[str] = None
        self.lora_path = lora_path
        self.lora_seed = random.randint(0, 2**31 - 1)
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        if role == "system":
            msg = SystemMessage(content=content)
        elif role == "user":
            msg = HumanMessage(content=content)
        elif role == "assistant":
            msg = AIMessage(content=content)
        else:
            raise ValueError(f"Invalid role: {role}")
        
        self.conversation_history.append(msg)
        
    def get_conversation_for_vllm(self) -> List[Dict[str, str]]:
        """Get conversation history in OpenAI format for vLLM."""
        return [
            {
                "role": "system" if isinstance(msg, SystemMessage) else 
                        "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self.conversation_history
        ]
    
    def initialize_log_file(self, log_dir: str):
        """Initialize the agent's log file."""
        self.log_filepath = os.path.join(log_dir, f"{self.name}.json")
        with open(self.log_filepath, 'w') as f:
            json.dump([], f, indent=2)
    
    def log_conversation(self):
        """Update the log file with current conversation history."""
        if not self.log_filepath:
            return
        
        with open(self.log_filepath, 'w') as f:
            json.dump(self.get_conversation_for_vllm(), f, indent=2)


class LLMEngine:
    """Handles vLLM inference with LoRA adapter support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None
        self.initialize_llm()
        
    def initialize_llm(self):
        """Initialize the vLLM model with LoRA support."""
        llm_kwargs = {
            "model": self.config.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            "tensor_parallel_size": self.config.get("num_gpus", 4),
            "gpu_memory_utilization": 0.8,
            "max_model_len": 8192,
            "distributed_executor_backend": "ray",
            "enforce_eager": True,
            "enable_prefix_caching": True,
            "enable_lora": True,
            "max_lora_rank": 64,
            "max_loras": 8
        }
        
        self.llm = LLM(**llm_kwargs)
    
    def generate(
        self, 
        agents: List[Agent], 
        json_schema: Dict,
        max_tokens: int = 256
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple agents with LoRA support.
        
        Args:
            agents: List of Agent objects with conversation history
            json_schema: Optional JSON schema for guided decoding
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of response dictionaries with 'text' and 'json' (if applicable)
        """
        conversations = [agent.get_conversation_for_vllm() for agent in agents]
        
        params_kwargs = {
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        
        params_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
        
        sampling_params = SamplingParams(**params_kwargs)
        
        lora_requests = []
        for agent in agents:
            if agent.lora_path and agent.lora_seed is not None:
                lora_requests.append(
                    LoRARequest(
                        lora_name=f"{agent.name}_seed_{agent.lora_seed}",
                        lora_int_id=agent.lora_seed,
                        lora_path=agent.lora_path
                    )
                )
            else:
                lora_requests.append(None)
        
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            lora_request=lora_requests if any(lora_requests) else None,
            use_tqdm=True
        )
        
        responses = []
        for output in outputs:
            response_text = output.outputs[0].text.strip()
            response_data = {"text": response_text}
            
            if json_schema:
                try:
                    response_data["json"] = json.loads(response_text)
                except json.JSONDecodeError:
                    response_data["json"] = None
                    response_data["error"] = "JSON decode error"
            
            responses.append(response_data)
        
        return responses
