"""
Agent module for handling LLM inference with vLLM.
Uses LangChain for context management and OpenAI format for conversations.
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class Agent:
    """Represents a firm agent with its own conversation history."""
    
    id: int
    name: str
    marginal_cost: float
    capital: float
    is_active: bool = True
    initial_mc: float = 0.0
    initial_capital: float = 0.0
    conversation_history: List[BaseMessage] = field(default_factory=list)
    log_filepath: Optional[str] = None
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history in OpenAI format."""
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
    """Handles vLLM inference with shared base model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None
        self.initialize_llm()
        
    def initialize_llm(self):
        """Initialize the vLLM model with configuration."""
        tensor_parallel_size = self.config.get("num_gpus", 4)
        
        llm_kwargs = {
            "model": self.config.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 4096,
            "distributed_executor_backend": "ray",
            "enforce_eager": True,
            "enable_prefix_caching": True
        }
        
        self.llm = LLM(**llm_kwargs)
    
    def generate(
            self, 
            agents: List[Agent], 
            json_schema: Optional[Dict] = None,
            max_tokens: int = 256
        ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple agents using chat format.
        
        Args:
            agents: List of Agent objects (conversation already added to their history)
            json_schema: Optional JSON schema for guided decoding
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of response dictionaries with 'text' and 'json' (if applicable)
        """
        conversations = []
        for agent in agents:
            conversations.append(agent.get_conversation_for_vllm())
        
        params_kwargs = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": max_tokens,
        }
        
        if json_schema:
            params_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
        
        sampling_params = SamplingParams(**params_kwargs)
        
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        
        responses = []
        for output, agent in zip(outputs, agents):
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
