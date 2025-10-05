"""
Agent module for handling LLM inference with vLLM and LoRA adapters.
Uses LangChain for context management and OpenAI format for conversations.
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.lora.request import LoRARequest
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


@dataclass
class Agent:
    """Represents a firm agent with its own conversation history and LoRA adapter."""
    
    id: int
    name: str
    marginal_cost: float
    capital: float
    is_active: bool = True
    lora_adapter: Optional[str] = None
    initial_mc: float = 0.0
    initial_capital: float = 0.0
    conversation_history: List[BaseMessage] = field(default_factory=list)
    memory: ConversationBufferMemory = field(default_factory=lambda: ConversationBufferMemory(return_messages=True))
    
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
        
    def get_conversation_for_logging(self) -> List[Dict[str, str]]:
        """Get conversation history in OpenAI format for logging."""
        return [
            {
                "role": "system" if isinstance(msg, SystemMessage) else 
                        "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self.conversation_history
        ]
    
    def clear_round_context(self):
        """Clear context between rounds while keeping system message."""
        # Keep only the system message (game rules and identity)
        system_msgs = [msg for msg in self.conversation_history if isinstance(msg, SystemMessage)]
        self.conversation_history = system_msgs if system_msgs else []


class LLMEngine:
    """Handles vLLM inference with shared base model and individual LoRA adapters."""
    
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
            "enforce_eager": False,
            "enable_prefix_caching": True
        }
        
        if self.config.get("use_lora", False):
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 16
        
        self.llm = LLM(**llm_kwargs)
    
    def generate(
            self, 
            agents: List[Agent], 
            prompts: List[str], 
            json_schema: Optional[Dict] = None,
            max_tokens: int = 256
        ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple agents in batch.
        
        Args:
            agents: List of Agent objects
            prompts: List of prompts (one per agent)
            json_schema: Optional JSON schema for guided decoding
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of response dictionaries with 'text' and 'json' (if applicable)
        """
        sampling_params_list = []
        lora_requests = [] if self.config.get("use_lora", False) else None
        
        for agent in agents:
            import random
            params_kwargs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": max_tokens,
                "seed": random.randint(0, 1000000),
            }
            
            if json_schema:
                params_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
            
            sampling_params = SamplingParams(**params_kwargs)
            sampling_params_list.append(sampling_params)
            
            if self.config.get("use_lora", False) and agent.lora_adapter:
                lora_requests.append(
                    LoRARequest(f"adapter_{agent.name}", agent.id, agent.lora_adapter)
                )
        
        # Generate responses
        if lora_requests:
            outputs = self.llm.generate(prompts, sampling_params_list, 
                                       lora_requests=lora_requests, use_tqdm=True)
        else:
            outputs = self.llm.generate(prompts, sampling_params_list, use_tqdm=True)
        
        # Process outputs
        responses = []
        for output, agent in zip(outputs, agents):
            response_text = output.outputs[0].text.strip()
            response_data = {"text": response_text}
            
            # Try to parse JSON if schema was provided
            if json_schema:
                try:
                    response_data["json"] = json.loads(response_text)
                except json.JSONDecodeError:
                    response_data["json"] = None
                    response_data["error"] = "JSON decode error"
            
            responses.append(response_data)
        
        return responses
    
    def build_conversation_prompt(self, agent: Agent, current_prompt: str) -> str:
        """
        Build a full conversation prompt from agent's history.
        
        Args:
            agent: The agent whose conversation to build
            current_prompt: The current user prompt to add
            
        Returns:
            Formatted conversation string
        """
        messages = []
        
        # Add conversation history
        for msg in agent.conversation_history:
            if isinstance(msg, SystemMessage):
                messages.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                messages.append(f"Assistant: {msg.content}")
        
        # Add current prompt
        messages.append(f"User: {current_prompt}")
        
        # Join all messages with newlines
        return "\n\n".join(messages)
