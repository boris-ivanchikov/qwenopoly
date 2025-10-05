import json
from typing import Dict, List
from textwrap import dedent


class PromptTemplates:
    @staticmethod
    def get_base_context(firm_name: str, firm_mc: float, active_firms: List, round_number: int, max_rounds: int) -> str:
        public_info = "\n".join([
            f"- {f.name}: Capital=${f.capital:.2f}"
            for f in active_firms
        ])
        
        return dedent(f"""
            You are {firm_name}, a firm competing in a strategic market game.
            
            PUBLIC INFORMATION:
            {public_info}
            
            YOUR PRIVATE INFORMATION:
            Your marginal cost: ${firm_mc:.2f}
            
            Round: {round_number + 1}/{max_rounds}
        """).strip()
    
    @staticmethod
    def phase1_messaging(base_context: str, other_firms: List[str], max_tokens: int) -> str:
        firms_enum = json.dumps(other_firms + ["None"])
        return base_context + dedent(f"""
            
            PHASE 1: PRIVATE MESSAGING
            
            You can send a private message to one other firm or choose 'None' to skip.
            Available targets: {firms_enum}
            
            Your message can propose collaboration, share information, or contain strategic communication.
            Be brief (max {max_tokens} tokens).
            
            Respond in JSON format:
            {{"to": "<firm_name or None>", "message": "<your message>"}}
        """).strip()
    
    @staticmethod
    def phase2_public(base_context: str, messages_received: List[Dict], max_tokens: int) -> str:
        msg_context = "\n".join([
            f"From {msg['from']}: {msg['message']}"
            for msg in messages_received
        ]) if messages_received else "No messages received."
        
        return base_context + dedent(f"""
            
            PRIVATE MESSAGES YOU RECEIVED:
            {msg_context}
            
            PHASE 2: PUBLIC STATEMENT
            
            Make a public statement that all firms will see.
            Be brief (max {max_tokens} tokens).
            
            Respond in JSON format:
            {{"to": "all", "message": "<your public statement>"}}
        """).strip()
    
    @staticmethod
    def phase3_investment(base_context: str, public_statements: Dict, all_firms: List[str], 
                         firm_capital: float, collaboration_synergy: float, investment_efficiency: float) -> str:
        stmt_context = "\n".join([
            f"{fname}: {stmt}"
            for fname, stmt in public_statements.items()
        ])
        
        firms_enum = json.dumps(all_firms)
        
        return base_context + dedent(f"""
            
            PUBLIC STATEMENTS:
            {stmt_context}
            
            PHASE 3: INVESTMENT DECISION
            
            Decide how much to invest in R&D and which firm to invest in (including yourself).
            If two firms invest in each other, investments combine with {collaboration_synergy}x synergy.
            Available firms: {firms_enum}
            
            Your capital: ${firm_capital:.2f}
            Investment reduces marginal cost: cost_reduction = investment * {investment_efficiency}
            
            Respond in JSON format:
            {{"to": "<firm_name>", "invest": <amount as integer>}}
        """).strip()
    
    @staticmethod
    def phase4_quantity(base_context: str, market_size: float, firm_mc: float) -> str:
        return base_context + dedent(f"""
            
            Investment decisions are being finalized...
            
            PHASE 4: QUANTITY DECISION
            
            Set your production quantity. Market price will be determined by total supply.
            Inverse demand: P = {market_size} - Total_Quantity
            
            Your profit = (Price - MC) * Quantity - Investment
            Your current MC: ${firm_mc:.2f}
            
            Note: You must decide quantity BEFORE knowing final MC changes from investments.
            
            Respond in JSON format:
            {{"quantity": <integer amount>}}
        """).strip()


class JSONSchemas:
    @staticmethod
    def phase1_messaging(other_firms: List[str]) -> Dict:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": other_firms + ["None"]},
                "message": {"type": "string", "maxLength": 256}
            },
            "required": ["to", "message"]
        }
    
    @staticmethod
    def phase2_public() -> Dict:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": ["all"]},
                "message": {"type": "string", "maxLength": 256}
            },
            "required": ["to", "message"]
        }
    
    @staticmethod
    def phase3_investment(all_firms: List[str]) -> Dict:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": all_firms},
                "invest": {"type": "integer", "minimum": 0}
            },
            "required": ["to", "invest"]
        }
    
    @staticmethod
    def phase4_quantity() -> Dict:
        return {
            "type": "object",
            "properties": {
                "quantity": {"type": "integer", "minimum": 0}
            },
            "required": ["quantity"]
        }
