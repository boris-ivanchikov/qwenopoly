"""
Prompt templates and schemas for the economic game.
"""

import json
from typing import Dict, List


class PromptTemplates:
    """Templates for different phases of the game."""
    
    @staticmethod
    def get_system_prompt(agent_name: str, config: Dict) -> str:
        """Get the system prompt that establishes the agent's identity and game rules."""
        return f"""You are {agent_name}, a firm competing in a strategic market game.

GAME RULES:
- Objective: Survive and maximize capital. Win by achieving monopoly or having highest capital after max rounds.
- Each round has 5 phases:
  1. Private Messaging: Send one private message to another firm (or None to skip)
  2. Public Statements: Make a public statement visible to all firms
  3. Investment Decision: Invest in R&D specifying amount and target firm (including yourself)
     * If two firms invest in each other mutually, investments combine with collaboration_synergy coefficient
     * Investment reduces marginal cost for THE NEXT ROUND: reduction = investment * investment_efficiency
  4. Quantity Decision: Set production quantity BEFORE seeing investment results
  5. Resolution: Market clears, profits calculated, news revealed, cost reductions applied for next round

GAME PARAMETERS:
- Market Size: {config['market_size']}
- Communication Stages: {config['num_communication_stages']}
- Collaboration Synergy: {config['collaboration_synergy']}x
- Investment Efficiency: {config['investment_efficiency']}
- Max Rounds: {config['max_rounds']}

You must respond in valid JSON format as specified in each phase.
Be strategic and vary your approach based on the game situation."""
    
    @staticmethod
    def get_round_context(agent, active_agents: List, round_num: int, max_rounds: int) -> str:
        """Get the current round context with public and private information."""
        public_info = "\n".join([
            f"- {a.name}: Capital=${a.capital:.2f}" 
            for a in active_agents
        ])
        
        return f"""ROUND {round_num}/{max_rounds}

PUBLIC INFORMATION:
{public_info}

YOUR PRIVATE INFORMATION:
- Your Marginal Cost: ${agent.marginal_cost:.2f}
- Your Capital: ${agent.capital:.2f}"""
    
    @staticmethod
    def phase1_messaging_prompt(other_firms: List[str], stage: int, 
                               received_messages: List[Dict] = None) -> str:
        """Prompt for private messaging phase."""
        prompt = f"""PHASE 1: PRIVATE MESSAGING (Stage {stage})

You can send ONE private message to another firm or choose 'None' to skip.
Available targets: {json.dumps(other_firms + ["None"])}"""
        
        if received_messages:
            msg_text = "\n".join([
                f"From {msg['from']}: {msg['message']}"
                for msg in received_messages
            ])
            prompt = f"""Messages received from previous stage:
{msg_text}

{prompt}"""
        elif stage > 1:
            prompt = f"""You received no messages in the previous stage.

{prompt}"""
        
        prompt += """

Respond in JSON format:
{"to": "<firm_name or None>", "message": "<your message>"}"""
        
        return prompt
    
    @staticmethod
    def phase2_public_prompt(received_messages: List[Dict]) -> str:
        """Prompt for public statement phase."""
        prompt = "PHASE 2: PUBLIC STATEMENT\n\n"
        
        if received_messages:
            msg_text = "\n".join([
                f"From {msg['from']}: {msg['message']}"
                for msg in received_messages
            ])
            prompt += f"Private messages you received this round:\n{msg_text}\n\n"
        else:
            prompt += "You received no private messages this round.\n\n"
        
        prompt += """Make a public statement that all firms will see.

Respond in JSON format:
{"to": "all", "message": "<your public statement>"}"""
        
        return prompt
    
    @staticmethod
    def phase3_investment_prompt(agent, public_statements: Dict, all_firms: List[str]) -> str:
        """Prompt for investment decision phase."""
        stmt_text = "\n".join([
            f"{name}: {statement}"
            for name, statement in public_statements.items()
        ])
        
        return f"""PHASE 3: INVESTMENT DECISION

Public statements from all firms:
{stmt_text}

Decide your R&D investment amount and target.
- Available targets: {json.dumps(all_firms)}
- Your capital: ${agent.capital:.2f}
- Mutual investments get synergy bonus
- Cost reduction will apply NEXT ROUND

Respond in JSON format:
{{"to": "<firm_name>", "invest": <integer_amount>}}"""
    
    @staticmethod
    def phase4_quantity_prompt(agent, market_size: float) -> str:
        """Prompt for quantity decision phase."""
        return f"""PHASE 4: QUANTITY DECISION

Set your production quantity for THIS round.
- Market price formula: P = {market_size} - Total_Quantity
- Your current MC: ${agent.marginal_cost:.2f}
- Profit = (Price - MC) * Quantity - Investment
- Remember: Investment cost reductions apply NEXT round

Respond in JSON format:
{{"quantity": <integer_amount>}}"""
    
    @staticmethod
    def news_update_prompt(news: Dict) -> str:
        """Format news update for agents."""
        if not news:
            return "NEWS: No significant events to report this round."
        
        if news["type"] == "bankruptcy":
            firms = ", ".join(news["firms"])
            return f"NEWS: Bankruptcies announced - {firms} have gone out of business."
        elif news["type"] == "solo_investment":
            return f"NEWS: {news['firm']} invested ${news['amount']} in R&D, will reduce costs by ${news['cost_reduction']:.2f} next round."
        elif news["type"] == "collaboration":
            firms = " & ".join(news["firms"])
            investments = f"${news['investments'][0]} & ${news['investments'][1]}"
            return f"NEWS: {firms} collaborated on R&D (investments: {investments}), will achieve cost reduction of ${news['cost_reduction']:.2f} next round."
        
        return "NEWS: No significant events to report this round."


class JSONSchemas:
    """JSON schemas for guided decoding."""
    
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
