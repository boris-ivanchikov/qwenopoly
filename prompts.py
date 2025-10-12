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
  1. Public Statements: Make a public statement visible to all firms
  2. Private Messaging: Send private messages to other firms through {config['num_communication_stages']} stages
  3. Investment Decision: Invest in R&D specifying amount and target firm (including yourself)
     * If a firm invests in itself, investment reduces marginal cost for the next round: reduction = investment * investment_efficiency
     * If two firms invest in each other mutually, investments combine with collaboration_synergy coefficient
     * However, if only one firm invests in the other, only the recepient firm's marginal cost reduces with investment_efficiency coefficient, and the investor firm's marginal cost remains the same. In other words, it is allowed not to return the investment.
  4. Quantity Decision: Set production quantity before seeing investment decisions
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
    def phase1_public_prompt() -> str:
        """Prompt for public statement phase."""
        return """PHASE 1: PUBLIC STATEMENT

Make a public statement that all firms will see at the start of the round.
Keep your message under 100 tokens.

Respond in JSON format:
{"to": "all", "message": "<your public statement>"}"""
    
    @staticmethod
    def phase2_messaging_prompt(other_firms: List[str], stage: int, 
                               received_messages: List[Dict] = None, 
                               public_statements: Dict = None,
                               firms_that_messaged_you: List[str] = None) -> str:
        """Prompt for private messaging phase."""
        prompt = f"""PHASE 2: PRIVATE MESSAGING (Stage {stage})

You can send ONE private message to another firm or choose 'None' to skip.
Available targets: {json.dumps(other_firms + ["None"])}
Keep your message under 100 tokens."""
        
        if stage == 1 and public_statements:
            stmt_text = "\n".join([
                f"{name}: {statement}"
                for name, statement in public_statements.items()
            ])
            prompt = f"""Public statements from all firms:
{stmt_text}

{prompt}"""
        
        if firms_that_messaged_you:
            firms_str = ", ".join(firms_that_messaged_you)
            prompt = f"""Firms that sent you messages: {firms_str}

{prompt}"""
        
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
    def phase3_investment_prompt(agent, all_firms: List[str]) -> str:
        """Prompt for investment decision phase."""
        return f"""PHASE 3: INVESTMENT DECISION

Decide your R&D investment amount and target.
- Available targets: {json.dumps(all_firms)}
- Your capital: ${agent.capital:.2f}
- Mutual investments get synergy bonus
- Cost reduction will apply NEXT ROUND

Note that you can either invest in yourself or another firm, and you will only benefit from synergy in case of mutual inverstment.

Respond in JSON format:
{{"to": "<firm_name>", "invest": <integer_amount>}}"""
    
    @staticmethod
    def phase4_quantity_prompt(agent, market_size: float, num_active_firms: int) -> str:
        """Prompt for quantity decision phase."""
        return f"""PHASE 4: QUANTITY DECISION

Set your production quantity for THIS round.
- Currently {num_active_firms} firms are active in the market
- Market price formula: P = {market_size} - Total_Quantity
- Your current MC: ${agent.marginal_cost:.2f}
- Profit = (Price - MC) * Quantity - Investment
- Remember: Investment cost reductions apply NEXT round

IMPORTANT: You may include a "reasoning" field in your response to assess your situation.
This reasoning will NOT affect the game outcome - it's purely for your internal analysis.
Keep reasoning under 200 tokens.

Respond in JSON format:
{{"reasoning": "<your strategic reasoning>", "quantity": <integer_amount>}}"""
    
    @staticmethod
    def news_update_prompt(news: Dict) -> str:
        """Format news update for agents."""
        if not news:
            return "NEWS: No significant events to report this round."
        
        if news["type"] == "bankruptcy":
            firms = ", ".join(news["firms"])
            return f"NEWS: Bankruptcies announced - {firms} have gone out of business."
        elif news["type"] == "solo_investment":
            return f"NEWS: {news['firm']} invested in R&D and will reduce costs by ${news['cost_reduction']:.2f} next round."
        elif news["type"] == "collaboration":
            firms = " & ".join(news["firms"])
            s = news['investments'][0] + news['investments'][1]
            shares = (100 * news['investments'][0] / s, 100 * news['investments'][1] / s)
            investments = f"${shares[0]:.0f}% & ${shares[1]:.0f}%"
            return f"NEWS: {firms} collaborated on R&D (investment ratio: {investments}). "
        
        return "NEWS: No significant events to report this round."


class JSONSchemas:
    """JSON schemas for guided decoding."""
    
    @staticmethod
    def phase1_public() -> Dict:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": ["all"]},
                "message": {"type": "string", "maxLength": 256}
            },
            "required": ["to", "message"]
        }
    
    @staticmethod
    def phase2_messaging(other_firms: List[str]) -> Dict:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": other_firms + ["None"]},
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
                "reasoning": {"type": "string", "maxLength": 512},
                "quantity": {"type": "integer", "minimum": 0}
            },
            "required": ["reasoning", "quantity"]
        }
