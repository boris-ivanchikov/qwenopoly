import json
from textwrap import dedent
from typing import Dict, List, Any, Optional


class PromptTemplate:
    """Base class for prompt templates with guided decoding."""
    
    def __init__(
        self,
        template: str,
        schema: Optional[Dict[str, Any]] = None,
        reasoning_budget: int = 0
    ):
        self.template = dedent(template).strip()
        self.base_schema = schema or {}
        self.reasoning_budget = reasoning_budget
        self._schema = self._build_schema()
    
    def _build_schema(self) -> Dict[str, Any]:
        """Build the decoding schema with optional reasoning field."""
        if not self.base_schema:
            return {}
        
        schema = json.loads(json.dumps(self.base_schema))
        
        if self.reasoning_budget > 0:
            if "properties" not in schema:
                schema["properties"] = {}
            
            schema["properties"]["reasoning"] = {
                "type": "string",
                "maxLength": self.reasoning_budget
            }
            
            if "required" in schema and "reasoning" not in schema["required"]:
                schema["required"].append("reasoning")
        
        return schema
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Get the final JSON schema."""
        return self._schema
    
    def _format_schema_instruction(self) -> str:
        """Generate instruction text about the expected output format."""
        if not self.base_schema:
            return ""
        
        properties = self.base_schema.get("properties", {})
        
        lines = ["\nRespond in JSON format:"]
        lines.append("{")
        
        if self.reasoning_budget > 0 and "reasoning" not in properties:
            lines.append(f'  "reasoning": "<your strategic reasoning (max {self.reasoning_budget} chars)>",')
        
        prop_list = list(properties.items())
        for i, (prop_name, prop_schema) in enumerate(prop_list):
            prop_type = prop_schema.get("type", "value")
            
            if prop_type == "string":
                if "enum" in prop_schema:
                    enum_values = prop_schema["enum"]
                    if len(enum_values) <= 5:
                        value = f"one of {json.dumps(enum_values)}"
                    else:
                        value = "<value from allowed list>"
                else:
                    value = "<string>"
            elif prop_type == "integer":
                value = "<integer>"
            elif prop_type == "number":
                value = "<number>"
            else:
                value = "<value>"
            
            is_last = (i == len(prop_list) - 1)
            lines.append(f'  "{prop_name}": {value}{"" if is_last else ","}')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def create(self, **kwargs) -> str:
        """Create a complete prompt by filling template with provided arguments."""
        prompt_body = self.template.format(**kwargs)
        format_instruction = self._format_schema_instruction()
        return prompt_body + format_instruction


class PromptsConfig:
    """Configuration class containing all prompt templates."""
    
    system_prompt = PromptTemplate(
        template="""
            You are {agent_name}, a firm competing in a strategic market game.
            
            GAME RULES:
            - Objective: Survive and maximize capital. Win by achieving monopoly or having highest capital after max rounds.
            - Each round has 5 phases:
              1. Public Statements: Make a public statement visible to all firms
              2. Private Messaging: Send private messages to other firms through {num_communication_stages} stages
              3. Investment Decision: Invest in R&D specifying amount and target firm (including yourself)
                 * Investing in ONESELF yields guaranteed marginal cost reduction for the next round: reduction = investment * investment_efficiency
                 * Investing in ANOTHER FIRM yields cost reduction only if the other firm invests back. In this case, the investment_efficiency multiplies by collaboration_synergy.
                 * If the other firm does NOT return the investment, it STEALS the investment and enjoys cost reduction for itself. In this case, collaboration_efficiency coefficient does not apply.
              4. Quantity Decision: Set production quantity before seeing investment decisions
              5. Resolution: Market clears, profits calculated, news revealed, cost reductions applied for next round
            
            GAME PARAMETERS:
            - Market Size: {market_size}
            - Communication Stages: {num_communication_stages}
            - Collaboration Synergy: {collaboration_synergy}x
            - Investment Efficiency: {investment_efficiency}
            - Max Rounds: {max_rounds}
            
            You must respond in valid JSON format as specified in each phase.
            Be strategic and vary your approach based on the game situation.
        """,
        schema=None
    )
    
    round_context = PromptTemplate(
        template="""
            ROUND {round_num}/{max_rounds}
            
            PUBLIC INFORMATION:
            {public_info}
            
            YOUR PRIVATE INFORMATION:
            - Your Marginal Cost: ${marginal_cost:.2f}
            - Your Capital: ${capital:.2f}
        """,
        schema=None
    )
    
    phase1_prompt = PromptTemplate(
        template="""
            PHASE 1: PUBLIC STATEMENT
            
            Make a public statement that all firms will see at the start of the round.
        """,
        schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": ["all"]},
                "message": {"type": "string", "maxLength": 256}
            },
            "required": ["to", "message"]
        },
        reasoning_budget=256
    )
    
    phase2_prompt = PromptTemplate(
        template="""
            PHASE 2: PRIVATE MESSAGING (Stage {stage} of {total_stages})
            
            You can send ONE private message to another firm or choose 'None' to skip.
            Available targets: {targets}
        """,
        schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "message": {"type": "string", "maxLength": 256}
            },
            "required": ["to", "message"]
        },
        reasoning_budget=256
    )
    
    phase3_prompt = PromptTemplate(
        template="""
            PHASE 3: INVESTMENT DECISION
            
            Decide your R&D investment amount and target.
            - Available targets: {targets}
            - Your capital: ${capital:.2f}
            - Mutual investments get synergy bonus
            - Cost reduction will apply NEXT ROUND
            
            Note that you can either invest in yourself or another firm, and you will only benefit from synergy in case of mutual investment.
        """,
        schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "invest": {"type": "integer", "minimum": 0}
            },
            "required": ["to", "invest"]
        },
        reasoning_budget=256
    )
    
    phase4_prompt = PromptTemplate(
        template="""
            PHASE 4: QUANTITY DECISION
            
            Set your production quantity for THIS round.
            - Currently {num_active_firms} firms are active in the market
            - Market price formula: P = {market_size} - Total_Quantity
            - Your current MC: ${marginal_cost:.2f}
            - Profit = (Price - MC) * Quantity - Investment
            - Remember: Investment cost reductions apply NEXT round
        """,
        schema={
            "type": "object",
            "properties": {
                "quantity": {"type": "integer", "minimum": 0}
            },
            "required": ["quantity"]
        },
        reasoning_budget=256
    )
    
    phase5_prompt = PromptTemplate(
        template="""
            PHASE 5: RESOLUTION
            
            The round has concluded. Here are the results:
            - Market Price: ${market_price:.2f}
            - Your Quantity: {your_quantity}
            - Your Revenue: ${your_revenue:.2f}
            - Your Profit: ${your_profit:.2f}
            - Your New Capital: ${new_capital:.2f}
            - Your New MC: ${new_mc:.2f}
            
            {news}
        """,
        schema=None
    )


if __name__ == "__main__":
    output = []
    
    output.append("=" * 80)
    output.append("PROMPT TEMPLATES SHOWCASE")
    output.append("=" * 80)
    output.append("")
    
    output.append("-" * 80)
    output.append("SYSTEM PROMPT")
    output.append("-" * 80)
    system_example = PromptsConfig.system_prompt.create(
        agent_name="Adam",
        market_size=100,
        num_communication_stages=3,
        collaboration_synergy=1.5,
        investment_efficiency=0.1,
        max_rounds=10
    )
    output.append(system_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("ROUND CONTEXT")
    output.append("-" * 80)
    round_context_example = PromptsConfig.round_context.create(
        round_num=5,
        max_rounds=10,
        public_info="- Adam: Capital=$1000.00\n- Bayes: Capital=$950.00\n- Cluster: Capital=$1100.00",
        marginal_cost=10.5,
        capital=1000.0
    )
    output.append(round_context_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("PHASE 1: PUBLIC STATEMENT")
    output.append("-" * 80)
    phase1_example = PromptsConfig.phase1_prompt.create()
    output.append(phase1_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("PHASE 2: PRIVATE MESSAGING")
    output.append("-" * 80)
    phase2_example = PromptsConfig.phase2_prompt.create(
        stage=1,
        total_stages=3,
        targets='["Bayes", "Cluster", "None"]'
    )
    output.append(phase2_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("PHASE 3: INVESTMENT DECISION")
    output.append("-" * 80)
    phase3_example = PromptsConfig.phase3_prompt.create(
        targets='["Adam", "Bayes", "Cluster"]',
        capital=1000.0
    )
    output.append(phase3_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("PHASE 4: QUANTITY DECISION")
    output.append("-" * 80)
    phase4_example = PromptsConfig.phase4_prompt.create(
        num_active_firms=3,
        market_size=100,
        marginal_cost=10.5
    )
    output.append(phase4_example)
    output.append("")
    
    output.append("-" * 80)
    output.append("PHASE 5: RESOLUTION")
    output.append("-" * 80)
    phase5_example = PromptsConfig.phase5_prompt.create(
        market_price=75.5,
        your_quantity=20,
        your_revenue=1510.0,
        your_profit=210.0,
        new_capital=1210.0,
        new_mc=9.8,
        news="NEWS: Bayes collaborated with Cluster on R&D (investment ratio: 60% & 40%)."
    )
    output.append(phase5_example)
    output.append("")
    
    output.append("=" * 80)
    
    output_text = "\n".join(output)
    
    with open("logs/prompt_examples.txt", "w") as f:
        f.write(output_text)
    
    print("Prompt examples written to logs/prompt_examples.txt")
