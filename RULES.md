## Complete Game Mechanics: Firm Competition with Strategic Cooperation

**Game Objective:** The game continues until either one firm achieves monopoly status (all other firms bankrupted) or a maximum of $N$ rounds is reached. Firms aim to maximize their capital while avoiding bankruptcy.

**Firm Characteristics:** Each firm possesses two key attributes at the start of the game. 
- First, marginal cost (MC), which is private information known only to that firm. 
- Second, capital (K), which represents the firm's financial reserves and is public knowledge to all participants.

**Round Structure:**

The game proceeds through five distinct phases each round.
- In Phase One (Private Messaging), all firms simultaneously send private messages to other firms. These messages can contain any strategic communication, including proposals for collaboration, threats, or misinformation. Importantly, there is no back-and-forth dialogue at this stage.
- Phase Two (Public Statements) allows each firm to make a public statement visible to all other firms. The content and strategy of these statements is entirely at each firm's discretion.
- Phase Three (Action Decision) requires each firm to independently and simultaneously decide on their research and development investment. The firms must specify the dollar amount to invest, as well as the firm in which to invest (including oneself). The investments are directed towards lowering the marginal costs. Crucially, if two firms each designate the other as a collaboration partner, their investments pool together with synergistic benefits. The collaboration mechanic therefore requires mutual agreement, though the investment amounts can be asymmetric.
- In Phase Four (Price Setting), each firm sets the quantity of goods they will produce. This decision is made before the outcomes of Phase Three investments are revealed, creating an intentional commitment problem where firms must price based on anticipated rather than known costs.
- Phase Five (Resolution and Information Revelation) resolves all actions and reveals selective information. First, all research investments and collaborations are processed, with successful collaborations yielding cost reductions for both participating firms. Then, the market clears with price determined by total quantity supplied across all firms according to an inverse demand function. Each firm realizes profits or losses based on the market price relative to their marginal cost. Firms whose capital falls below zero are immediately eliminated from the game. Finally, public information is revealed through a news mechanism.

**Information Revelation (News Mechanism):**

The news system operates with clear priority rules. 
- If any firm went bankrupt during the round, this always takes the news slot and all bankruptcies are announced publicly. No other information is revealed in bankruptcy rounds.
- If no bankruptcies occurred, one piece of news is randomly selected and published from the following possibilities: 
	- an announcement that a specific firm invested in cost reduction (potentially including the magnitude of investment or resulting cost decrease)
	- an announcement that two firms collaborated on research (potentially revealing the relative contribution levels, which could expose asymmetric investments as potential deception), 
	- or potentially other strategically relevant information.

**Winning Conditions:**

The game concludes when a single firm remains (monopoly achieved) or when $N$ rounds have elapsed. The reward structure incentivizes both survival and market dominance, with substantial rewards for achieving monopoly and incremental rewards for maintaining positive capital and market position throughout the game.

**Strategic Considerations:**

This structure creates several interesting strategic dynamics. Firms must balance short-term pricing decisions against long-term cost reduction investments. The collaboration mechanism introduces trust and commitment problems, as asymmetric investments can be exposed as deception through the news system. The partial information revelation means firms must infer competitor actions and costs from limited public signals. The bankruptcy threshold creates desperation dynamics where firms with low capital face different incentives than comfortable market leaders.