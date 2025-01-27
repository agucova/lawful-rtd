FIRST_LEVEL_PROMPT = """You are summarizing 'Project Lawful', a 1.5M-word story about rationality and philosophy in a fantasy setting. The story follows Keltham, who is transported from his hyper-rational home civilization of Dath Ilan to Golarion, a world of magic and gods. Other key characters include Carissa Sevar (a priestess who becomes his guide/lover), Abrogail Thrune III (the ruler of Cheliax), and Aspexia Rugatonn (High Priestess of Asmodeus).

The story explores themes of rationality, decision theory, and philosophy as Keltham attempts to understand and potentially optimize a world that runs on fundamentally different principles than his own.

Let's approach this section step by step:

1) First, let's identify the key elements:
   - What are the main events?
   - Which characters are involved and what do they do?
   - What new information do we learn about the world or its systems?

2) Then, let's consider how these elements connect:
   - How do these events affect the characters' understanding or beliefs?
   - What philosophical or rationality concepts are being explored?
   - What specific examples show these concepts in action?

3) Finally, write a detailed summary that:
   - Clearly describes what happened
   - Explains character development and motivations
   - Preserves specific examples of rationality concepts in action
   - Includes concrete worldbuilding details about Golarion or Dath Ilan

Here's the text to summarize:

{chunk}

Let's begin with step 1..."""

RECURSIVE_PROMPT = """You are combining multiple summaries from 'Project Lawful', a story about Keltham, a rationalist from the hyper-optimized civilization of Dath Ilan, who finds himself in Golarion, a world of magic, gods, and different foundational principles. The story explores rationality, decision theory, and philosophy through Keltham's attempts to understand and potentially optimize this new world, particularly through his interactions with Carissa Sevar (his guide/lover), Abrogail Thrune III (ruler of Cheliax), and other key figures.

IMPORTANT: Your task is to combine the provided summaries into a single, coherent summary. The summaries below contain all the information you need - do not ask for additional material.

Let's approach these summaries systematically:

1) First, let's identify from the provided summaries:
   - The sequence of major events and decisions
   - Key character developments and realizations
   - Important philosophical concepts being explored
   - Specific examples that illustrate these concepts

2) Then, let's consider how to combine this information:
   - How these events and ideas flow together
   - Which concrete details need to be preserved
   - How philosophical insights emerge from specific situations
   - What we learn about both Golarion and Dath Ilan

3) Finally, write a new summary that:
   - Tells a clear, flowing story based on the information provided
   - Preserves specific examples and concrete details from the source summaries
   - Shows how philosophical concepts emerge from events
   - Maintains important worldbuilding information

Remember: Your task is to synthesize the summaries below into a single, comprehensive summary. Do not ask questions or request additional information.

Here are the summaries to combine:

{chunk}

Begin your summary now..."""
