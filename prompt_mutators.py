"""
Prompt mutation and crossover module for the GAPE framework.
Uses Agno agents to generate variations of prompts through mutation and crossover operations.
"""
from agno.agent import Agent
from agno.models.anthropic import Claude


class MutationEngine:
    """Handles prompt mutation and crossover operations using Agno agents."""
    
    def __init__(self, model=None):
        self.model = model or Claude(id="claude-3-7-sonnet-latest")
        self.agent = Agent(
            model=self.model,
            instructions=[
                "You are an expert at creating effective variations of prompts.",
                "When asked to modify a prompt, focus on improving its clarity, specificity, and effectiveness."
            ]
        )
    
    def mutate_prompt(self, prompt, mutation_type=None):
        """
        Generate a mutated version of the prompt.
        
        mutation_type can be:
        - "instruction": Modify the instruction component
        - "wording": Change the wording/phrasing
        - "structure": Modify the structure/format
        - "random": Apply a random mutation
        - None: Agent decides best mutation approach
        """
        mutation_prompts = {
            "instruction": f"Create a variation of this prompt by modifying the core instruction while preserving its intent:\n\n{prompt.text}",
            "wording": f"Rephrase this prompt using different words but keeping the same meaning:\n\n{prompt.text}",
            "structure": f"Restructure this prompt by changing its format, order, or organization:\n\n{prompt.text}",
            "random": f"Create an interesting variation of this prompt that might work better:\n\n{prompt.text}"
        }
        
        # If mutation type not specified, use random
        mutation_type = mutation_type or "random"
        
        # Get mutation prompt
        mutation_prompt = mutation_prompts.get(mutation_type, mutation_prompts["random"])
        
        # Generate the mutated prompt
        mutated_text = self.agent.get_response(mutation_prompt)
        
        # Return a new Prompt object (importing from local module)
        from gape_core import Prompt
        return Prompt(mutated_text)
    
    def crossover_prompts(self, prompt1, prompt2):
        """Generate offspring prompts by combining elements from two parent prompts."""
        crossover_prompt = f"""
        Combine elements from these two prompts to create two new prompts that might be more effective:
        
        PROMPT 1:
        {prompt1.text}
        
        PROMPT 2:
        {prompt2.text}
        
        Create two new prompts labeled "OFFSPRING 1:" and "OFFSPRING 2:" that combine the strengths of both parent prompts.
        Each offspring should be different and combine different aspects of the parents.
        """
        
        response = self.agent.get_response(crossover_prompt)
        
        # Parse the response to extract the two offspring
        offspring_texts = []
        
        if "OFFSPRING 1:" in response and "OFFSPRING 2:" in response:
            # Split by the offspring markers
            parts = response.split("OFFSPRING 1:")[1].split("OFFSPRING 2:")
            offspring1 = parts[0].strip()
            offspring2 = parts[1].strip()
            offspring_texts = [offspring1, offspring2]
        else:
            # Fallback if format not followed - just split the response in half
            lines = response.strip().split("\n")
            mid = len(lines) // 2
            offspring1 = "\n".join(lines[:mid]).strip()
            offspring2 = "\n".join(lines[mid:]).strip()
            offspring_texts = [offspring1, offspring2]
        
        # Return new Prompt objects
        from gape_core import Prompt
        return [Prompt(text) for text in offspring_texts]
    
    def initialize_population(self, seed_prompt, size=10, strategies=None):
        """Generate an initial population of prompts from a seed prompt."""
        from gape_core import Prompt, Population
        
        # Default strategies for initialization
        strategies = strategies or [
            "instruction", "wording", "structure", "random"
        ]
        
        prompts = [Prompt(seed_prompt)]  # Include the seed prompt
        
        # Generate variations until we reach the desired size
        while len(prompts) < size:
            # Cycle through strategies
            strategy = strategies[(len(prompts) - 1) % len(strategies)]
            
            # Create a variation
            new_prompt = self.mutate_prompt(prompts[0], strategy)
            prompts.append(new_prompt)
        
        # Create and return a population
        return Population(prompts, size=size) 