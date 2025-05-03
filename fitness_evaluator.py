"""
Fitness evaluation module for the GAPE framework.
Uses Agno agents to evaluate the quality of prompts based on their generated outputs.
"""
from agno.agent import Agent
from agno.models.google import Gemini


class FitnessEvaluator:
    """Evaluates the fitness of prompts using an Agno agent."""
    
    def __init__(self, target_model=None, evaluation_model=None, task_description=None):
        self.task_description = task_description
        
        # Target model that will execute the prompts
        self.target_model = target_model or Gemini(id="gemini-2.0-flash")
        self.target_agent = Agent(model=self.target_model)
        
        # Evaluation model that will score the outputs
        self.evaluation_model = evaluation_model or Gemini(id="gemini-2.0-flash")
        self.evaluation_agent = Agent(
            model=self.evaluation_model,
           
            instructions=[
                "Your task is to evaluate the quality of an output based on specific criteria.",
                "Rate the output on a scale of 0-100 where 100 is perfect.",
                "Provide a numerical score only."
            ]
        )
    
    def evaluate_prompt(self, prompt):
        """Evaluate a single prompt and return its fitness score."""
        # Generate output using the target model
        output = self.target_agent.run(prompt.text).content
        
        # Evaluate the output using the evaluation model
        eval_prompt = f"""
        Task Description: {self.task_description}
        
        Generated Output:
        {output}
        
        Evaluate how well this output fulfills the task described.
        Score from 0-100:
        """
        
        score = self.evaluation_agent.run(eval_prompt).content
        
        # Extract numerical score (handling potential non-numeric responses)
        try:
            score = float(score.strip())
        except ValueError:
            # If the model didn't return just a number, try to extract it
            import re
            match = re.search(r'(\d+)', score)
            if match:
                score = float(match.group(1))
            else:
                score = 0  # Default if no numeric score found
        
        # Normalize score to 0-1 range
        prompt.fitness = score / 100.0
        return prompt.fitness
    
    def evaluate_population(self, population):
        """Evaluate all prompts in a population."""
        for prompt in population.prompts:
            self.evaluate_prompt(prompt)
        return population 