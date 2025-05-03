"""
Main implementation of the Genetic Algorithm for Prompt Evolution (GAPE) framework
using Agno agents for prompt evolution and evaluation.
"""
import os
import time
from agno.models.google import Gemini

# Import our GAPE components
from gape_core import Prompt, Population
from fitness_evaluator import FitnessEvaluator
from prompt_mutators import MutationEngine


class GAPE:
    """
    Genetic Algorithm for Prompt Evolution (GAPE) system.
    
    This class orchestrates the entire GAPE process, managing the evolution
    of prompts over multiple generations to optimize for a specific task.
    """
    
    def __init__(
        self,
        task_description,
        seed_prompt,
        population_size=10,
        generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=1,
        selection_method="tournament",
        target_model=None,
        evaluation_model=None,
        mutation_model=None
    ):
        self.task_description = task_description
        self.seed_prompt = seed_prompt
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        
        # Initialize models (default to Gemini)
        self.target_model = target_model or Gemini(id="gemini-2.0-flash")
        self.evaluation_model = evaluation_model or self.target_model
        self.mutation_model = mutation_model or self.target_model
        
        # Initialize components
        self.mutation_engine = MutationEngine(model=self.mutation_model)
        self.fitness_evaluator = FitnessEvaluator(
            target_model=self.target_model,
            evaluation_model=self.evaluation_model,
            task_description=self.task_description
        )
        
        # Initialize population
        self.population = None
        self.generation = 0
        self.best_prompt = None
        self.history = []  # Track evolution history
    
    def initialize(self):
        """Initialize the population from the seed prompt."""
        print(f"Initializing population with {self.population_size} prompts...")
        self.population = self.mutation_engine.initialize_population(
            self.seed_prompt, 
            size=self.population_size
        )
        self.generation = 0
    
    def evaluate(self):
        """Evaluate the fitness of all prompts in the current population."""
        print(f"Evaluating population fitness (Generation {self.generation})...")
        self.fitness_evaluator.evaluate_population(self.population)
        
        # Update best prompt if found better one
        best_in_gen = self.population.get_best_prompt()
        if self.best_prompt is None or best_in_gen.fitness > self.best_prompt.fitness:
            self.best_prompt = best_in_gen
        
        # Record this generation's stats
        avg_fitness = sum(p.fitness for p in self.population.prompts if p.fitness is not None) / len(self.population.prompts)
        self.history.append({
            'generation': self.generation,
            'best_fitness': best_in_gen.fitness,
            'avg_fitness': avg_fitness,
            'best_prompt': best_in_gen.text
        })
        
        # Print generation summary
        print(f"Generation {self.generation} stats:")
        print(f"  Best fitness: {best_in_gen.fitness:.4f}")
        print(f"  Average fitness: {avg_fitness:.4f}")
    
    def evolve(self):
        """Evolve the population to the next generation."""
        print(f"Evolving to generation {self.generation + 1}...")
        self.population.evolve(
            mutation_engine=self.mutation_engine,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            elitism_count=self.elitism_count,
            selection_method=self.selection_method
        )
        self.generation += 1
    
    def run(self):
        """Run the complete GAPE process."""
        start_time = time.time()
        
        # Initialize if not already done
        if self.population is None:
            self.initialize()
        
        # Run the evaluation for the initial population
        self.evaluate()
        
        # Main evolution loop
        for _ in range(self.generations):
            # Evolve to the next generation
            self.evolve()
            
            # Evaluate the new generation
            self.evaluate()
            
            # Check if we should terminate early (e.g., reached perfect fitness)
            if self.best_prompt.fitness >= 0.99:
                print("Reached optimal fitness. Terminating early.")
                break
        
        # Report final results
        elapsed_time = time.time() - start_time
        print("\nGAPE completed in {:.2f} seconds".format(elapsed_time))
        print(f"Ran for {self.generation} generations")
        print(f"Best prompt found (fitness: {self.best_prompt.fitness:.4f}):")
        print(f"\n{self.best_prompt.text}\n")
        
        return self.best_prompt


def main():
    """Example usage of the GAPE system."""
    # Example task: generating a creative story
    task_description = """
    The goal is to create a prompt that will make an LLM generate a creative, engaging short story
    that is original, has interesting characters, and a surprising twist ending.
    The story should be appropriate for all ages and between 300-500 words.
    """
    
    # Seed prompt
    seed_prompt = """
    Write a creative short story with interesting characters and a surprise ending.
    Keep it between 300-500 words.
    """
    
    # Create and run the GAPE system
    gape = GAPE(
        task_description=task_description,
        seed_prompt=seed_prompt,
        population_size=8,
        generations=3,
        mutation_rate=0.4,
        crossover_rate=0.7,
        elitism_count=1
    )
    
    best_prompt = gape.run()
    
    # Demonstrate the best prompt in action
    print("Generating a final story using the best prompt...")
    from agno.agent import Agent
    final_agent = Agent(model=Gemini(id="gemini-2.0-flash"))
    final_story = final_agent.run(best_prompt.text).content
    
    print("\nFINAL STORY GENERATED WITH THE BEST PROMPT:")
    print("=" * 80)
    print(final_story)
    print("=" * 80)


if __name__ == "__main__":
    main() 