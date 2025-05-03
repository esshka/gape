"""
Core components for the Genetic Algorithm for Prompt Evolution (GAPE) framework.
This module contains the fundamental classes for prompt representation and population management.
"""
import random
from copy import deepcopy


class Prompt:
    """Represents an individual prompt in the genetic algorithm."""
    def __init__(self, text, fitness=None):
        self.text = text
        self.fitness = fitness
    
    def __repr__(self):
        return f"Prompt(fitness={self.fitness}, text='{self.text[:50]}...')"


class Population:
    """Manages a population of prompts."""
    def __init__(self, prompts=None, size=10):
        self.prompts = prompts or []
        self.size = size
    
    def initialize(self, seed_prompt, size=None):
        """Initialize population from a seed prompt."""
        # This will be implemented using the mutation engine
        pass
    
    def select_parents(self, selection_method="tournament", tournament_size=3):
        """Select parents for reproduction based on fitness."""
        if not self.prompts:
            return []
        
        # Sort prompts by fitness (descending)
        sorted_prompts = sorted(self.prompts, key=lambda p: p.fitness if p.fitness is not None else -float('inf'), reverse=True)
        
        if selection_method == "tournament":
            # Tournament selection
            selected = []
            for _ in range(2):  # Select two parents
                tournament = random.sample(self.prompts, min(tournament_size, len(self.prompts)))
                winner = max(tournament, key=lambda p: p.fitness if p.fitness is not None else -float('inf'))
                selected.append(winner)
            return selected
        
        elif selection_method == "roulette":
            # Roulette wheel selection
            total_fitness = sum(p.fitness for p in self.prompts if p.fitness is not None)
            if total_fitness <= 0:
                return random.sample(self.prompts, 2)  # Random if all fitness is zero
                
            # Select two parents
            selected = []
            for _ in range(2):
                # Spin the wheel
                pick = random.uniform(0, total_fitness)
                current = 0
                for prompt in self.prompts:
                    if prompt.fitness is not None:
                        current += prompt.fitness
                        if current >= pick:
                            selected.append(prompt)
                            break
                
                # Fallback if no selection (shouldn't happen, but just in case)
                if len(selected) <= _:
                    selected.append(random.choice(self.prompts))
                    
            return selected
        
        elif selection_method == "rank":
            # Rank-based selection
            # Assign selection probability based on rank, not raw fitness
            ranks = list(range(1, len(sorted_prompts) + 1))
            rank_sum = sum(ranks)
            # Select two parents with probability proportional to rank
            selected = []
            for _ in range(2):
                pick = random.uniform(0, rank_sum)
                current = 0
                for i, prompt in enumerate(sorted_prompts):
                    current += ranks[i]
                    if current >= pick:
                        selected.append(prompt)
                        break
            return selected
        
        else:
            # Default: select the top two
            return sorted_prompts[:2]
    
    def evolve(self, mutation_engine, crossover_rate=0.7, mutation_rate=0.3, elitism_count=1, selection_method="tournament"):
        """Evolve the population to the next generation."""
        if not self.prompts:
            return
        
        # Sort the current population by fitness (descending)
        sorted_prompts = sorted(self.prompts, key=lambda p: p.fitness if p.fitness is not None else -float('inf'), reverse=True)
        
        # Apply elitism - carry over the best individuals unchanged
        new_population = sorted_prompts[:elitism_count].copy()
        
        # Keep creating new individuals until we fill the population
        while len(new_population) < self.size:
            # Select parents
            parent1, parent2 = self.select_parents(selection_method=selection_method)
            
            # Apply crossover with probability crossover_rate
            if random.random() < crossover_rate and parent1 != parent2:
                offspring = mutation_engine.crossover_prompts(parent1, parent2)
            else:
                # No crossover, create copies
                offspring = [deepcopy(parent1), deepcopy(parent2)]
            
            # Apply mutation with probability mutation_rate
            for child in offspring:
                if random.random() < mutation_rate:
                    child = mutation_engine.mutate_prompt(child)
                
                # Add to new population if there's space
                if len(new_population) < self.size:
                    new_population.append(child)
                else:
                    break
        
        # Update the population
        self.prompts = new_population
    
    def get_best_prompt(self):
        """Return the prompt with the highest fitness."""
        return max(self.prompts, key=lambda p: p.fitness if p.fitness is not None else -float('inf')) 