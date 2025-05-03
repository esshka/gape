# GAPE: Genetic Algorithm for Prompt Evolution

GAPE is a framework for automatically optimizing prompts for Large Language Models (LLMs) using genetic algorithm principles. It leverages Agno agents to handle prompt evolution, evaluation, and selection.

## Overview

The GAPE framework treats prompts as individuals within a population that evolves over generations. It uses LLMs for both generating prompt variations (mutation and crossover) and evaluating prompt fitness based on the quality of the LLM's output.

Key components:
- Population management of prompt candidates
- LLM-driven prompt mutation and crossover
- Automated fitness evaluation of generated outputs
- Evolutionary selection based on fitness scores

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/gape.git
cd gape

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from main import GAPE

# Define your task
task_description = """
The goal is to create a prompt that will make an LLM generate a detailed analysis 
of climate change impacts, including scientific data, economic effects, and policy 
recommendations. The output should be comprehensive, well-structured, and factually accurate.
"""

# Initial seed prompt
seed_prompt = """
Write a detailed analysis of climate change impacts, including scientific data, 
economic effects, and policy recommendations.
"""

# Create and run GAPE
gape = GAPE(
    task_description=task_description,
    seed_prompt=seed_prompt,
    population_size=10,
    generations=5,
    mutation_rate=0.3,
    crossover_rate=0.7
)

# Run the evolutionary process
best_prompt = gape.run()

# The best_prompt object contains the optimized prompt
print(f"Best prompt:\n{best_prompt.text}")
print(f"Fitness score: {best_prompt.fitness}")
```

## Architecture

The framework consists of four main Python modules:

1. `gape_core.py` - Core components (Prompt and Population classes)
2. `fitness_evaluator.py` - Uses Agno agents to evaluate prompt outputs
3. `prompt_mutators.py` - LLM-based prompt mutation and crossover operations
4. `main.py` - The main GAPE implementation and example usage

## Customization

You can customize GAPE for different tasks by:

- Modifying the task description and seed prompt
- Adjusting genetic algorithm parameters (population size, generations, mutation/crossover rates)
- Creating custom mutation strategies in the `MutationEngine` class
- Implementing specialized fitness functions for specific requirements

## Dependencies

- [Agno](https://docs.agno.com/) - For building and managing LLM agents
- [Anthropic](https://www.anthropic.com/) - For Claude models (via Agno)

## Example Applications

- Content generation optimization
- Code generation prompt refinement
- Query-answering optimization
- Creative writing prompt evolution

## License

MIT 