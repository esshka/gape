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

## API Key Setup

GAPE requires an API key for Google's Gemini model. You'll need to:

1. Obtain a Google API key from Google AI Studio (https://makersuite.google.com/)
2. Set the environment variable:

```bash
# On Linux/Mac
export GOOGLE_API_KEY=your_api_key_here

# On Windows (Command Prompt)
set GOOGLE_API_KEY=your_api_key_here

# On Windows (PowerShell)
$env:GOOGLE_API_KEY="your_api_key_here"
```

## Usage

GAPE requires an Agno setup with access to a Google Gemini model. By default, it uses the Gemini 2.0 Flash model.

Basic usage:

```python
from main import GAPE

# Define your task
task_description = """
The goal is to create a prompt that will make an LLM write a creative story
with interesting characters and a surprise ending.
"""

# Create a seed prompt
seed_prompt = """
Write a creative short story with a surprise ending.
Keep it between 300-500 words.
"""

# Create and run GAPE
gape = GAPE(
    task_description=task_description,
    seed_prompt=seed_prompt,
    population_size=10,
    generations=5
)

# Run the evolution process
best_prompt = gape.run()

# The best_prompt object contains the optimized prompt
print(f"Best prompt: {best_prompt.text}")
print(f"Fitness score: {best_prompt.fitness}")
```

## Example Tasks

The `example_tasks.py` file contains several example tasks showing how GAPE can be used for different applications:

```bash
# Run the creative story example
python example_tasks.py creative_story

# Or specify directly in your code
from example_tasks import run_example
results = run_example("scientific_explanation", generations=3, population_size=8)
```

## Customization

You can customize various aspects of GAPE:

- Use different models for different components
- Adjust genetic algorithm parameters
- Implement custom fitness functions
- Define your own mutation strategies

Example:

```python
from agno.models.google import Gemini
from main import GAPE

# Use a specific Google Gemini model
my_model = Gemini(id="gemini-2.0-flash")

gape = GAPE(
    task_description="Your task description",
    seed_prompt="Your seed prompt",
    target_model=my_model,
    mutation_rate=0.4,
    crossover_rate=0.6,
    elitism_count=2
)
```

## How It Works

1. **Initialization**: GAPE starts with a seed prompt and creates initial variations
2. **Evaluation**: Each prompt is evaluated using an LLM to generate output and scoring the result
3. **Selection**: Prompts with higher fitness scores are more likely to be selected for reproduction
4. **Reproduction**: New prompts are created through mutation and crossover operations
5. **Evolution**: The process repeats for multiple generations, with the population improving over time

## License
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.