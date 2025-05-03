"""
Example tasks for the GAPE framework.

This file contains various task descriptions and seed prompts for different domains
to demonstrate how GAPE can be used across different applications.
"""
from main import GAPE
from agno.agent import Agent
from agno.models.google import Gemini


# Example task definitions, each with task_description and seed_prompt
EXAMPLE_TASKS = {
    "creative_story": {
        "task_description": """
        The goal is to create a prompt that will make an LLM generate a creative, engaging short story
        that is original, has interesting characters, and a surprising twist ending.
        The story should be appropriate for all ages and between 300-500 words.
        """,
        "seed_prompt": """
        Write a creative short story with interesting characters and a surprise ending.
        Keep it between 300-500 words.
        """
    },
    
    "code_generation": {
        "task_description": """
        The goal is to create a prompt that will make an LLM generate high-quality, efficient, 
        and readable Python code for a function that performs binary search. The code should be 
        correct, well-commented, handle edge cases, and follow best practices.
        """,
        "seed_prompt": """
        Write a Python function that implements the binary search algorithm. Include comments and handle edge cases.
        """
    },
    
    "scientific_explanation": {
        "task_description": """
        The goal is to create a prompt that will make an LLM generate a clear, accurate, and 
        educational explanation of how nuclear fusion works. The explanation should be accessible 
        to high school students but scientifically accurate, include key concepts, and use helpful analogies.
        """,
        "seed_prompt": """
        Explain how nuclear fusion works in a way that high school students can understand.
        """
    },
    
    "product_description": {
        "task_description": """
        The goal is to create a prompt that will make an LLM generate a compelling product description
        for a new smartphone. The description should highlight key features, use persuasive language,
        address potential customer needs, and differentiate the product from competitors.
        """,
        "seed_prompt": """
        Write a product description for a new smartphone that will appeal to potential customers.
        """
    },
    
    "travel_itinerary": {
        "task_description": """
        The goal is to create a prompt that will make an LLM generate a detailed 3-day travel itinerary
        for Tokyo, Japan. The itinerary should include realistic timings, a mix of popular and lesser-known
        attractions, food recommendations, transportation information, and practical tips.
        """,
        "seed_prompt": """
        Create a 3-day travel itinerary for visiting Tokyo, Japan.
        """
    }
}


def run_example(task_key, generations=3, population_size=8, verbose=True):
    """Run GAPE on a specific example task."""
    if task_key not in EXAMPLE_TASKS:
        raise ValueError(f"Unknown task key: {task_key}. Available tasks: {list(EXAMPLE_TASKS.keys())}")
    
    task = EXAMPLE_TASKS[task_key]
    
    if verbose:
        print(f"Running GAPE for task: {task_key}")
        print("-" * 80)
        print(f"Task description: {task['task_description'].strip()}")
        print(f"Seed prompt: {task['seed_prompt'].strip()}")
        print("-" * 80)
    
    # Run GAPE
    gape = GAPE(
        task_description=task["task_description"],
        seed_prompt=task["seed_prompt"],
        population_size=population_size,
        generations=generations,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=1
    )
    
    best_prompt = gape.run()
    
    if verbose:
        # Show the output generated with the best prompt
        print("\nGenerating output with the best prompt...")
        model = Gemini(id="gemini-2.0-flash")
        agent = Agent(model=model)
        output = agent.chat(best_prompt.text)
        
        print("\nFINAL OUTPUT GENERATED WITH THE BEST PROMPT:")
        print("=" * 80)
        print(output)
        print("=" * 80)
    
    return {
        "task": task_key,
        "best_prompt": best_prompt,
        "history": gape.history
    }


def main():
    """Run all example tasks or a specific one."""
    import sys
    
    # If a task key is provided as argument, run only that task
    if len(sys.argv) > 1 and sys.argv[1] in EXAMPLE_TASKS:
        task_key = sys.argv[1]
        run_example(task_key)
    else:
        # Run a single example for demonstration
        run_example("creative_story", generations=2, population_size=6)
        
        # Uncomment to run all examples:
        # for task_key in EXAMPLE_TASKS:
        #     run_example(task_key, generations=2, population_size=6)


if __name__ == "__main__":
    main() 