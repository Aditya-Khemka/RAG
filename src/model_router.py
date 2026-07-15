from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate

from src.chat_model import *

class ModelRouter:
    """Route queries to an appropriate model based on complexity."""

    def __init__(self):
        # Models
        self.cheap_model = get_gpt_mini(temperature=0)
        self.expensive_model = get_gpt_50(temperature=0)

        # Use mini as the classifier to keep costs low
        self.classifier = get_gpt_mini(temperature=0)

        # Stats
        self.stats = {
            "mini_requests": 0,
            "gpt5_requests": 0,
            "total_requests": 0,
            "total_estimated_cost": 0.0,
        }

    def estimate_tokens(self, text: str) -> int:
        """Very rough token estimate."""
        return int(len(text.split()) * 1.3)

    @traceable(name="classify_complexity")
    def classify_complexity(self, query: str) -> str:
        """
        Classify query as simple or complex.
        """

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple:
- Basic facts
- Definitions
- Short answers
- Simple calculations

Complex:
- Analysis
- Reasoning
- Creative writing
- Multi-step tasks
- Architecture/design questions
- Comparisons and tradeoffs

Query:
{query}

Respond with ONLY:
simple
or
complex
"""
        )

        response = self.classifier.invoke(
            prompt.format(query=query)
        )

        result = response.content.strip().lower()

        if result not in {"simple", "complex"}:
            return "complex"

        return result

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Returns:
            (response, model_name, estimated_cost)
        """

        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "gpt-4o-mini"

            # Approx pricing 
            cost_per_1k_input = 0.00015

            self.stats["mini_requests"] += 1
        else:
            model = self.expensive_model
            model_name = "gpt-5"

            cost_per_1k_input = 0.0025

            self.stats["gpt5_requests"] += 1

        response = model.invoke(query)

        input_tokens = self.estimate_tokens(query)
        estimated_cost = (
            input_tokens / 1000
        ) * cost_per_1k_input

        self.stats["total_requests"] += 1
        self.stats["total_estimated_cost"] += estimated_cost

        return (
            response.content,
            model_name,
            estimated_cost,
        )

    def get_stats(self) -> dict:
        total = self.stats["total_requests"]

        return {
            **self.stats,
            "avg_cost_per_request": (
                self.stats["total_estimated_cost"] / max(total, 1)
            ),
        }
    


if __name__ == "__main__":
    router = ModelRouter()

    queries = [
        "What is 2 + 2?",
        "What color is the sky?",
        "Analyze the economic implications of AI on the job market.",
        "Design a scalable RAG architecture for enterprise search.",
    ]

    print("\nModel Routing Demo:\n")

    total_cost = 0

    for query in queries:
        result, model, cost = router.invoke(query)

        total_cost += cost

        print(f"Query: {query}")
        print(f"Model: {model}")
        print(f"Estimated Cost: ${cost:.6f}")
        print(f"Response: {result[:80]}...")
        print("-" * 60)

    print("\nRouter Stats")
    print(router.get_stats())
    print(f"\nTotal Estimated Cost: ${total_cost:.6f}")



'''
Response : 

Model Routing Demo:

Query: What is 2 + 2?
Model: gpt-4o-mini
Estimated Cost: $0.000001
Response: 2 + 2 equals 4....
------------------------------------------------------------
Query: What color is the sky?
Model: gpt-4o-mini
Estimated Cost: $0.000001
Response: The color of the sky can vary depending on several factors, including the time o...
------------------------------------------------------------
Query: Analyze the economic implications of AI on the job market.
Model: gpt-5
Estimated Cost: $0.000032
Response: AI is reshaping the job market through several interacting economic forces rathe...
------------------------------------------------------------
Query: Design a scalable RAG architecture for enterprise search.
Model: gpt-5
Estimated Cost: $0.000025
Response: Below is a reference architecture for a scalable, production-grade RAG (Retrieva...
------------------------------------------------------------

Router Stats
{'mini_requests': 2, 'gpt5_requests': 2, 'total_requests': 4, 'total_estimated_cost': 5.9300000000000005e-05, 'avg_cost_per_request': 1.4825000000000001e-05}

Total Estimated Cost: $0.000059
'''