# Add the project root (RAG/) to Python's path.
from pathlib import Path
import sys
from langsmith import traceable
project_root = Path(__file__).resolve().parent.parent.parent    
sys.path.insert(0, str(project_root))


from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings import get_embedding_model

@traceable(name="semantic_split_text")
def semantic_split_text(text: str,threshold_type: str = "percentile",threshold_amount: int = 90):
    """
    Split raw text into semantic chunks.

    Parameters
    ----------
    text : str
        Text to split.

    threshold_type : str
        percentile
        standard_deviation
        interquartile
        gradient

    threshold_amount : int | float
        Controls aggressiveness of splitting.
    """

    embeddings = get_embedding_model()

    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )

    chunks = semantic_chunker.split_text(text)

    return chunks


if __name__ == "__main__":
    text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Topic 1: Renewable Energy and Electric Vehicles
Renewable energy adoption has accelerated over the past decade as the cost of solar panels and wind turbines has declined. Governments and utilities increasingly invest in large-scale renewable projects to reduce carbon emissions and improve energy security. However, integrating intermittent energy sources into the electrical grid requires significant investments in storage and transmission infrastructure.
Battery technology plays a central role in this transition. Advances in lithium-ion chemistry have improved energy density and reduced manufacturing costs, making battery storage systems economically viable for both homes and utility-scale installations.

## Topic 2: Marine Ecosystems
Coral reefs support a remarkable diversity of marine life despite occupying only a small fraction of the ocean floor. These ecosystems provide shelter, breeding grounds, and food sources for thousands of species. Scientists often use reef health as an indicator of broader changes occurring in the oceans.
Rising ocean temperatures have increased the frequency of coral bleaching events, in which corals expel the algae living within their tissues. Researchers increasingly rely on underwater drones and satellite imagery to monitor reef conditions and better understand long-term ecological changes.

## Topic 3: Electric Vehicles and Transportation Systems
Electric vehicles have become an important component of modern transportation policy. Many cities encourage EV adoption through charging infrastructure, tax incentives, and preferential parking policies. The same battery technologies that support renewable energy storage are also driving improvements in vehicle range and performance.
Transportation researchers argue that electrifying cars alone will not solve urban congestion problems. Public transit, cycling infrastructure, and walkable neighborhoods remain essential for creating efficient and sustainable cities.

"""

    chunks = semantic_split_text(
        text,
        threshold_type="gradient",
        threshold_amount=75
    )

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)