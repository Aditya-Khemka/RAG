from csv_loader import * 
from embeddings import *
from db.chroma_db import *

def upsert_articles_to_chroma(articles):
    """Upsert articles into the Chroma collection with embeddings and metadata."""
    
    # Create a new collection for articles in the Chroma database
    collection = chroma_client.get_or_create_collection(name="articles_collection")

    # Upsert articles into the collection
    for article in articles.iter_rows(named=True):
        collection.upsert(
            ids=[str(article["id"])],
            documents=[article["Article"]],
            embeddings=[embed_text(article["Article"])],
            metadatas=[
                {
                    "Heading": article["Heading"],
                    "Date": article["Date"],
                }
            ],
        )


if __name__ == "__main__":

    # Step 0 : 
    # Upsert articles into the Chroma collection
    upsert_articles_to_chroma(articles)

    # Step 1 : the question 
    query = "Public transport fares by 7 per cent, what is the reason for this increase?"

    # Step 2 : Embedding the query using the same embedding model used for the articles
    query_embedding = embed_text(query)

    # Step 3 : Query the Chroma collection for the most similar articles based on the query embedding
    collection = chroma_client.get_collection(name="articles_collection")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    # Step 4 : Display the results
    print(f"Query: {query}")
    print("Top 3 most similar articles:")
    for i, article in enumerate(results["documents"][0]):
        print(f"{i+1}. {article}")