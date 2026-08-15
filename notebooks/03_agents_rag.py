"""
Research Notebook 3: LLM Agents & RAG Pipeline
Demonstrates the video analysis agent with tool-calling and RAG.

Run: python notebooks/03_agents_rag.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.video_agent import VideoAnalysisAgent
from src.agents.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("Notebook 3: LLM Agents & RAG Pipeline")
    print("=" * 60)
    
    # 1. Video Analysis Agent
    print("\n--- Video Analysis Agent ---")
    agent = VideoAnalysisAgent(model_name="gpt-4", temperature=0.3)
    
    # Task planning
    tasks = [
        "Detect all scene boundaries in the video",
        "Extract keyframes from the video",
        "Summarize the video content",
        "What action is happening in the video?",
    ]
    
    for task in tasks:
        plan = agent.plan(task)
        print(f"  Task: '{task}'")
        print(f"  Plan: {plan}\n")
    
    # Execute a task
    print("--- Agent Execution ---")
    response = agent.execute_task("Summarize the video scenes", video_path="sample.mp4")
    print(f"  Task: {response.task}")
    print(f"  Tools used: {response.tool_used}")
    print(f"  Confidence: {response.confidence}")
    print(f"  Result keys: {list(response.result.keys())}")
    
    # 2. RAG Pipeline
    print("\n--- RAG Pipeline ---")
    rag = RAGPipeline()
    
    # Index video content descriptions
    corpus = [
        "Scene 1: A chef is preparing Italian pasta in a modern kitchen with fresh basil and tomatoes.",
        "Scene 2: The camera transitions to an outdoor market where vendors are selling fresh produce.",
        "Scene 3: A group of friends is enjoying the pasta dish at a dinner table with wine.",
        "Scene 4: A tutorial segment showing how to properly cook al dente pasta with timing tips.",
        "Scene 5: The chef explains the importance of using fresh ingredients for authentic Italian cooking.",
        "Scene 6: Close-up shots of the finished dish with garnish and olive oil drizzle.",
    ]
    
    print("Indexing video content descriptions...")
    rag.index_documents(corpus)
    print(f"  Indexed {len(corpus)} documents")
    print(f"  Embedding dimension: {rag.embedder.dim}")
    
    # Query the RAG pipeline
    queries = [
        "What happens in the kitchen scene?",
        "How do you cook pasta properly?",
        "What is the final dish presentation?",
    ]
    
    print("\n--- RAG Queries ---")
    for query in queries:
        result = rag.generate(query, top_k=2)
        print(f"\n  Query: {query}")
        print(f"  Answer: {result['answer']}")
        print(f"  Retrieved: {len(result['retrieved'])} docs")
        for r in result['retrieved']:
            print(f"    [{r['rank']}] score={r['score']:.3f}: {r['text'][:80]}...")
    
    # 3. Agent with RAG context
    print("\n--- Agent + RAG Integration ---")
    agent.add_context(corpus)
    context = agent.retrieve_context("What ingredients are used?", top_k=2)
    print(f"  Retrieved {len(context)} context items")
    for item in context:
        print(f"    {item['text'][:80]}...")
    
    answer = agent.answer_question("What ingredients are used?", video_context={})
    print(f"  Answer: {answer}")
    
    print("\n✓ Agent + RAG integration verified")
    print("Key: LLM agent orchestrates tools, RAG provides content retrieval")
    print("Combined: autonomous video understanding with grounded answers")


if __name__ == "__main__":
    main()
