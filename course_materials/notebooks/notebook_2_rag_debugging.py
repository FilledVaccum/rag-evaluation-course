"""
Notebook 2: RAG Pipeline Debugging - Hands-On Exercises

This notebook provides hands-on exercises for building and debugging RAG pipelines.
Students will learn to identify and fix component-level failures.

Learning Objectives:
- Build an end-to-end RAG pipeline from scratch
- Debug retrieval failures independently
- Debug generation failures independently
- Distinguish between retrieval and generation issues
- Apply systematic debugging workflows

Requirements Coverage: 5.6, 10.2
"""

# %% [markdown]
# # RAG Pipeline Debugging: Hands-On Exercises
# 
# ## Introduction
# 
# In this notebook, you'll build a complete RAG pipeline and learn to debug it
# at each stage. You'll encounter intentional bugs that simulate real-world
# failures and practice systematic debugging techniques.
# 
# **Key Skills You'll Develop:**
# - Component-level debugging
# - Failure diagnosis workflows
# - Retrieval vs. generation failure identification
# - Systematic problem-solving for RAG systems

# %% [markdown]
# ## Setup and Imports

# %%
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# For this exercise, we'll use simplified implementations
# In production, you'd use libraries like LangChain, LlamaIndex, or custom implementations

# Placeholder for vector store (in practice, use Milvus, Pinecone, Chroma, etc.)
class SimpleVectorStore:
    """Simplified vector store for educational purposes."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]):
        """Add documents to the store."""
        self.documents.extend(documents)
        # In practice, you'd generate real embeddings here
        self.embeddings.extend([f"embedding_{i}" for i in range(len(documents))])
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for relevant documents.
        
        INTENTIONAL BUG #1: This always returns the same documents regardless of query!
        This simulates a retrieval failure where the embedding model or search
        algorithm isn't working correctly.
        """
        # BUG: Should use query to find relevant docs, but returns first k docs
        results = []
        for i in range(min(top_k, len(self.documents))):
            results.append((self.documents[i], 0.9 - i * 0.1))
        return results


# Placeholder for LLM (in practice, use NVIDIA NIM, OpenAI, etc.)
class SimpleLLM:
    """Simplified LLM for educational purposes."""
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.
        
        INTENTIONAL BUG #2: This LLM ignores the context and generates
        generic responses! This simulates a generation failure where the
        LLM doesn't use the retrieved context.
        """
        # BUG: Should use context from prompt, but generates generic response
        if "prerequisite" in prompt.lower():
            return "I don't have specific information about prerequisites."
        elif "course" in prompt.lower():
            return "This is a course about computer science topics."
        else:
            return "I don't have enough information to answer that question."


# %%
# Sample course catalog data
COURSE_CATALOG = [
    "CSCI 270: Introduction to Algorithms and Theory of Computing. Prerequisites: CSCI 104.",
    "CSCI 567: Machine Learning. Prerequisites: CSCI 270, MATH 225, and MATH 226.",
    "CSCI 570: Analysis of Algorithms. Prerequisites: CSCI 270.",
    "MATH 225: Linear Algebra and Linear Differential Equations. No prerequisites.",
    "MATH 226: Calculus III. Prerequisites: MATH 125 or MATH 127.",
    "CSCI 104: Data Structures and Object Oriented Design. Prerequisites: CSCI 102.",
]

print("Course catalog loaded:")
for course in COURSE_CATALOG:
    print(f"  - {course[:50]}...")

# %% [markdown]
# ## Exercise 1: Build a Basic RAG Pipeline
# 
# Your first task is to build a simple RAG pipeline with three stages:
# 1. Retrieval: Search vector store for relevant documents
# 2. Augmentation: Format context into a prompt
# 3. Generation: Generate response using LLM

# %%
class RAGPipeline:
    """Simple RAG pipeline for debugging exercises."""
    
    def __init__(self, vector_store: SimpleVectorStore, llm: SimpleLLM):
        self.vector_store = vector_store
        self.llm = llm
        self.last_retrieval_results = None
        self.last_augmented_prompt = None
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Stage 1: Retrieval"""
        start_time = time.time()
        results = self.vector_store.search(query, top_k)
        retrieval_time = (time.time() - start_time) * 1000
        
        self.last_retrieval_results = results
        print(f"[Retrieval] Found {len(results)} documents in {retrieval_time:.2f}ms")
        return results
    
    def augment(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """Stage 2: Augmentation"""
        # Format contexts into prompt
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, (ctx, score) in enumerate(contexts)])
        
        prompt = f"""You are a helpful assistant answering questions about course prerequisites.

Retrieved Context:
{context_str}

Question: {query}

Answer the question using ONLY the information provided in the context above. If the answer is not in the context, say so.

Answer:"""
        
        self.last_augmented_prompt = prompt
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Stage 3: Generation"""
        start_time = time.time()
        response = self.llm.generate(prompt)
        generation_time = (time.time() - start_time) * 1000
        
        print(f"[Generation] Generated response in {generation_time:.2f}ms")
        return response
    
    def process_query(self, query: str) -> str:
        """End-to-end pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")
        
        # Stage 1: Retrieval
        contexts = self.retrieve(query)
        
        # Stage 2: Augmentation
        prompt = self.augment(query, contexts)
        
        # Stage 3: Generation
        response = self.generate(prompt)
        
        return response


# %%
# Initialize components
vector_store = SimpleVectorStore()
vector_store.add_documents(COURSE_CATALOG)

llm = SimpleLLM()

pipeline = RAGPipeline(vector_store, llm)

# Test the pipeline
query = "What are the prerequisites for CSCI 567?"
response = pipeline.process_query(query)

print(f"\nFinal Response: {response}")

# %% [markdown]
# ## Exercise 2: Debug Retrieval Failures
# 
# **Problem**: The pipeline isn't returning correct answers. Let's debug!
# 
# **Your Task**: 
# 1. Inspect the retrieved documents
# 2. Check if the answer is in the retrieved context
# 3. Identify the retrieval failure
# 4. Fix the bug in SimpleVectorStore.search()

# %%
def debug_retrieval(pipeline: RAGPipeline, query: str):
    """
    Debug retrieval stage independently.
    
    This function helps you inspect what the retrieval stage is returning
    and whether it contains the information needed to answer the query.
    """
    print(f"\n{'='*60}")
    print(f"RETRIEVAL DEBUG: {query}")
    print(f"{'='*60}")
    
    # Run retrieval
    contexts = pipeline.retrieve(query)
    
    # Inspect results
    print(f"\nRetrieved {len(contexts)} documents:")
    for i, (doc, score) in enumerate(contexts):
        print(f"\n  Document {i+1} (score: {score:.2f}):")
        print(f"    {doc}")
    
    # Manual inspection questions
    print(f"\n{'='*60}")
    print("DIAGNOSTIC QUESTIONS:")
    print(f"{'='*60}")
    print("1. Do the retrieved documents contain the answer to the query?")
    print("2. Are the most relevant documents ranked highest?")
    print("3. Is the query semantically similar to the retrieved documents?")
    print("4. Are there better documents in the catalog that weren't retrieved?")
    
    return contexts


# %%
# Debug the retrieval stage
query = "What are the prerequisites for CSCI 567?"
contexts = debug_retrieval(pipeline, query)

# %% [markdown]
# ### Your Analysis:
# 
# **Question**: What did you observe about the retrieved documents?
# 
# **Answer**: (Write your observations here)
# - Are the retrieved documents relevant to CSCI 567?
# - Does the retrieval change when you use different queries?
# - What's the bug in the SimpleVectorStore.search() method?
# 
# **Hint**: Try running debug_retrieval with different queries and see if the
# retrieved documents change.

# %%
# Test with different queries
debug_retrieval(pipeline, "What are the prerequisites for CSCI 270?")
debug_retrieval(pipeline, "Tell me about MATH 225")

# %% [markdown]
# ### Fix the Retrieval Bug
# 
# **Your Task**: Modify the SimpleVectorStore.search() method to actually
# search for relevant documents based on the query.
# 
# **Hint**: For this simplified example, you can use keyword matching.
# In production, you'd use real embeddings and vector similarity.

# %%
# TODO: Fix the SimpleVectorStore.search() method
# 
# Here's a corrected version:

class FixedVectorStore:
    """Fixed vector store with proper search."""
    
    def __init__(self):
        self.documents = []
    
    def add_documents(self, documents: List[str]):
        self.documents.extend(documents)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant documents using keyword matching."""
        # Extract key terms from query
        query_lower = query.lower()
        
        # Score each document
        scored_docs = []
        for doc in self.documents:
            doc_lower = doc.lower()
            
            # Simple scoring: count matching words
            score = 0.0
            for word in query_lower.split():
                if len(word) > 3 and word in doc_lower:  # Skip short words
                    score += 1.0
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


# Test the fixed version
fixed_vector_store = FixedVectorStore()
fixed_vector_store.add_documents(COURSE_CATALOG)

fixed_pipeline = RAGPipeline(fixed_vector_store, llm)

print("Testing fixed retrieval:")
contexts = debug_retrieval(fixed_pipeline, "What are the prerequisites for CSCI 567?")

# %% [markdown]
# ## Exercise 3: Debug Generation Failures
# 
# **Problem**: Even with correct retrieval, the LLM isn't using the context!
# 
# **Your Task**:
# 1. Inspect the augmented prompt
# 2. Check if the LLM is using the provided context
# 3. Identify the generation failure
# 4. Fix the bug in SimpleLLM.generate()

# %%
def debug_generation(pipeline: RAGPipeline, query: str, contexts: List[Tuple[str, float]]):
    """
    Debug generation stage independently.
    
    This function helps you inspect the augmented prompt and the generated
    response to identify if the LLM is using the context correctly.
    """
    print(f"\n{'='*60}")
    print(f"GENERATION DEBUG: {query}")
    print(f"{'='*60}")
    
    # Create augmented prompt
    prompt = pipeline.augment(query, contexts)
    
    print("\nAugmented Prompt:")
    print(f"{'='*60}")
    print(prompt)
    print(f"{'='*60}")
    
    # Generate response
    response = pipeline.generate(prompt)
    
    print(f"\nGenerated Response:")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}")
    
    # Manual inspection questions
    print(f"\n{'='*60}")
    print("DIAGNOSTIC QUESTIONS:")
    print(f"{'='*60}")
    print("1. Does the response use information from the context?")
    print("2. Are there claims in the response not supported by context?")
    print("3. Does the response actually answer the question?")
    print("4. Is the LLM ignoring the context and using parametric knowledge?")
    
    return response


# %%
# Debug the generation stage
query = "What are the prerequisites for CSCI 567?"
response = debug_generation(fixed_pipeline, query, contexts)

# %% [markdown]
# ### Your Analysis:
# 
# **Question**: What did you observe about the generated response?
# 
# **Answer**: (Write your observations here)
# - Does the response mention CSCI 270, MATH 225, and MATH 226?
# - Is the LLM using the context or generating generic responses?
# - What's the bug in the SimpleLLM.generate() method?

# %% [markdown]
# ### Fix the Generation Bug
# 
# **Your Task**: Modify the SimpleLLM.generate() method to actually use
# the context provided in the prompt.
# 
# **Hint**: For this simplified example, you can extract information from
# the prompt. In production, you'd use a real LLM API.

# %%
# TODO: Fix the SimpleLLM.generate() method
#
# Here's a corrected version:

class FixedLLM:
    """Fixed LLM that uses context."""
    
    def generate(self, prompt: str) -> str:
        """Generate response using context from prompt."""
        # Extract context and question from prompt
        if "Context" in prompt and "Question:" in prompt:
            # Parse the prompt
            context_section = prompt.split("Question:")[0]
            question = prompt.split("Question:")[1].split("Answer:")[0].strip()
            
            # Simple extraction: find prerequisites in context
            if "prerequisite" in question.lower():
                # Look for "Prerequisites:" in context
                for line in context_section.split("\n"):
                    if "Prerequisites:" in line:
                        # Extract the prerequisites part
                        prereqs = line.split("Prerequisites:")[1].strip()
                        # Extract course code from question
                        words = question.split()
                        for word in words:
                            if word.startswith("CSCI") or word.startswith("MATH"):
                                course_code = word.rstrip("?.,")
                                if course_code in line:
                                    return f"The prerequisites for {course_code} are: {prereqs}"
                
                return "I found prerequisite information in the context, but couldn't match it to the specific course."
            
            # Generic response using context
            return "Based on the provided context, I can see course information, but I need more specific guidance to answer this question."
        
        return "I don't have enough context to answer this question."


# Test the fixed version
fixed_llm = FixedLLM()
fully_fixed_pipeline = RAGPipeline(fixed_vector_store, fixed_llm)

print("Testing fixed generation:")
response = fully_fixed_pipeline.process_query("What are the prerequisites for CSCI 567?")
print(f"\nFinal Response: {response}")

# %% [markdown]
# ## Exercise 4: End-to-End Debugging Workflow
# 
# **Your Task**: Apply the complete debugging workflow to diagnose and fix
# issues in a RAG pipeline.
# 
# **Workflow**:
# 1. Run the query through the pipeline
# 2. If the answer is wrong, debug retrieval first
# 3. If retrieval is correct, debug generation
# 4. Fix the identified issue
# 5. Verify the fix works

# %%
def complete_debugging_workflow(pipeline: RAGPipeline, query: str, expected_answer: str):
    """
    Complete debugging workflow for RAG pipeline.
    
    Args:
        pipeline: RAG pipeline to debug
        query: Test query
        expected_answer: What the correct answer should contain
    """
    print(f"\n{'='*70}")
    print(f"COMPLETE DEBUGGING WORKFLOW")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Expected answer should contain: {expected_answer}")
    
    # Step 1: Run end-to-end
    print(f"\n{'='*70}")
    print("STEP 1: Run End-to-End Pipeline")
    print(f"{'='*70}")
    response = pipeline.process_query(query)
    print(f"\nResponse: {response}")
    
    # Check if answer is correct
    is_correct = expected_answer.lower() in response.lower()
    print(f"\nAnswer correct? {is_correct}")
    
    if is_correct:
        print("\n✓ Pipeline working correctly!")
        return
    
    # Step 2: Debug retrieval
    print(f"\n{'='*70}")
    print("STEP 2: Debug Retrieval Stage")
    print(f"{'='*70}")
    contexts = debug_retrieval(pipeline, query)
    
    # Check if answer is in retrieved contexts
    answer_in_context = any(expected_answer.lower() in ctx[0].lower() for ctx in contexts)
    print(f"\nAnswer in retrieved context? {answer_in_context}")
    
    if not answer_in_context:
        print("\n✗ RETRIEVAL FAILURE: Answer not in retrieved documents")
        print("  → Fix: Improve embeddings, chunking, or search algorithm")
        return
    
    # Step 3: Debug generation
    print(f"\n{'='*70}")
    print("STEP 3: Debug Generation Stage")
    print(f"{'='*70}")
    response = debug_generation(pipeline, query, contexts)
    
    print(f"\n✗ GENERATION FAILURE: Answer in context but not in response")
    print("  → Fix: Improve prompts, use better LLM, or add explicit instructions")


# %%
# Test the workflow with the broken pipeline
broken_pipeline = RAGPipeline(SimpleVectorStore(), SimpleLLM())
broken_pipeline.vector_store.add_documents(COURSE_CATALOG)

complete_debugging_workflow(
    broken_pipeline,
    "What are the prerequisites for CSCI 567?",
    "CSCI 270"
)

# %%
# Test the workflow with the fixed pipeline
complete_debugging_workflow(
    fully_fixed_pipeline,
    "What are the prerequisites for CSCI 567?",
    "CSCI 270"
)

# %% [markdown]
# ## Exercise 5: Practice Debugging (Your Turn!)
# 
# **Your Task**: Debug these queries and identify whether the failure is in
# retrieval or generation.
# 
# For each query:
# 1. Run the complete debugging workflow
# 2. Identify the failure type (retrieval or generation)
# 3. Propose a fix

# %%
# Test queries
test_queries = [
    ("What are the prerequisites for CSCI 270?", "CSCI 104"),
    ("Does MATH 225 have any prerequisites?", "No prerequisites"),
    ("What course requires MATH 226?", "CSCI 567"),
]

for query, expected in test_queries:
    complete_debugging_workflow(fully_fixed_pipeline, query, expected)
    print("\n" + "="*70 + "\n")

# %% [markdown]
# ## Key Takeaways
# 
# **Component-Level Debugging is Essential**:
# - Always debug retrieval and generation independently
# - Don't assume the LLM is the problem
# - 80% of RAG failures are retrieval issues
# 
# **Systematic Workflow**:
# 1. Run end-to-end first
# 2. If wrong, check retrieval
# 3. If retrieval correct, check generation
# 4. Fix the identified issue
# 
# **Common Failure Patterns**:
# - Retrieval: Wrong docs, poor ranking, semantic mismatch
# - Generation: Context ignorance, hallucinations, partial answers
# 
# **Next Steps**:
# - Module 4: Learn to generate synthetic test data
# - Module 5: Master evaluation metrics (Ragas framework)
# - Module 6: Evaluate legacy semantic search systems

# %% [markdown]
# ## Additional Challenges (Optional)
# 
# 1. **Improve the vector store**: Implement real embeddings using NVIDIA NIM
# 2. **Add re-ranking**: Implement a re-ranking stage after retrieval
# 3. **Measure metrics**: Calculate precision, recall, and faithfulness
# 4. **Handle edge cases**: What happens with empty queries or no results?
# 5. **Optimize performance**: Measure and improve retrieval/generation time

print("\n" + "="*70)
print("Notebook 2: RAG Debugging - Complete!")
print("="*70)
