# evaluation.py

from .retrieval import PolicyRetriever
from .rag_chain import PolicyRAG


RETRIEVAL_TESTS = [
    "What is the policy for requesting time off?",
    "What is the total number of paid time off leaves for paternity leave?",
    "What are the basic security protocols for laptop usage?",
    "Who is Brad Pitt?",
]


GENERATION_TESTS = [
    "What is the policy for sexual harassment?",
    "Is remote work allowed? If so, what are the guidelines?",
    "What is the total working hours per week?",
    "What is the total number of paid time off leaves for maternity leave?",
]


def run_retrieval_tests():
    """
    Runs retrieval-only evaluation.
    """

    retriever = PolicyRetriever()

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for idx, question in enumerate(RETRIEVAL_TESTS, start=1):

        print(f"\n[{idx}/{len(RETRIEVAL_TESTS)}]")
        print(f"Question: {question}")

        docs = retriever.retrieve_context(question)

        if not docs:
            print("\nNo relevant documents retrieved.")
            print("-" * 70)
            continue

        print(
            f"\nRetrieved {len(docs)} relevant document(s).\n"
        )

        print(retriever.format_context(docs))

        print("-" * 70)


def run_generation_tests():
    """
    Runs the complete RAG pipeline.
    """

    rag = PolicyRAG()

    print("=" * 70)
    print("RAG GENERATION EVALUATION")
    print("=" * 70)

    for idx, question in enumerate(GENERATION_TESTS, start=1):

        print(f"\n[{idx}/{len(GENERATION_TESTS)}]")
        print(f"Question: {question}")

        answer = rag.generate(question)

        print("\nAnswer:\n")
        print(answer)

        print("-" * 70)


def main():

    print()

    run_retrieval_tests()

    print("\n\n")

    run_generation_tests()


if __name__ == "__main__":
    main()