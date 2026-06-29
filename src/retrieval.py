# retrieval.py

from .utils import (
    TOP_K,
    MAX_DISTANCE_THRESHOLD,
    get_vector_store,
)


class PolicyRetriever:
    """
    Handles semantic retrieval from the Chroma vector store.
    """

    def __init__(self):
        self.vector_store = get_vector_store()

    def retrieve_context(self, query: str, top_k: int = TOP_K):
        """
        Searches the vector store and filters retrieved documents
        based on the configured maximum distance threshold.
        """

        docs_with_distances = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
        )

        filtered_docs = []

        for doc, distance in docs_with_distances:

            print(
                f"   [Debug] Found chunk in "
                f"'{doc.metadata.get('source')}' "
                f"with raw distance: {distance:.4f}"
            )

            if distance <= MAX_DISTANCE_THRESHOLD:

                doc.metadata["raw_distance"] = round(distance, 4)

                filtered_docs.append(doc)

        return filtered_docs

    def format_context(self, docs):
        """
        Formats retrieved chunks into a clean context string while
        preventing duplicate chunks.
        """

        context_blocks = []
        seen_contents = set()

        for idx, doc in enumerate(docs, start=1):

            clean_content = doc.page_content.strip()

            if clean_content in seen_contents:
                continue

            seen_contents.add(clean_content)

            source_file = doc.metadata.get(
                "source",
                "Unknown Document",
            )

            distance = doc.metadata.get(
                "raw_distance",
                0.0,
            )

            header_path = []

            for level in (
                "Header 1",
                "Header 2",
                "Header 3",
                "Header 4",
            ):

                if level in doc.metadata:
                    header_path.append(doc.metadata[level])

            structure = (
                " > ".join(header_path)
                if header_path
                else "General Section"
            )

            block = (
                f"[Source Document {idx}]: {source_file} "
                f"(Raw Distance: {distance})\n"
                f"[Section Hierarchy]: {structure}\n"
                f"[Content]:\n"
                f"{clean_content}\n"
                f"{'-' * 40}"
            )

            context_blocks.append(block)

        return "\n\n".join(context_blocks)