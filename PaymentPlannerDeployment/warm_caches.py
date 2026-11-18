"""Build-time downloads only; do not create disk-backed user data at build."""
import tiktoken
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from cache_config import configure_caches

if __name__ == "__main__":
    configure_caches()
    tiktoken.get_encoding("cl100k_base")
    DefaultEmbeddingFunction()(["Verified synthetic transaction example"])
    print("Tokenizer and local embedding caches are ready.")
