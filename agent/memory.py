from collections import deque

import chromadb
from chromadb.utils import embedding_functions
import uuid


class FinancialSituationMemory:
    """
    独立记忆体：支持为不同角色创建隔离的 Collection
    """

    def __init__(self, collection_name, db_path):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.long_term_rules = deque(maxlen=5)
        self.log_buffer = deque(maxlen=7)

    def add_situations(self, situation_reflection_pairs):
        """
        :param situation_reflection_pairs: [(situation_text, reflection_result), ...]
        """
        if not situation_reflection_pairs:
            return

        documents = []  # 存 Situation (用于检索匹配)
        metadatas = []  # 存 Reflection/Belief (作为结果)
        ids = []

        for sit, ref in situation_reflection_pairs:
            documents.append(sit)
            metadatas.append({"reflection": ref})
            ids.append(str(uuid.uuid4()))
            self.log_buffer.append(ref)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def get_memories(self, current_situation, n_matches=2):
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[current_situation],
            n_results=min(n_matches, self.collection.count()),
            include=['documents', 'metadatas', 'distances']
        )

        retrieved_items = []
        if results['ids']:
            # 遍历检索结果
            for i in range(len(results['ids'][0])):
                reflection_text = results['metadatas'][0][i].get("reflection", "")
                situation_text = results['documents'][0][i]
                distance = results['distances'][0][i]

                retrieved_items.append(
                    {
                        "matched_situation": situation_text,
                        "recommendation": reflection_text,
                        "similarity_score": 1 - distance,
                    }
                )

        return retrieved_items

    def add_weekly_rule(self, new_rule: str):
        if new_rule not in self.long_term_rules:
            self.long_term_rules.append(new_rule)

    def get_rules(self):
        if not self.long_term_rules:
            return ""
        return "\n近期实战备忘录 (基于周度复盘):\n" + "\n".join([f"- {r}" for r in self.long_term_rules])

    def fetch_recent_logs(self, k=7) -> list[str]:
        return list(self.log_buffer)[-k:]
