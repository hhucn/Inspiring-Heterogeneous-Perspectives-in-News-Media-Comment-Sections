from typing import List

from sklearn.neighbors import NearestNeighbors
from torch import nn

from KnowledgeBase.KnowledgeBase import KnowledgeBase


class BaselineModelWithCandidateSet:
    def __init__(self, model, n_neighbors: int = 1):
        self.knowledge_base = KnowledgeBase(model.embeddings_path, model.article_embedding_filename,
                                            model.comments_embedding_filename)
        self.model = model
        self.n_neighbors = n_neighbors

    def find_similar_article(self, user_article_keywords: List) -> [List, List]:
        """
        Find the k-most similar article for the given article keywords
        :param user_article_keywords:
        :return:
        """
        user_article_keywords_embeddings = self.model.compute_embedding(user_article_keywords)
        candidate_article_embeddings = self.knowledge_base.get_article_embeddings()
        k_nearest_neighbor_model = NearestNeighbors(n_neighbors=self.n_neighbors)
        k_nearest_neighbor_model.fit(candidate_article_embeddings)
        article_indexes = k_nearest_neighbor_model.kneighbors([user_article_keywords_embeddings.tolist()],
                                                              return_distance=False)[0]
        return self.knowledge_base.get_article_ids(article_indexes), article_indexes

    def find_n_most_similar_comments(self, article_ids: List, user_comment: str, n_comments: int = 5) -> List:
        """
        Finds the n-most similar comments for the given article and comment
        :param article_ids:
        :param user_comment:
        :param n_comments:
        :return:
        """
        user_comment_embedding = self.model.compute_embedding(user_comment)
        comments = [[comment["commentBody"], comment["embedding"], comment["article_id"]] for comment in
                    self.knowledge_base.comments if
                    comment["article_id"] in article_ids]

        comments_with_score = []
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for comment in comments:
            if comment[1].shape != user_comment_embedding.shape:
                raise ValueError("User comment shape and comment shape does not fit!")
            score = cos(comment[1], user_comment_embedding)
            if user_comment == comment[0]:
                continue
            comments_with_score.append([comment[0], score, comment[2]])
        comments_with_score.sort(key=lambda x: x[1], reverse=True
                                 )
        return comments_with_score[:n_comments]
