from typing import List

import torch


class KnowledgeBase:
    def __init__(self, path: str, article_filename: List, comments_filename: List):
        self.comments = self.read_precomputed_comments(path, filenames=comments_filename)
        self.articles = self.read_precomputed_articles(path, filenames=article_filename)

    @staticmethod
    def read_precomputed_comments(path: str, filenames: List) -> List:
        """
        Read the comments with their precomputed embeddings
        :param path:
        :param filenames:
        :return:
        """
        embeddings = []
        [embeddings.extend(torch.load(path + filename, map_location=torch.device('cpu'))) for filename in filenames]
        return embeddings

    @staticmethod
    def read_precomputed_articles(path: str, filenames: List) -> List:
        """
        Read the articles with their precomputed embeddings for the keywords
        :param path:
        :param filenames:
        :return:
        """
        embeddings = []
        [embeddings.extend(torch.load(path + filename, map_location=torch.device('cpu'))) for filename in filenames]
        return embeddings

    def get_article_embeddings(self):
        """
        Returns the embeddings of the keywords of all articles
        :return:
        """
        embeddings = []
        for article in self.articles:
            embeddings.append(article["embedding"].tolist())
        return embeddings

    def get_article_headlines(self, article_indexes: List) -> List:
        """
        Returns the headline for the given article
        :param article_indexes:
        :return:
        """
        return [self.articles[index]["headline"] for index in article_indexes]

    def get_article_ids(self, article_indexes: List) -> List:
        """
        Returns the article id for the give article index
        :param article_indexes:
        :return:
        """
        return [self.articles[index]["ID"] for index in article_indexes]

    def get_article_keywords(self, article_id: str) -> List:
        """
        Returns the keywords for the given article id
        :param article_id:
        :return:
        """
        for article in self.articles:
            if article["ID"] == article_id:
                return article["keywords"]
        else:
            raise ValueError("Article not found")

    def get_user_comment(self, user_comment_id: str):
        """
        Returns the comment for the given comment id
        :param user_comment_id:
        :return:
        """
        for comment in self.comments:
            if str(comment["ID"]) == user_comment_id:
                return comment["commentBody"]
        else:
            raise ValueError("User comment not found")
