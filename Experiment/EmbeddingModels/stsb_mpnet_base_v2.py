import torch
from sentence_transformers import SentenceTransformer

ARTICLE_FILENAME = ["embedded_article_ArticlesApril2017_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesApril2018_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesFeb2017_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesFeb2018_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesJan2017_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesJan2018_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesMarch2017_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesMarch2018_stsb-mpnet-base-v2.pt",
                    "embedded_article_ArticlesMay2017_stsb-mpnet-base-v2.pt"]
COMMENTS_FILENAME = ["embedded_comments_CommentsApril2017.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsApril2018.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsFeb2017.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsFeb2018.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsJan2017.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsJan2018.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsMarch2017.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsMarch2018.csv_stsb-mpnet-base-v2.pt",
                     "embedded_comments_CommentsMay2017.csv_stsb-mpnet-base-v2.pt"]
EMBEDDINGS_PATH = "Embeddings/"


class StsbMpnetBasev2:
    def __init__(self):
        self.model = SentenceTransformer('stsb-mpnet-base-v2')
        self.article_embedding_filename = ARTICLE_FILENAME
        self.comments_embedding_filename = COMMENTS_FILENAME
        self.embeddings_path = EMBEDDINGS_PATH

    def __str__(self):
        return "stsb-mpnet-base-v2"

    def compute_embedding(self, text: str) -> torch.Tensor:
        """
                        Compute the embeddings for the given text
                        :param text:
                        :return:
        """
        return self.model.encode([text], convert_to_tensor=True)[0]
