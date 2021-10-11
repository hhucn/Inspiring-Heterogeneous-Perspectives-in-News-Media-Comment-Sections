import torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings

ARTICLE_FILENAME = ["embedded_article_ArticlesApril2017_bert-large-uncased.pt",
                    "embedded_article_ArticlesApril2018_bert-large-uncased.pt",
                    "embedded_article_ArticlesFeb2017_bert-large-uncased.pt",
                    "embedded_article_ArticlesFeb2018_bert-large-uncased.pt",
                    "embedded_article_ArticlesJan2017_bert-large-uncased.pt",
                    "embedded_article_ArticlesJan2018_bert-large-uncased.pt",
                    "embedded_article_ArticlesMarch2017_bert-large-uncased.pt",
                    "embedded_article_ArticlesMarch2018_bert-large-uncased.pt",
                    "embedded_article_ArticlesMay2017_bert-large-uncased.pt"]
COMMENTS_FILENAME = ["embedded_comments_CommentsApril2017.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsApril2018.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsFeb2017.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsFeb2018.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsJan2017.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsJan2018.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsMarch2017.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsMarch2018.csv_bert-large-uncased.pt",
                     "embedded_comments_CommentsMay2017.csv_bert-large-uncased.pt"]
EMBEDDINGS_PATH = "Embeddings/"


class BertEmbeddings:
    def __init__(self):
        self.model = TransformerDocumentEmbeddings('bert-large-uncased', fine_tune=False)
        self.article_embedding_filename = ARTICLE_FILENAME
        self.comments_embedding_filename = COMMENTS_FILENAME
        self.embeddings_path = EMBEDDINGS_PATH

    def __str__(self):
        return "bert-large-uncased"

    def compute_embedding(self, text) -> torch.Tensor:
        """
        Compute the embeddings for the given text
        :param text:
        :return:
        """
        text_sentence = Sentence(text)
        self.model.embed(text_sentence)
        return text_sentence.embedding
