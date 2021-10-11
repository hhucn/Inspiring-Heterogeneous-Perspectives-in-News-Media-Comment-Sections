import time
from statistics import mean
from typing import List

import pandas as pd
from tqdm import tqdm

from BaselineModelWithCandidateSet import BaselineModelWithCandidateSet
from EmbeddingModels.BertEmbeddings import BertEmbeddings
from EmbeddingModels.ParaphraseMiniLML12V2 import ParaphraseMiniLML12V2
from EmbeddingModels.ParaphraseMpnetBaseV2 import ParaphraseMpnetBaseV2
from EmbeddingModels.StsbRobertaBaseV2 import StsbRobertaBaseV2
from EmbeddingModels.stsb_mpnet_base_v2 import StsbMpnetBasev2

USER_COMMENTS_IDS_LIST = [{"comment_id": "22023491.0", "article_id": "58e1f7f27c459f24986d8045"},
                          {"comment_id": "22023446.0", "article_id": "58e1f7f27c459f24986d8045"},

                          {"comment_id": "22023450.0", "article_id": "58e1f7f27c459f24986d8044"},
                          {"comment_id": "22023192.0", "article_id": "58e1f7f27c459f24986d8044"},

                          {"comment_id": "26867449.0", "article_id": "5adef218068401528a2aa514"},
                          {"comment_id": "26860388.0", "article_id": "5adef218068401528a2aa514"},

                          {"comment_id": "26846120.0", "article_id": "5add2004068401528a2aa14b"},
                          {"comment_id": "26846008.0", "article_id": "5add2004068401528a2aa14b"},

                          {"comment_id": "26831320.0", "article_id": "5adb8334068401528a2a9f74"},
                          {"comment_id": "26831204.0", "article_id": "5adb8334068401528a2a9f74"},

                          {"comment_id": "21411837.0", "article_id": "589ad50a95d0e0392607f2d1"},
                          {"comment_id": "21411352.0", "article_id": "589ad50a95d0e0392607f2d1"},

                          {"comment_id": "26171286.0", "article_id": "5a9825cc410cf7000162eac9"},

                          {"comment_id": "26180988.0", "article_id": "5a982e2a410cf7000162eaf1"},

                          {"comment_id": "26249262.0", "article_id": "5aa081d947de81a90120b67d"}]

EMBEDDINGS_MODELS = [StsbMpnetBasev2(), BertEmbeddings(), ParaphraseMpnetBaseV2(), StsbRobertaBaseV2(),
                     ParaphraseMiniLML12V2()]
N_NEIGHBORS = 4
N_COMMENTS = 6


def run_model(model) -> List:
    """
    Runs the model for the given user comment ids in USER_COMMENTS_ID_LIST
    :param model:
    :return:
    """
    baseline_model = BaselineModelWithCandidateSet(model=model, n_neighbors=N_NEIGHBORS)
    data = []
    elapsed_time = []
    for user_comment_id in USER_COMMENTS_IDS_LIST:
        start_time = time.time()
        similar_article_ids, _ = baseline_model.find_similar_article(
            baseline_model.knowledge_base.get_article_keywords(user_comment_id["article_id"]))
        user_comment = baseline_model.knowledge_base.get_user_comment(user_comment_id["comment_id"])
        comment_suggestions = baseline_model.find_n_most_similar_comments(similar_article_ids, user_comment,
                                                                          n_comments=N_COMMENTS)
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        for sugesstion in comment_suggestions:
            row = {}
            row["model"] = str(model)
            row["user_comment"] = user_comment
            row["suggestion"] = sugesstion[0]
            row["similarity_score"] = sugesstion[1].item()
            row["suggestion_article_id"] = sugesstion[2]
            row["user_comment_article_id"] = user_comment_id["article_id"]
            data.append(row)
        row = {}
        row["model"] = str(model)
        row["user_comment"] = user_comment
        row.update({"suggestion_" + str(comment_suggestions.index(suggestion)): suggestion[0] for suggestion in
                    comment_suggestions})
        row["similarity_score"] = -1
        data.append(row)
        print(f"Average time to find suggestions for user comment {mean(elapsed_time)}")
    return data


if __name__ == '__main__':
    experiment_data = []
    for embedding_model in tqdm(EMBEDDINGS_MODELS):
        experiment_data.extend(run_model(embedding_model))
    experiment_data_df = pd.DataFrame(experiment_data)
    experiment_data_df.to_csv('ExperimentResults/ExperimentResults.csv', sep=";")
