from tqdm import tqdm
from elasticsearch import Elasticsearch
import pandas as pd
import yaml
from eland.ml.ltr import LTRModelConfig, QueryFeatureExtractor
import numpy
from eland.ml.ltr import FeatureLogger
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from eland.ml import MLModel

tqdm.pandas()

with open("./config/config.yml", "r") as file:
    config = yaml.safe_load(file)

es = Elasticsearch(
    cloud_id=config["elastic"]["cloud_id"], api_key=config["elastic"]["api_key"]
)

judgments_df = pd.read_csv(config["elastic"]["judgement_list_filename"])
print(judgments_df)

ltr_config = LTRModelConfig(
    feature_extractors=[
        QueryFeatureExtractor(
            feature_name="story_bm25", query={"match": {"story": "{{query}}"}}
        )
    ]
)


feature_logger = FeatureLogger(es, config["elastic"]["index_name"], ltr_config)


# This method will be applied for each query group in the judgment log:
def _extract_query_features(query_judgements_group):
    # Retrieve document ids in the query group as strings.
    doc_ids = query_judgements_group["doc_id"].astype("str").to_list()

    # Resolve query params for the current query group (e.g.: {"query": "batman"}).
    query_params = {"query": query_judgements_group["query"].iloc[0]}

    # Extract the features for the documents in the query group:
    doc_features = feature_logger.extract_features(query_params, doc_ids)

    # Adding a column to the dataframe for each feature:
    for feature_index, feature_name in enumerate(ltr_config.feature_names):
        query_judgements_group[feature_name] = numpy.array(
            [doc_features[doc_id][feature_index] for doc_id in doc_ids]
        )

    return query_judgements_group


judgments_with_features = judgments_df.groupby(
    "query_id", group_keys=False
).progress_apply(_extract_query_features)


print(judgments_with_features)

# Create the ranker model:
ranker = XGBRanker(
    objective="rank:ndcg",
    eval_metric=["ndcg@10"],
    early_stopping_rounds=20,
)

# Shaping training and eval data in the expected format.
X = judgments_with_features[ltr_config.feature_names]
y = judgments_with_features["grade"]
groups = judgments_with_features["query_id"]

# Split the dataset in two parts respectively used for training and evaluation of the model.
group_preserving_splitter = GroupShuffleSplit(n_splits=1, train_size=0.9).split(
    X, y, groups
)
train_idx, eval_idx = next(group_preserving_splitter)

train_features, eval_features = X.loc[train_idx], X.loc[eval_idx]
train_target, eval_target = y.loc[train_idx], y.loc[eval_idx]
train_query_groups, eval_query_groups = groups.loc[train_idx], groups.loc[eval_idx]

# Training the model
ranker.fit(
    X=train_features,
    y=train_target,
    group=train_query_groups.value_counts().sort_index().values,
    eval_set=[(eval_features, eval_target)],
    eval_group=[eval_query_groups.value_counts().sort_index().values],
    verbose=True,
)


# Import the model to Elastic

MLModel.import_ltr_model(
    es_client=es,
    model=ranker,
    model_id=config["elastic"]["model_id"],
    ltr_model_config=ltr_config,
    es_if_exists="replace",
)
