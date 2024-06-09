from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


classification_pipeline = Pipeline(

    [
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeInputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DomainProcessing',pp.DomainProcessing(variables_to_modify=config.FEATURES_TO_MODIFY,
                                                variables_to_add=config.FEATURES_TO_ADD)),
        ('DropFeatures',pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Logtransform',pp.LogTransformer(variables=config.LOG_FEATURES)),
        ('MinMaxscaler',MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))

    ]
)

