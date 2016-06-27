__author__ = 'Fan'

import boto3

from algorithms.OnlineBase import OnlineBase
from utils.logger import *

logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AWSOnline(OnlineBase):
    def __init__(self, model_id, label_p, label_n, n_features, val_name, ftype, error):
        self.aws_client = boto3.client('machinelearning')
        super(self.__class__, self).__init__(model_id, label_p, label_n, None, n_features, ftype, error)
        self.val_name = val_name
        self.modelID = model_id

        def predict(x):
            assert len(x) == self.n_features, 'require %d, got %d' % (self.n_features, len(x))
            record = dict(zip(self.val_name, map(str, x)))
            response = self.aws_client.predict(
                MLModelId=model_id,
                Record=record,
                PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
            )
            return int(response['Prediction']['predictedLabel'])

        self.clf1 = predict

    def collect_with_score(self, n, spec=None):
        X = []

        collected = 0
        while True:
            x = self.random_vector(self.n_features, spec=None)
            y = self.query(x)

            assert len(x) == self.n_features, \
                'require %d, got %d' % (self.n_features, len(x))
            record = dict(zip(self.val_name, map(str, x)))
            response = self.aws_client.predict(
                MLModelId=self.modelID,
                Record=record,
                PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
            )
            score = response['Prediction']['predictedScores'].values()
            if score[0] < 1.0:
                collected += 1
                X.append((x, y, score))

            if collected >= n:
                break

        import pickle
        with open('queries_with_score-%d' % n, 'wb') as infile:
            pickle.dump(X, infile)

    def batch_predict(self, Xs, count=False):
        r = []
        if hasattr(Xs, 'tolist'):
            Xs = Xs.tolist()
        for x in Xs:
            r.append(self.clf1(x))
        return r
