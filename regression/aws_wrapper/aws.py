import boto3
from models import models
import numpy as np

client = boto3.client('machinelearning')

model = models['adult10QB']
model_id = model['id']


response = client.create_realtime_endpoint(
    MLModelId=model_id
)
print response


"""
val_name = model['features']
feature_vector = np.zeros(len(val_name))
feature_vector = dict(zip(val_name, map(str, feature_vector)))

response = client.predict(
    MLModelId=model_id,
    Record=feature_vector,
    PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
)

print response

val_name = model['features']
feature_vector = np.zeros(len(val_name))
feature_vector[0] = 0.1
feature_vector = dict(zip(val_name, map(str, feature_vector)))

response = client.predict(
    MLModelId=model_id,
    Record=feature_vector,
    PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
)

print response

val_name = model['features']
feature_vector = np.zeros(len(val_name))
feature_vector[-1] = 0.1
feature_vector = dict(zip(val_name, map(str, feature_vector)))

response = client.predict(
    MLModelId=model_id,
    Record=feature_vector,
    PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
)

print response
"""
"""
response = client.get_ml_model(
    MLModelId=model_id,
    Verbose=True
)
print response

print

response = client.get_data_source(
    DataSourceId=model['data'],
    Verbose=True
)
print response
"""

"""
response = client.describe_ml_models()
print response
"""

