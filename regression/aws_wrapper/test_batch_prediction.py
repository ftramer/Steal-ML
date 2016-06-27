import boto3
from models import models
import re
import json
import uuid
import time
import timeit

s3_client = boto3.client('s3')
ml_client = boto3.client('machinelearning')

model = models['digits']
model_id = model['id']

aws_model = ml_client.get_ml_model(
    MLModelId=model_id,
    Verbose=True
)

schema = aws_model['Schema']
target_reg = '"targetAttributeName":"(.*?)"'
target = re.findall(target_reg, schema)[0]
feature_reg = '"attributeName":"(.*?)","attributeType":"(.*?)"'
feature_info = re.findall(feature_reg, schema)
val_name = [f for (f, t) in feature_info if f != target]


start_time = timeit.default_timer()

s3_client.upload_file("temp.txt", Bucket='fz84-aml', Key="digits-batch",
                      ExtraArgs={'ACL': 'public-read'})

with open("temp_schema.json") as f:
    ds_schema = json.load(f)

ds_id = 'digits_batch_source_' + str(uuid.uuid4())

response = ml_client.create_data_source_from_s3(
    DataSourceId=ds_id,
    DataSourceName='digits-batch-source',
    DataSpec={
        'DataLocationS3': 's3://fz84-aml/digits-batch',
        'DataSchema': json.dumps(ds_schema)
    }
)
print response

pred_id = 'digits_batch_' + str(uuid.uuid4())

response = ml_client.create_batch_prediction(
    BatchPredictionId=pred_id,
    BatchPredictionName='digits-batch',
    MLModelId=model_id,
    BatchPredictionDataSourceId=ds_id,
    OutputUri='s3://fz84-aml/digits/'
)

print response

i = 0
while True:
    response = ml_client.get_batch_prediction(
        BatchPredictionId=pred_id
    )
    print '{} sec: {}'.format(i, response['Status'])
    if response['Status'] == 'COMPLETED':
        break

    i += 1
    time.sleep(1)

end_time = timeit.default_timer()

print 'Batch prediction took {} seconds'.format(end_time - start_time)