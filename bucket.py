import json
import boto3


def S3():
    s3 = boto3.resource("s3", region_name='us-east-2',
        aws_access_key_id="AKIAVVZS666ZXRWNM6HM",
        aws_secret_access_key="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").Bucket("data-close")
    return s3




def S3_t():
    s3 = boto3.resource("s3", region_name='us-east-2',
        aws_access_key_id="AKIAVVZS666ZXRWNM6HM",
        aws_secret_access_key="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").Bucket("data-targetdict")
    return s3