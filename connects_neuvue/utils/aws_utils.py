
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

aws_credentials_instruction = """

1. create the creditial file: ~/.aws/credentials
2. Add your aws_access_key_id and aws_secret_access_key id to the credentail file

[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
"""

def check_for_aws_credentials() -> bool:
    f"""
    Checks whether the AWS credentials file exists at ~/.aws/credentials.
    
    if the file doesn't exist follow the setup instructions below
    {aws_credentials_instruction}

    Returns:
        bool: True if the file exists, False otherwise.
    """
    credentials_path = Path.home() / ".aws" / "credentials"
    return credentials_path.is_file()


def get_secret() -> dict:
    f"""
    to retrieve the secret dictionary key using your stored credentials

    Setup
    -----
    For this to work on your machine you need 
    to set up the following configuration file
    {aws_credentials_instruction}
    """
    if not check_for_aws_credentials():
        raise FileNotFoundError(f"Need to first create aws credentials file \n{aws_credentials_instruction} ")

    secret_name = "rds!cluster-fbdb8fcc-f745-4361-b3ba-cf5f8349bc9a"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

    return eval(secret)