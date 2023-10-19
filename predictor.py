def lambda_handler(event, context):
    response = {
        'statusCode': 200,
        'body': 'This is Hello World #1 and this is Predictor example
        Testing second change',
        
        'headers': {
            'Content-Type': 'application/json'
        }
    }
    return response

