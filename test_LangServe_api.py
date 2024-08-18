import requests

ENDPOINT="http://localhost:9000/llama2"

def test_can_call_input_schema():
    response = input_schema()
    assert response.status_code==200
    data = response.json()

def test_can_invoke():
    payload=new_payload()
    response = invoke(payload)
    assert response.status_code == 200
    data=response.json()

def test_can_batch():
    payload=new_payload_batch()
    response = batch(payload)
    assert response.status_code == 200
    data=response.json()

def test_can_stream():
    payload=new_payload()
    response = stream(payload)
    assert response.status_code == 200

    # for chunk in response:
    #     print(chunk)

def input_schema():
    return requests.get(ENDPOINT + "/input_schema")

def invoke(payload):
    return requests.post(ENDPOINT + "/invoke", json=payload)

def batch(payload):
    return requests.post(ENDPOINT + "/batch", json=payload)

def stream(payload):
    return requests.post(ENDPOINT + "/stream", json=payload)

def new_payload():
    return {
        "input": {"input": "hello"},
        "config": {"configurable":{"temperature": 0}},
        "kwargs": {}
    }

def new_payload_batch():
    return {
        "inputs": [{"input": "hi"}
        ],
        "config": {"configurable": {"temperature": 0}},
        "kwargs": {}
    }