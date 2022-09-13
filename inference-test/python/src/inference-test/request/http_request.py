import asyncio
import json
import time
from typing import List
import aiohttp
import requests


def callDummySericeBatch(questions: List[str], texts: List[str]):
    '''
    동기 request batch
    '''
    responses = []
    for question, text in zip(questions, texts):
        stime = time.time()

        parameter = {
            "q": question,
            "t": text
        }
        service_url = "http://192.168.1.65:5000/lucy/lucyBERTDummy"

        response = requests.post(service_url, data=json.dumps(parameter))

        responses.append(response.text)
        print(f"건당 처리 시간 : {time.time() - stime:.5f} sec")
    return responses


def callDummySericeBatchAsync(questions: List[str], texts: List[str]):
    '''
    비동기 request batch
    '''
    service_url = "http://192.168.1.65:5000/lucy/lucyBERTDummy"

    async def fetch(conn, data):
        async with aiohttp.ClientSession(connector=conn, connector_owner=False) as session:
            async with session.post(service_url, data=json.dumps(data)) as response:
                return await response.text()

    async def batch():
        async with aiohttp.TCPConnector(limit_per_host=620, limit=620) as conn:
            async_requests = []

            for question, text in zip(questions, texts):
                parameter = {
                    "q": question,
                    "t": text
                }
                async_requests.append(fetch(conn, parameter))

            return await asyncio.gather(*async_requests)

    result = asyncio.run(batch())

    return result


# batch
standard_batch = 9000
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
questions, texts = [question] * standard_batch, [text] * standard_batch

stime = time.time()
print(callDummySericeBatch(questions, texts))
print(f"동기 호출 : {time.time() - stime:.5f} sec")

stime = time.time()
print(callDummySericeBatchAsync(questions, texts))
print(f"비동기 호출 : {time.time() - stime:.5f} sec")

