from inference_test.model.nsmc_model import NsmcKoelectraSmallTokenizer
from inference_test.request.http_request import ManyHTTPRequester

import numpy as np

import time

import json

class UserBatchDPSMetricTester:

    def __init__(self, host_url: str):
        self.host_url = host_url

    def run_test(self, data_count: int, request_batch_count: int, start_user_count: int, end_user_count: int):

        host_url = self.host_url

        # 테스트 데이터 만들기
        test_model_inputs = self.create_nsmc_test_data(data_count)

        # triton api request parameter 형식으로 만들기
        # 몇개씩 쪼개지?
        triton_parameters = self.parse_model_inputs_to_triton_parameters(
            test_model_inputs['input_ids'],
            test_model_inputs['attention_mask'],
            test_model_inputs['token_type_ids'],
            batch_size=request_batch_count
        )
        triton_request_bodys = [json.dumps(triton_parameter) for triton_parameter in triton_parameters]

        time_results = []
        response_results = []
        dps_results = []
        user_counts = []
        # 유저 단위 루프
        for user_count in range(start_user_count, end_user_count + 1):
            request_caller = ManyHTTPRequester(True, user_count)

            start_time = time.time()
            responses = request_caller.request_all(host_url, triton_request_bodys)

            # 시간 측정
            request_time = time.time() - start_time
            time_results.append(request_time)

            user_counts.append(user_count)

            # response 저장
            response_results.append(responses)

            # dps 측정
            dps_results.append(data_count / request_time)


        # 결과 Print
        for user_count, time_result, dps_result, response_result in zip(user_counts, time_results, dps_results, response_results):
            print("=================================")
            print(f"user count : {user_count}")
            print(f"total request time : {time_result:.5f} sec")
            print(f"dps : {dps_result} dps")
            # print(f"response_result : {time_result:.5f} sec")
            # print(response_result[:10] + response_result[:-10])

    def create_nsmc_test_data(self, batch_size=10):
        test_test = "영화 " * 512
        model_inputs = NsmcKoelectraSmallTokenizer.tokenize_model_input(test_test)
        # input_ids
        # attention_mask
        # token_type_ids

        model_inputs['input_ids'] = np.repeat(model_inputs['input_ids'], batch_size, axis=0)
        model_inputs['attention_mask'] = np.repeat(model_inputs['attention_mask'], batch_size, axis=0)
        model_inputs['token_type_ids'] = np.repeat(model_inputs['token_type_ids'], batch_size, axis=0)
        return model_inputs

    def parse_model_inputs_to_triton_parameters(self, input_ids, attention_masks, token_type_ids, batch_size=5):

        batch_requests = []

        start_idx = 0
        for idx in range(int(len(input_ids) / batch_size) + 1):
            last_idx = start_idx + batch_size

            request_parameter = self.get_nsmc_triton_infer_template()
            request_parameter["id"] = str(last_idx + 1)

            slice_input_ids = input_ids[start_idx:last_idx]
            request_parameter["inputs"][0]["data"] = slice_input_ids.tolist()
            request_parameter["inputs"][0]["shape"] = list(slice_input_ids.shape)

            slice_attention_masks = attention_masks[start_idx:last_idx]
            request_parameter["inputs"][1]["data"] = slice_attention_masks.tolist()
            request_parameter["inputs"][1]["shape"] = list(slice_attention_masks.shape)

            slice_token_type_ids = token_type_ids[start_idx:last_idx]
            request_parameter["inputs"][2]["data"] = slice_token_type_ids.tolist()
            request_parameter["inputs"][2]["shape"] = list(slice_token_type_ids.shape)

            start_idx = last_idx

            batch_requests.append(request_parameter)
        return batch_requests

    def get_nsmc_triton_infer_template(self):
        request_format = \
        {
            "inputs": [
                {
                    "name": "input_1",
                    "shape": [512],
                    "datatype": "INT32",
                    "data": []
                },
                {
                    "name": "input_2",
                    "shape": [512],
                    "datatype": "INT32",
                    "data": []
                },
                {
                    "name": "input_3",
                    "shape": [512],
                    "datatype": "INT32",
                    "data": []
                }
            ],
            "outputs": [
                {
                    "name": "dense"
                }
            ]
        }
        return request_format
