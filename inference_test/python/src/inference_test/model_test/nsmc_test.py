from inference_test.model.nsmc_model import NsmcKoelectraSmallTokenizer
from inference_test.metric.dps_metric import UserBatchDPSMetricTester

import numpy as np

import json


class NSMCOnnxTritonTester:

    def __init__(self, host_url: str):
        self.host_url = host_url
        self.user_batch_metric_tester = UserBatchDPSMetricTester(host_url)

    def run_test(self, data_count: int, request_batch_count: int, start_user_count: int, end_user_count: int):

        # 테스트 데이터 만들기
        test_request_parameters = self.create_test_request_parameters(data_count, request_batch_count)

        self.user_batch_metric_tester.run_test(
            test_request_parameters, start_user_count, end_user_count
        )


    def create_test_request_parameters(self, data_count: int, request_batch_count: int):
        test_test = "영화 " * 512
        model_inputs = self.create_nsmc_test_model_inputs(data_count)

        # triton api request parameter 형식으로 만들기
        triton_parameters = self.parse_model_inputs_to_triton_parameters(
            model_inputs['input_ids'],
            model_inputs['attention_mask'],
            model_inputs['token_type_ids'],
            batch_size=request_batch_count
        )

        triton_request_bodys = [json.dumps(triton_parameter) for triton_parameter in triton_parameters]

        return triton_request_bodys

    def create_nsmc_test_model_inputs(self, data_count: int):
        test_test = "영화 " * 512
        model_inputs = NsmcKoelectraSmallTokenizer.tokenize_model_input(test_test)

        model_inputs['input_ids'] = np.repeat(model_inputs['input_ids'], data_count, axis=0)
        model_inputs['attention_mask'] = np.repeat(model_inputs['attention_mask'], data_count, axis=0)
        model_inputs['token_type_ids'] = np.repeat(model_inputs['token_type_ids'], data_count, axis=0)

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
                    "shape": [],
                    "datatype": "INT32",
                    "data": []
                },
                {
                    "name": "input_2",
                    "shape": [],
                    "datatype": "INT32",
                    "data": []
                },
                {
                    "name": "input_3",
                    "shape": [],
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
