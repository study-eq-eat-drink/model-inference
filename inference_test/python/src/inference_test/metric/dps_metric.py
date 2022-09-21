from inference_test.request.http_request import ManyHTTPRequester
from typing import List

import time


class UserBatchDPSMetricTester:

    def __init__(self, host_url: str):
        self.host_url = host_url

    def run_test(self, data_count: int, test_request_list: List[str], start_user_count: int, end_user_count: int):

        host_url = self.host_url

        triton_request_bodys = test_request_list

        time_results = []
        response_results = []
        dps_results = []
        user_counts = []
        # 유저 단위 루프
        for user_count in range(start_user_count, end_user_count + 1):
            print(f"# request user count : {user_count}...")
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
            print(f"total request count : {data_count}")
            print(f"total request time : {time_result:.5f} sec")
            print(f"dps : {dps_result} dps")
            # print(f"response_result : {time_result:.5f} sec")
            print(response_result[:10] + response_result[:-10])



