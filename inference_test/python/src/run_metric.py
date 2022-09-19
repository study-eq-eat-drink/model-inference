from inference_test.metric.nsmc_triton_metric import UserBatchDPSMetricTester


def test_metric_nsmc_triton():
    host = "http://localhost:9000/v2/models/nsmc-onnx/infer"
    tester = UserBatchTPSMetricTester(host)
    data_count = 1000
    request_batch_count = 8
    start_user_count = 3
    end_user_count = 5

    tester.run_test(data_count, request_batch_count, start_user_count, end_user_count)


if __name__ == "__main__":
    test_metric_nsmc_triton()