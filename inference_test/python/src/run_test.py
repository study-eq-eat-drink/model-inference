from inference_test.model_test.nsmc_test import NSMCOnnxTritonTester


def test_metric_nsmc_triton():
    host = "http://localhost:9000/v2/models/nsmc-onnx/infer"
    tester = NSMCOnnxTritonTester(host)
    data_count = 100
    request_batch_count = 8
    start_user_count = 1
    end_user_count = 3

    tester.run_test(data_count, request_batch_count, start_user_count, end_user_count)


if __name__ == "__main__":
    test_metric_nsmc_triton()