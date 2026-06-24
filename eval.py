import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_local_evaluation():
    test_cases = []
    with open('capitals.json', 'r') as data_file:
        data = json.load(data_file)
    
    for item in data.items():
        input_value = item[1]["input"]
        print(input_value)
        actual_output_value = item[1]["actual_output"]
        if "expected_output" in item[1]:
            expected_output_value = item[1]["expected_output"]
        else:
            expected_output_value = ''
        retrieval_context_value = item[1]["retrieval_context"]
        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            expected_output=expected_output_value,
            retrieval_context=[retrieval_context_value],
        )
        test_cases.append(test_case)
    
    metric = AnswerRelevancyMetric(threshold=0.7)
    
    evaluate(test_cases=test_cases, metrics=[metric])

if __name__ == "__main__":
    test_local_evaluation()