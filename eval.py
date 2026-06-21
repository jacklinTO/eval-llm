from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_local_evaluation():
    test_cases = [
            LLMTestCase(
            input="What is the capital of United Kingdom?",
            actual_output="The capital of United Kingdom is London.",
            retrieval_context=["London is the capital and most populous city of United Kingdom."]),
            LLMTestCase(
            input="What is the capital of Canada?",
            actual_output="The capital of Canada is Toronto.",
            expected_output="The capital of Canada is Ottawa.",
            retrieval_context=["Ottawa is the capital and fourth populous city of Canada."]),
    ]
    
    metric = AnswerRelevancyMetric(threshold=0.7)
    
    evaluate(test_cases=test_cases, metrics=[metric])

if __name__ == "__main__":
    test_local_evaluation()