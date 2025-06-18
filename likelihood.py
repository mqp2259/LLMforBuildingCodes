# for logarithm likelihood metric on lm evaluation harness
from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, perplexity


_CITATION = ""


class NBC_MC(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = ""    # insert path to testing dataset
    DATASET_NAME = None

    def has_training_docs(self):
        # No training data, so `False`.
        return False

    def has_validation_docs(self):
        # No validation data, so `False`.
        return False

    def has_test_docs(self):
        # Task has test data, so `True`
        return True

    def training_docs(self):
        if self.has_training_docs():
            raise Exception("No training data")
        # No training data
        return None

    def validation_docs(self):
        if self.has_validation_docs():
            raise Exception("No validation data")
        # No validation data
        return None

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
        else:
            raise Exception("There must be test data")

    def _process_doc(self, doc):
        return {
            "query": self.doc_to_text(doc),    # The query prompt.
            "choices": doc["answers"],    # The list of choices.
            "gold": 0,    # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        question = doc["question"]
        context = doc["context"]
        return f"You are a helpful assistant who truthfully answers a human's questions based on provided context.\n\nHuman: Consider the following: \"{context}\". {question}\nAI:"
