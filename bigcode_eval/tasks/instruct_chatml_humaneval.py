"""Instruction version of HumanEval used for ChatML Models evaluation
Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""
import re
import warnings

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def generate_prompt(sample):
    return f"<|im_start|>system\n" \
           f"You are an expert in Software Development.<|im_end|>\n" \
           f"<|im_start|>user\n" \
           f"Complete a Python function to solve the following problem:\n" \
           f"\n" \
           f"```python\n" \
           f"{sample['prompt'].strip()}\n" \
           f"```\n" \
           f"Use only single Markdown code block in your response!<|im_end|>\n" \
           f"<|im_start|>assistant"


class HumanEvalChatML(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/instructhumaneval"

    def __init__(self):
        super().__init__(
            stop_words=["<|im_end|>"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = generate_prompt(doc)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        cropped_generation = generation[generation.find("assistant\n") + len("assistant\n"):]
        sample = self.get_dataset()[idx]
        function_name = sample["signature"]
        try:
            cropped_generation = get_completion(cropped_generation, function_name)
            final_generation = sample["prompt"].strip().split("def ")[0] + cropped_generation
        except:
            print(f"Error in postprocessing generation for {function_name}")
            print(generation)
            final_generation = generation
        return final_generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results


def get_completion(response, function_name):
    code_snippets = re.findall(
        "```(.*?)```",
        response,
        re.DOTALL
    )
    function_name_title = function_name[:function_name.find("(")].strip()
    code_snippets = [code_snippet for code_snippet in code_snippets if
                     "return " in code_snippet and f"def " + function_name_title in code_snippet]
    if len(code_snippets) > 1:
        warnings.warn(f"More than one code snippet found for {function_name}")
        warnings.warn(str(code_snippets))
    code_snippet = sorted(code_snippets, key=lambda x: len(x), reverse=True)[0]
    code_snippet = code_snippet.replace("python\n", "", 1) if code_snippet.startswith("python\n") else code_snippet
    code_snippet = code_snippet.strip()
    return code_snippet


def get_completion_(response, function_name):
    code_snippets = [
        code_snippet
        for code_snippet in
        re.findall(
            escape_special_characters(f"def {function_name}:\\n") + "(.*?)```",
            response,
            re.DOTALL
        )
    ]
    code_snippets = [code_snippet for code_snippet in code_snippets if "return" in code_snippet]
    code_snippet = code_snippets[0]
    code_snippet = code_snippet.replace("python\n", "", 1) if code_snippet.startswith("python\n") else code_snippet
    code_snippet = code_snippet.rstrip()
    return code_snippet
    # code_snippet = re.sub(r'""".*?"""', '', code_snippet, flags=re.DOTALL)
    # return "    " + code_snippet.strip()


def escape_special_characters(pattern):
    special_characters = ".[](){}*+?^$|"
    escaped_pattern = "".join(f'\\{char}' if char in special_characters else char for char in pattern)
    return escaped_pattern
