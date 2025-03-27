import requests
import re
import random
from evaluator import Evaluator
import os


class Chem_Evaluator(Evaluator):
    def __init__(self, choices, api_url, api_key):
        super(Chem_Evaluator, self).__init__(choices, "chem_api_model", k=-1)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        cot=False,
        save_result_dir=None,
        with_prompt=False,
        constrained_decoding=False,
        do_test=False,
    ):
        all_answers = {}
        correct_num = 0
        if save_result_dir:
            result = []
            score = []

        answers = ["NA"] * len(test_df) if do_test is True else list(test_df["answer"])

        for row_index, row in test_df.iterrows():
            question = self.format_example(
                row,
                include_answer=False,
            )

            # Prepare API request
            payload = {
                "model": "internvl2_5_chemvlm20250306",
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            }

            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload
                )
                response.raise_for_status()
                gen_ans = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"API call failed: {e}")
                gen_ans = ""

            ans = self.extract_answer(row, gen_ans)

            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0

            print(f"\n=======begin {str(row_index)}=======")
            print("question: ", question)
            print("response: ", gen_ans)
            print("ans: ", ans)
            print("ground truth: ", answers[row_index], "\n")

            if save_result_dir:
                result.append(gen_ans)
                score.append(correct)

            print(f"=======end {str(row_index)}=======")
            all_answers[str(row_index)] = ans

        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_test.csv"))

        return correct_ratio, all_answers

    def extract_answer(self, line, gen_ans):
        # Same extraction logic as Llama_Evaluator
        m = re.findall(r"所以答案是(.+?)。", gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True

        answer_patterns = [
            r"([ABCD])是正确的",
            r"选项([ABCD])正确",
            r"答案为([ABCD])",
            r"答案是([ABCD])",
            r"答案([ABCD])",
            r"选择([ABCD])",
            r"答案：([ABCD])",
            r"选择答案([ABCD])",
        ]

        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False

        m = re.findall(r"[ABCD]", gen_ans, re.M)
        if len(m) >= 1:
            answer = m[0]
            return answer, False

        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(line[f"{c}"])] = c
            pattern += re.escape(str(line[f"{c}"])) + "|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)

        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False

        return random.choice("ABCD"), False
