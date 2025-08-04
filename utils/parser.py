import re
import ollama

from utils.check_math_answer import grade_answer


def last_parenthesis_only_string(string):

    idx = string.rfind("((")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "(":
            num_left_braces_open += 1
        if string[i] == ")":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
        if retval.startswith("((") and retval.endswith("))"):
            retval = retval[2:-2]
    
    return retval


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def parse_gsm8k_answer(response: str):
    matches = re.findall(r"\(\(([-+]?\d+\.\d*|[-+]?\d+)\)\)", response)

    if len(matches) == 0:
        matches = re.findall(r"\\boxed\{([-+]?\d+\.\d*|[-+]?\d+)\}", response)

    if len(matches) == 0:
        matches = re.findall(r"\\boxed\{\(([-+]?\d+\.\d*|[-+]?\d+)\)\}", response)

    if len(matches) == 0:
        matches = re.findall(r"\(\( ([-+]?\d+\.\d*|[-+]?\d+) \)\)", response)

    if len(matches) == 0:
        matches = re.findall(r"\(([-+]?\d+\.\d*|[-+]?\d+)\)", response)

    if len(matches) == 0:
        return ""
    else:
        # assert len(matches) >= 1, "No answer be parsed in response for question in GSM8K dataset."
        answer = matches[-1]
        answer = answer.replace(",", "")

        return answer


def parse_mmlu_answer(response: str):
    matches = re.findall(r"\(\(([ABCDabcd])\)\)", response)

    if len(matches) == 0:
        matches = re.findall(r"\(([ABCDabcd])\)", response)

    # assert len(matches) >= 1, "No answer be parsed in response for question in MMLU dataset."
    if len(matches) == 0:
        return ""

    return matches[-1].upper()


def parse_csqa_answer(response: str):
    matches = re.findall(r"\(\(([ABCDEabcde])\)\)", response)

    if len(matches) == 0:
        matches = re.findall(r"\(([ABCDEabcde])\)", response)

    # assert len(matches) >= 1, "No answer be parsed in response for question in CSQA dataset."
    if len(matches) == 0:
        return ""

    return matches[-1].upper()

def format_math_str(text):
    text = text.replace(" ", "")
    text = text.replace('"', "")
    # text = text.replace("\"", "")
    # if text.endswith("\)"):
    #     text = text[1:-2]

    # text = text.replace("((", "")
    # text = text.replace("))", "")
    # text = text.replace(")\)", "")

    text = text[2:-2]
    return text


def parse_math500_answer(response: str):
    search_answer = last_parenthesis_only_string(response)

    if search_answer is not None:
        return search_answer

    search_answer = last_boxed_only_string(response)
    if search_answer is not None:
        return search_answer

    return ""


def parse_gpqa_answer(response: str):
    search_answer = last_parenthesis_only_string(response)

    if search_answer is not None:
        return search_answer

    return ""


def extract_MATH_answer(solution: str) -> str:

    extract_answer = last_boxed_only_string(solution)
    
    if extract_answer is not None:
        return extract_answer
    else:
        return ""


def extract_final_answer(solution_text, model_name='qwen2.5:7b-instruct-fp16'):
    extract_answer_prompt = (
        "You are a precise mathematical assistant. Please carefully read the following mathematical solution process and concisely extract and return the final answer. Note:\n"
        "1. Only extract the final calculation result, without including the solution process\n"
        "2. If the answer includes numbers and units, include both\n"
        "3. The answer typically appears in the last part of the text\n"
        "4. If there are multiple answers, list them in order\n"
        "5. Maintain the original format of the answer (including decimal places, fractional form, etc.)\n"
        "Please return the answer directly without additional explanation."
        "The Solution text is as follows:\n{}"
    )

    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': extract_answer_prompt.format(solution_text),
        },
    ])
    
    return response['message']['content']


PARSE_ANSWER_FUNCS = {
    "MMLU": parse_mmlu_answer,
    "CSQA": parse_csqa_answer,
    "ARITHMETIC": parse_gsm8k_answer,
    "GSM8K": parse_gsm8k_answer,
    "MATH500": parse_math500_answer,
    "GPQA": parse_gpqa_answer,
    "MMLUPro_Law": parse_gpqa_answer,
    "MMLUPro_Economics": parse_gpqa_answer,
    "MMLUPro_Math_Valid": parse_gpqa_answer,
}

def check_answers_consensus(pred_answer: str, true_answer: str, question_type: str) -> bool:

    if question_type in ["MMLU", "CSQA", "GPQA", "MMLUPro_Law", "MMLUPro_Economics", "MMLUPro_Math_Valid"]:
        pred_answer = pred_answer.upper()
        true_answer = true_answer.upper()
        return pred_answer == true_answer
    elif question_type in ["ARITHMETIC", "GSM8K"]:
        true_answer = re.findall(r"#### ([-+]?\d+\.\d*|[-+]?\d+)", true_answer)[0].strip()
        true_answer = float(true_answer)
        pred_answer = float(pred_answer)
        return abs(pred_answer - true_answer) < 1e-6
    elif question_type in ["MATH500", "MATH"]:
        return grade_answer(pred_answer, true_answer)
    else:
        raise ValueError("Unknown question_type: {} when check answer consensus.".format(question_type))
