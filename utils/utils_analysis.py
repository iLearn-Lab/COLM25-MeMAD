import re


def calculate_ACC(df):
    question_num = len(df['question_id'].unique())
    final_acc = len(df[df['is_correct'] == "True"]['question_id'].unique()) / question_num
    print(df['is_correct'].value_counts(normalize=False).sort_index())
    print(df['is_correct'].value_counts(normalize=False).sort_index() / question_num)
    print(f"\nQuestion Number: {question_num}, Acc: {final_acc:.4f}\n")
    return final_acc, question_num


def delate_error_memory(case_dir, question_ids):
    for json_file in case_dir.glob("*.json"):
        name = json_file.stem

        for qid in question_ids:
            if name.startswith(f"Q{qid}_"):
                json_file.unlink()


def delate_error_question_log(log_path, case_dir, question_ids):

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []

    is_deleting = False
    current_question_id = None

    i = 0
    while i < len(lines):
        current_line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if next_line:
            for qid in question_ids:
                if f"PROMPT of Agent Debater-0 for question {qid}:" in next_line:
                    is_deleting = True
                    current_question_id = qid
                    i += 1
                    break

        if not is_deleting:
            filtered_lines.append(current_line)

        else:
            if next_line and f"FINAL_ANSWER for question {current_question_id}:" in next_line:
                is_deleting = False
                current_question_id = None
                i += 1

        i += 1

    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print("Delate error log.")


def extract_log_messages(log_path):

    with open(log_path, 'r') as file:
        log_text = file.read()

    pattern = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.+?) - (\w+)\n([\s\S]*?)(?=(^\d{4}-\d{2}-\d{2}|\Z))',
        re.MULTILINE
    )

    matches = pattern.findall(log_text)

    messages = [match[3].strip() for match in matches]

    return messages


def extract_final_answer(input_string):
    pattern = r"^(?P<info_type>[A-Z_]+) for question (?P<question_id>\d+): (?P<pred_answer>.*). The final answer is (?P<is_correct>.*)$"
    match = re.match(pattern, input_string, re.DOTALL) 

    if match:
        return {
            "info_type": match.group("info_type"),
            "question_id": int(match.group("question_id")),
            "pred_answer": match.group("pred_answer").strip(), 
            "is_correct": match.group("is_correct").strip(),
        }
    else:
        raise ValueError("The input string format does not meet the requirements.")


def extract_agent_answer(input_string):
    if input_string.startswith("PRED_ANSWER"):
        pattern = r"^(?P<info_type>[A-Z_]+) of agent Debater-(?P<debater_id>\d+) for question (?P<question_id>\d+) on Round-(?P<round>\d+):(?P<pred_answer>.*)$"
        match = re.match(pattern, input_string, re.DOTALL)  
        if match:
            data = {
                "info_type": match.group("info_type"),
                "debater_id": int(match.group("debater_id")),
                "question_id": int(match.group("question_id")),
                "pred_answer": match.group("pred_answer").strip(),
            }
        else:
            print(input_string)
            raise ValueError("The input string format does not meet the requirements.")
    elif input_string.startswith("ROUND_ANSWER"):
        pattern = r"^(?P<info_type>[A-Z_]+) for question (?P<question_id>\d+) on Round-(?P<round>\d+):(?P<pred_answer>.*). The round answer is (?P<is_correct>.*)$"
        match = re.match(pattern, input_string, re.DOTALL) 
        if match:
            data = {
                "info_type": match.group("info_type"),
                "question_id": int(match.group("question_id")),
                "round": int(match.group("round")),
                "pred_answer": match.group("pred_answer").strip(),
                "is_correct": match.group("is_correct").strip(),
            }
        else:
            raise ValueError("The input string format does not meet the requirements.")
    elif input_string.startswith("IS_CORRECT"):
        pattern = r"^(?P<info_type>[A-Z_]+) of agent Debater-(?P<debater_id>\d+) for question (?P<question_id>\d+) on Round-(?P<round>\d+):(?P<is_correct>.*)$"
        match = re.match(pattern, input_string, re.DOTALL)  
        if match:
            data = {
                "info_type": match.group("info_type"),
                "debater_id": int(match.group("debater_id")),
                "question_id": int(match.group("question_id")),
                "is_correct": match.group("is_correct").strip(),
                "round": int(match.group("round")),
            }
        else:
            raise ValueError("The input string format does not meet the requirements.")
    else:
        raise ValueError("'info_type' is wrong")

    return data


def extract_error_info(input_string):
    pattern = r"^Error processing agent Debater-(?P<debater_id>\d+) in Round-(?P<round>\d+) for question (?P<question_id>\d+): (?P<error_info>.*)$"
    match = re.match(pattern, input_string, re.DOTALL) 

    if match:
        return {
            "debater_id": int(match.group("debater_id")),
            "round": int(match.group("round")),
            "question_id": int(match.group("question_id")), 
            "error_info": match.group("error_info").strip(),
        }
    else:
        raise ValueError("The input string format does not meet the requirements.")