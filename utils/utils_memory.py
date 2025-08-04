import re
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
import chromadb

# sys.path.append("MeMAD folder Path")

from utils.agent_memory import MemoryMAD_VectorDB
from utils.utils import read_json
from utils.config import CONFIG


def read_memory_data(data_dir: Path) -> pd.DataFrame:
    """
    :param data_dir: Path
    :return: pd.DataFrame
    """
    all_data = {}

    for json_file in data_dir.glob("*.json"):
        file_name = json_file.stem

        file_content = read_json(json_file)

        all_data[file_name] = file_content

    df = pd.DataFrame(all_data).T
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "db_id"})
    df.sort_values("db_id", inplace=True)

    return df


def sample_memories(df: pd.DataFrame, sample_size: int = -1, seed: int = 10) -> pd.DataFrame:

    np.random.seed(seed)
    random.seed(seed)

    question_ids = sorted(df['question_id'].unique().tolist())
    random.shuffle(question_ids)

    assert isinstance(sample_size, int), "sample_size must be an integer"
    if (sample_size > len(question_ids)) or (sample_size == -1):
        sample_size = len(question_ids)
        print("Total memory data are sampled.")

    select_question_ids = question_ids[:sample_size]
    df_select = df[df['question_id'].isin(select_question_ids)]

    return df_select


def construct_memory(row, memory_content_type="QRE"):
    meta_data = dict()
    embedding_text = ""
    value_text = ""
    db_id = row['db_id']

    embedding_text += f"{row['question']}\n"
    
    if row['pre_round_responses'] is np.nan:
        embedding_text += ""
    else:
        agent_list = sorted(row['pre_round_responses'].keys())
        for agent_id in agent_list:
            if agent_id == row['agent_id']:
                embedding_text += f"{row['pre_round_responses'][agent_id]}\n"

    if memory_content_type == "QRE":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_agent_solution>\n{row['current_response']}\n</example_agent_solution>\n\n"

        if row['current_response_correct'] == "True":
            text = "The solution is correct."
        else:
            text = "The solution is incorrect."
        value_text += f"<example_agent_solution_correctness>\n{text}\n</example_agent_solution_correctness>\n\n"

        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    elif memory_content_type == "QS":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"
    elif memory_content_type == "E":
        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    elif memory_content_type == "QE":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    elif memory_content_type == "QSE":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"

        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    elif memory_content_type == "QSRE":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"
        value_text += f"<example_agent_solution>\n{row['current_response']}\n</example_agent_solution>\n\n"

        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    elif memory_content_type == "QSRSR":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"
        value_text += f"<example_agent_solution>\n{row['current_response']}\n</example_agent_solution>\n\n"
        value_text += f"<example_agent_self_reflection>\n{row['self_reflection']}\n</example_agent_self_reflection>\n\n"
    elif memory_content_type == "QSRPR":
        value_text += f"<example_question>\n{row['question']}\n</example_question>\n\n"
        value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"
        value_text += f"<example_agent_solution>\n{row['current_response']}\n</example_agent_solution>\n\n"
        if len(row['other_reflection']) > 0:
            for other_reflection in row['other_reflection']:
                value_text += f"<example_other_agent_reflection>\n{other_reflection}\n</example_other_agent_reflection>\n\n"
    else:
        print("Unknown memory content type: {memory_content_type}")
        raise RuntimeError(f"Unknown memory type: {memory_content_type}")

    meta_data['question_id'] = row['question_id']
    meta_data['round'] = row['round']
    meta_data['agent_id'] = row['agent_id']
    meta_data['response_correct'] = row['current_response_correct']

    return db_id, embedding_text, value_text, meta_data


def parse_db_id_time(db_id: str, default_year: int = 2025) -> datetime:
    """
    Parse the db_id string in format 'Q{question_id}_Round{round_num}_{agent_id}_{timestamp}'
    Example input: 'Q1_Round2_agent3_120514302'
    
    Args:
        db_id (str): The db_id string to parse
        default_year (int): Default year to use for timestamp (default: 2025)
        
    Returns: timestamp: datetime object
    """
    pattern = r'Q(\d+)_Round(\d+)_([^_]+)_(\d{10})'
    match = re.match(pattern, db_id)
    
    if not match:
        raise ValueError(f"Invalid db_id format: {db_id}")
    
    question_id, round_num, agent_id, timestamp_str = match.groups()

    month = int(timestamp_str[0:2])
    day = int(timestamp_str[2:4])
    hour = int(timestamp_str[4:6])
    minute = int(timestamp_str[6:8])
    second = int(timestamp_str[8:10])
    
    timestamp = datetime(default_year, month, day, hour, minute, second)
    
    return timestamp
