import sys
import argparse
import chromadb
from datetime import datetime

# sys.path.append("MeMAD folder Path")

from utils.agent_memory import MemoryMAD_VectorDB
from utils.config import CONFIG
from utils.utils_memory import (
    read_memory_data,
    sample_memories,
    construct_memory,
    parse_db_id_time,
)
from utils.utils import DataLoader


def main(args: argparse.Namespace):
    if args.question_type in ["MMLUPro_Law", "GPQA", "MMLUPro_Economics", "MATH500"]:
        collection_name = f"{args.question_type}_{args.memory_content_type}{args.collection_name}_{args.embedding_model}"
    else:
        collection_name = f"{args.question_type}_{args.memory_content_type}{args.collection_name}"
    print(f"collection name: {collection_name}")


    client = chromadb.PersistentClient(path=str(CONFIG["PERSISTENT_DIR"]))
    collection_names = client.list_collections()
    if collection_name in collection_names:
        client.delete_collection(collection_name)

    memory_db = MemoryMAD_VectorDB(
        persistent_dir=str(CONFIG["PERSISTENT_DIR"]),
        collection_name=collection_name, verbose=True, model_name=args.embedding_model,
    )


    data_dir = CONFIG["MEMORY_DATA_DIR"] / args.question_type
    df = read_memory_data(data_dir)
    df['time_stamp'] = df['db_id'].map(lambda x: parse_db_id_time(x))

    data_loader = DataLoader()
    questions = data_loader.load_dataset(args.question_type + "_TRAIN")

    df_sample = sample_memories(df, args.sample_size, seed=10)

    for _, row in df_sample.iterrows():
        db_id, embedding_text, value_text, meta_data = construct_memory(row, memory_content_type=args.memory_content_type)

        if args.question_type == "MATH500":
            question_id = meta_data['question_id']
            meta_data["subject"] = questions[str(question_id)]['subject']
        elif args.question_type == "GPQA":
            question_id = meta_data['question_id']
            meta_data["domain"] = questions[str(question_id)]['domain']
        else:
            pass

        memory_db.add_memory(
            db_id=db_id,
            embedding_text=embedding_text,
            document_text=value_text,
            add_meta_datas=meta_data,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Memory for MemoryMAD")
    parser.add_argument("--question_type", type=str, default="MMLUPro_Economics", help="Question type")
    parser.add_argument("--sample_size", type=int, default=-1, help="Sample size")
    parser.add_argument("--collection_name", type=str, default="", help="collection name")
    parser.add_argument("--memory_content_type", type=str, default="QRE", help="memory content type")
    parser.add_argument("--embedding_model", type=str, default="bgem3", help="embedding model")
    args = parser.parse_args()

    print("=" * 160)
    print(args)
    print("=" * 160)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
