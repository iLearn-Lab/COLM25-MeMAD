# start litellm: `litellm --config XXXXX/MeMAD/config/litellm_ollama_config.yaml`
# reference: https://docs.litellm.ai/docs/proxy/configs

import sys
from typing import List, Dict
import asyncio
import json
import argparse
from datetime import datetime

# sys.path.append("MeMAD folder Path")

from utils.utils import Logger, DataLoader
from utils.agent_memory import MemoryMAD_VectorDB
from utils.config import LLM_CONFIG, CONFIG
from utils.utils_agents import AgentId, Agent, DebateManager, Question


async def process_questions(debate_manager: DebateManager, question_type: str, is_training: bool,
                            questions: Dict, logger: Logger) -> None:

    print("=" * 100, f"question length: {len(questions)}", f"question type: {question_type}", "=" * 100)
    for question_id, item in questions.items():
        question_id = int(question_id)

        try:
            print(item)
            print("\n\n")
            question_content, true_answer, solution, category = DataLoader.format_question(item=item,
                                                                                    question_type=question_type)
            question = Question(content=question_content, question_id=question_id, answer=true_answer,
                                solution=solution, category=category)

            final_answer, is_final_correct = await debate_manager.debate(question=question, is_training=is_training)
            print("-" * 150)
            print(f"Question {question_id} final answer: {final_answer}, is correct: {is_final_correct}")
            print("-" * 150)
        except Exception as e:
            print(f"Exception while processing question {question_id}: {e}")
            logger.error(f"Error when processing question {question_id}: {str(e)}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run MemoryMAD")
    parser.add_argument("--question_type", type=str, default="GPQA", help="Question type")
    parser.add_argument("--models", type=str, default="gpt4omini;gpt4omini;gpt4omini", help="Model name")
    parser.add_argument("--if_train", action="store_true", help="Train or not")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    parser.add_argument("--if_use_memory", action="store_true", help="Use memory or not")
    parser.add_argument("--n_retrival", type=int, default=1, help="Number of retrival")
    parser.add_argument("--memory_type", type=str, default="PN", help="Memory type")
    parser.add_argument("--if_reflection", action="store_true", help="Use reflection or not")
    parser.add_argument("--log_name", type=str, default="", help="Log name")
    parser.add_argument("--force_max_round", action="store_true", help="Force max round")
    parser.add_argument("--memory_content_type", type=str, default="E", help="memory content type")
    parser.add_argument("--collection_name", type=str, default="", help="collection name")
    parser.add_argument("--same_sys_prompt", action="store_true", help="Same system prompt")
    parser.add_argument("--memory_db_type", type=str, default="MPN", help="Memory database type {MP,MN,MPN}")
    parser.add_argument("--embedding_model", type=str, default="bgem3", help="embedding model")
    parser.add_argument("--if_high", action="store_true", help="Using high level experience")
    parser.add_argument("--max_rounds", type=int, default=3, help="Number of max rounds.")
    args = parser.parse_args()
    return args


async def main(args: argparse.Namespace) -> None:
    print("=" * 160)
    print(args)
    print("=" * 160)

    model_names = [model_name.strip() for model_name in args.models.split(";")]
    print(model_names)

    if args.if_train:
        data_path = CONFIG["MEMORY_DATA_DIR"] / args.question_type
        data_path.mkdir(parents=True, exist_ok=True)

    if args.question_type == "MMLUPro_Math_Valid":
        collection_name = f"MATH500_{args.memory_content_type}{args.collection_name}"
    elif args.question_type in ["MMLUPro_Law", "GPQA", "MMLUPro_Economics", "MATH500"]:
        collection_name = f"{args.question_type}_{args.memory_content_type}{args.collection_name}_{args.embedding_model}"
    else:
        collection_name = f"{args.question_type}_{args.memory_content_type}{args.collection_name}"
    print(f"collection name: {collection_name}")

    memory_db = MemoryMAD_VectorDB(
        persistent_dir=str(CONFIG["PERSISTENT_DIR"]),
        collection_name=collection_name, verbose=True, model_name=args.embedding_model
    )

    log_path = (
            CONFIG["DATA_DIR"] / "logs" / "MMAD" / args.question_type /
            f"{args.log_name}_high{int(args.if_high)}_{args.embedding_model}_{args.memory_db_type}_agent{len(model_names)}same{int(args.same_sys_prompt)}_{args.memory_content_type}{args.collection_name}_{args.memory_type}_{int(args.if_use_memory)}memory{args.n_retrival}_{int(args.if_train)}train_{datetime.now().strftime('%Y%m%d%H%M')}.log"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(name="MemoryMAD", log_file=log_path)

    if args.verbose:
        print("Logger is initialized!")

    if args.verbose:
        print("MemoryDB is initialized")

    agents = {}
    for idx, model_name in enumerate(model_names):
        agent_id = AgentId(agent_type="Debater", agent_key=f"{idx}")
        agent = Agent(
            agent_type="Debater", agent_id=agent_id, client_config=LLM_CONFIG[model_name],
            memory_db=memory_db, n_retrival=args.n_retrival, memory_type=args.memory_type,
            if_use_memory=args.if_use_memory,
            logger=logger, question_type=args.question_type,
            calls_per_minute=CONFIG["API_RATE_LIMIT"], verbose=args.verbose,
            same_sys_prompt=args.same_sys_prompt, if_high=args.if_high
        )
        agents[agent_id] = agent

        logger.info(f"Agent {agent_id} based model {model_name} is initialized")

    if args.verbose:
        print("Agents are initialized!")

    debate_manager = DebateManager(
        agents=agents, wait_time=0.5,
        question_type=args.question_type, logger=logger, client_config=LLM_CONFIG['oqwen32b'],
        if_reflection=args.if_reflection,
        max_rounds=args.max_rounds, consensus_threshold=0.9, calls_per_minute=CONFIG["API_RATE_LIMIT"], verbose=args.verbose,
        force_max_round=args.force_max_round, memory_type=args.memory_type,
    )

    if args.verbose:
        print("DebateManager is initialized")

    data_loader = DataLoader()
    if args.if_train:
        questions = data_loader.load_dataset(args.question_type + "_TRAIN")
    else:
        questions = data_loader.load_dataset(args.question_type)

    await process_questions(debate_manager=debate_manager, question_type=args.question_type, questions=questions,
                            is_training=args.if_train, logger=logger)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(args=args))
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program terminated due to error: {str(e)}")
