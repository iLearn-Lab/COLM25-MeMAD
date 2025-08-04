import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import asyncio
import random

from datetime import datetime

from collections import defaultdict, Counter

# sys.path.append("MeMAD folder Path")

from utils.utils import Logger, ModelClient, RateLimiter
from utils.agent_memory import MemoryMAD_VectorDB
from utils.prompts import MEMORY_PROMPT
from utils.prompts import EXTRACT_ANSWER_PROMPT, CHECK_ANSWER_PROMPT
from utils.prompts import MAD_PROMPT, REFLECTION_PROMPT
from utils.prompts import MAD_SYSTEM_PROMPT
from utils.mmad_prompts import MMAD_SYS_PROMPTS, TASK_PROMPT
from utils.config import CONFIG
from utils.parser import PARSE_ANSWER_FUNCS, check_answers_consensus
from utils.utils import write_json, read_json

from autogen_core.models import (
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)


import re

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

@dataclass
class Question:
    """
    format question
    """
    content: str 
    question_id: int 
    answer: str  
    solution: str 
    category: str

    def __str__(self):
        return f"The Question {self.question_id} is ::{self.content}"

    def __repr__(self):
        return f"The Question {self.question_id} is ::{self.content}"


@dataclass(frozen=True)
class AgentId:
    agent_type: str
    agent_key: str

    def __str__(self) -> str:
        return f"{self.agent_type}-{self.agent_key}"

    def __repr__(self) -> str:
        return f"{self.agent_type}-{self.agent_key}"


@dataclass
class RetrievedMemory:
    """
    format retrieved memory
    """
    content: str
    question_id: int
    agent_id: AgentId

    def __str__(self):
        return f"The Memory of {self.agent_id} for Question {self.question_id} is ::{self.content}"

    def __repr__(self):
        return f"The Memory of {self.agent_id} for Question {self.question_id} is ::{self.content}"


@dataclass
class Reflection:
    """
    format reflection
    """
    content: str
    question_id: int
    debate_round: int
    reflection_agent_id: AgentId
    check_agent_id: AgentId

    def __str__(self):
        result = (
            f"The Reflection of {self.reflection_agent_id} for {self.check_agent_id} on Question {self.question_id} "
            f"at round {self.debate_round} is ::{self.content}"
        )
        return result

    def __repr__(self):
        result = (
            f"The Reflection of {self.reflection_agent_id} for {self.check_agent_id} on Question {self.question_id} "
            f"at round {self.debate_round} is ::{self.content}"
        )
        return result


class Agent:

    def __init__(self,
                 agent_type: str,
                 agent_id: AgentId,
                 client_config: Dict,
                 memory_db: MemoryMAD_VectorDB,
                 logger: Logger,
                 question_type: str,
                 calls_per_minute: int = 60,
                 if_use_memory: bool = False,
                 n_retrival: Optional[int] = None,
                 memory_type: str = "PN",
                 verbose: bool = False,
                 same_sys_prompt: bool = False,
                 if_high: bool = False,
                 ) -> None:

        self.agent_type = agent_type
        self.agent_id = agent_id

        self.llm_config = client_config
        self.model_client = ModelClient(**client_config)
        self.rate_limiter = RateLimiter(calls_per_minute)

        self.memory_db = memory_db
        self.logger = logger

        self.question_type = question_type
        self.if_use_memory = if_use_memory
        self.n_retrival = n_retrival
        self.memory_type = memory_type
        self.same_sys_prompt = same_sys_prompt
        self.if_high = if_high
        self.round = 0

        self.high_experience = read_json(CONFIG["MEMORY_DATA_DIR"] / f"HighExperience" / f"high_experience.json")

        self.verbose = verbose

        if self.same_sys_prompt:
            self._system_message = [SystemMessage(content=MAD_SYSTEM_PROMPT[question_type])]
            logger.info(MAD_SYSTEM_PROMPT[question_type])
        else:
            self._system_message = [SystemMessage(content=MMAD_SYS_PROMPTS[question_type][int(agent_id.agent_key)])]
            logger.info(MMAD_SYS_PROMPTS[question_type][int(agent_id.agent_key)])
        self._task_messages: List[LLMMessage] = []

        self.current_question: Question = None
        self.current_response: LLMMessage = None

    async def _call_model(self, messages: List[LLMMessage]) -> LLMMessage:

        await self.rate_limiter.wait()
        model_result = await self.model_client.client.create(messages)
        self.logger.info(f"TOKENS usage for {self.agent_id}: {str(model_result.usage)}")
        return model_result

    def initialize_conversation(self, question: Question) -> None:
        self.clear_conversation()

        self.current_question = question

        if self.verbose:
            print(f"Agent {self.agent_id} restart conversation with new question")

    def clear_conversation(self) -> None:

        self.current_question = None
        self.current_response = None
        self._task_messages.clear()
        self.round = 0

    def _build_prompt(self,
                      retrieved_memories: Dict[str, List[RetrievedMemory]] = None,
                      pre_round_responses: List[str] = None,
                      ) -> str:

        task_prompt = ""

        if self.if_high and (self.round == 0):
            key = f"{self.question_type}_round{self.round % 3}_{self.agent_id}"
            high_experience = self.high_experience.get(key, "")

            if len(high_experience) > 0:
                task_prompt += "Here are some experiences that might help you answer more accurately:\n"
                task_prompt += f"{high_experience}\n\n"

        if retrieved_memories is not None:
            task_prompt += "Here are some examples:\n\n"
            task_prompt += "<examples>\n"

            count = 0
            for case_type, memories in retrieved_memories.items():
                if len(memories) > 0:
                    for idx, memory in enumerate(memories):
                        task_prompt += f'<example index="{count}">\n{memory.content}\n</example>\n'
                        count += 1
                else:
                    print(f"Warning: There are no memories for {case_type}")
                    log_text = f"WARNINF of Agent {self.agent_id} for question {self.current_question.question_id}: There are no memories for {case_type}"
                    self.logger.info(log_text)

                task_prompt += "\n"

            task_prompt += "</examples>\n\n"

        if pre_round_responses is not None:
            task_prompt += "These are the solutions to the problem from other agents:\n\n"
            task_prompt += "<other_solutions>\n"
            for idx, item in enumerate(pre_round_responses):
                task_prompt += f'<other_solution index="{idx}">\n{remove_think_tags(item)}\n</other_solution>\n'
            task_prompt += "</other_solutions>\n\n"
            task_prompt += MAD_PROMPT.format(self.current_question.content)

        if pre_round_responses is None:
            task_prompt += TASK_PROMPT[self.question_type].format(self.current_question.content)

        log_text = f"PROMPT of Agent {self.agent_id} for question {self.current_question.question_id}: {task_prompt}"
        self.logger.info(log_text)
        return task_prompt

    async def generate_response(self,
                                retrieved_memories: Dict[str, List[RetrievedMemory]] = None,
                                pre_round_responses: List[str] = None,
                                ) -> LLMMessage:

        task_prompt = self._build_prompt(retrieved_memories=retrieved_memories, pre_round_responses=pre_round_responses)
        self._task_messages.append(UserMessage(content=task_prompt, source="user"))
        response = await self._call_model(self._system_message + self._task_messages)
        self._task_messages.append(AssistantMessage(content=remove_think_tags(response.content), source=f"Detater{self.agent_id.agent_key}"))
        self.current_response = response
        self.round = self.round + 1

        return self.current_response

    async def generate_pre_response(self, pre_round_responses: List[str] = None) -> LLMMessage:

        task_prompt = self._build_prompt(pre_round_responses=pre_round_responses)
        _temp_messages = [UserMessage(content=task_prompt, source="user")]
        response = await self._call_model(self._system_message + self._task_messages + _temp_messages)
        return response

    async def reflect_on_response(self, text: str, is_self: bool, self_is_correct: bool = False) -> LLMMessage:

        if is_self:
            if self_is_correct:
                self_answer_type = "correct"
            else:
                self_answer_type = "incorrect"

            task_prompt = REFLECTION_PROMPT["SELF"]
            task_prompt = task_prompt.replace("{{question}}", self.current_question.content)
            task_prompt = task_prompt.replace("{{correct_solution}}", self.current_question.solution)
            task_prompt = task_prompt.replace("{{llm_response}}", text)
            task_prompt = task_prompt.replace("{{is_correct}}", self_answer_type)

        else:
            task_prompt = REFLECTION_PROMPT['OTHER']
            task_prompt = task_prompt.replace("{{question}}", self.current_question.content)
            task_prompt = task_prompt.replace("{{correct_solution}}", self.current_question.solution)
            task_prompt = task_prompt.replace("{{correct_llm_response}}", self.current_response.content)
            task_prompt = task_prompt.replace("{{incorrect_llm_response}}", text)

        _temp_messages = [UserMessage(content=task_prompt, source="user"), ]
        response = await self._call_model(self._system_message + self._task_messages + _temp_messages)
        if is_self:
            self_reflection_content = f"The following are some reflective points based on the previous round of answers, which might help improve the accuracy and quality of the next responses.\n\n{response.content}"
            self._task_messages.append(AssistantMessage(content=self_reflection_content, source=f"Detater{self.agent_id.agent_key}"))
        return response


    def retrival_memory(self, query_text: str, filter_metadata: Dict = None, dis: float = None) -> List[RetrievedMemory]:
        similar_pairs = self.memory_db.query_similar(
            query_text=query_text, n_results=self.n_retrival, filter_metadata=filter_metadata
        )

        if dis is not None:
            similar_pairs = [item for item in similar_pairs if item[2] < dis]

        memories = [
            RetrievedMemory(
                content=item[1], question_id=self.current_question.question_id, agent_id=self.agent_id
            ) for item in similar_pairs
        ]

        return memories

    def random_retrieval_memory(self, filter_metadata: Dict = None) -> List[RetrievedMemory]:
        db_ids = self.memory_db.get_all_ids(filter_metadata=filter_metadata)
        random.shuffle(db_ids)
        selected_db_ids = db_ids[:self.n_retrival]

        similar_pairs = self.memory_db.get_by_ids(select_ids=selected_db_ids)

        memories = [
            RetrievedMemory(
                content=item[1], question_id=self.current_question.question_id, agent_id=self.agent_id
            ) for item in similar_pairs
        ]

        return memories

    def add_memory(self, db_id: str, key_text: str, value_text: str, metadata: Dict = None) -> None:

        if metadata is None:
            metadata = {}

        self.memory_db.add_memory(
            db_id=db_id,
            embedding_text=key_text,
            document_text=value_text,
            add_meta_datas=metadata,
        )

    def get_current_question(self) -> Question:

        return self.current_question

    def get_current_response(self) -> LLMMessage:

        return self.current_response


class DebateManager:

    def __init__(self,
                 agents: Dict[AgentId, Agent],
                 question_type: str,
                 logger: Logger,
                 client_config: Dict,
                 if_reflection: bool = True,
                 memory_type: str = "PN",
                 max_rounds: int = 3,
                 wait_time: float = 5.0,
                 consensus_threshold: float = 0.8,
                 calls_per_minute: int = 60,
                 verbose: bool = True,
                 force_max_round: bool = True,
                 memory_db_type: str = "MPN",
                 ) -> None:
        self.agents = agents
        self.agent_ids = list(agents.keys())

        self.question_type = question_type
        self._parse_answer_func = PARSE_ANSWER_FUNCS[question_type]
        self.logger = logger

        self.model_client = ModelClient(**client_config)
        self.rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)

        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.if_reflection = if_reflection
        self.wait_time = wait_time
        self.force_max_round = force_max_round
        self.memory_type = memory_type
        self.memory_db_type = memory_db_type

        self.verbose = verbose

        self.question: Question = None
        self.agent_responses = defaultdict(dict)
        self.current_round = 0
        self.debate_history = []
        self.status = 'null'  # ["null", "initialized", "running", "finished"]

        self.final_answer = None

    async def _call_model(self, messages: List[LLMMessage]) -> LLMMessage:

        await self.rate_limiter.wait()
        model_result = await self.model_client.client.create(messages)
        self.logger.info(f"TOKENS usage for extract or check Answer: {str(model_result.usage)}")
        return model_result

    async def debate(self, question: Question, is_training: bool = True) -> Tuple[str, bool]:
        self.clear_debate()

        self.initialize_debate(question)

        if self.verbose:
            print("=" * 100, f"Debate is started!", "=" * 100)
            print("-" * 200)
            print(f"Question: {question.content}")
            print("-" * 200)

        if is_training:
            for _ in range(self.max_rounds):
                print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)
                await self.run_debate_round(is_training=is_training)
                print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)

            self.status = 'finished'
        else:

            if self.force_max_round:
                self.status = 'running'

                for _ in range(self.max_rounds):
                    print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)
                    await self.run_debate_round(is_training=is_training)
                    print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)

                self.status = 'finished'
            else:
                self.status = 'running'

                while (self.current_round < self.max_rounds) and (self.status == 'running'):
                    # 运行一轮辩论
                    print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)
                    await self.run_debate_round(is_training=is_training)
                    print("=" * 100, f"Round: {self.current_round}, Status: {self.status}", "=" * 100)

                self.status = 'finished'

        self.current_round -= 1

        correct_count, error_count = 0, 0
        final_true_answer, final_false_answer = None, None

        for agent_id in self.agent_ids:
            if self.agent_responses[agent_id][f"round_{self.current_round}"]["is_correct"]:
                correct_count += 1
                final_true_answer = self.agent_responses[agent_id][f"round_{self.current_round}"]["pred_answer"]
            else:
                error_count += 1
                final_false_answer = self.agent_responses[agent_id][f"round_{self.current_round}"]["pred_answer"]

        if self.verbose:
            print("=" * 50, f"Round: {self.current_round}, Status: {self.status}, correct: {correct_count}, error: {error_count}", "=" * 50)

        is_final_correct = (correct_count > error_count)
        if correct_count > error_count:
            self.final_answer = final_true_answer
        else:
            self.final_answer = final_false_answer

        if self.status == 'running':
            self.status = 'finished'

        if self.verbose:
            print("=" * 200)
            print(f"Final Answer: {self.final_answer}; And the final answer is {is_final_correct}")
            print("=" * 100, f"Debate is finished!", "=" * 100)

        log_text = f"FINAL_ANSWER for question {self.question.question_id}: {self.final_answer}. The final answer is {is_final_correct}"
        self.logger.info(log_text)

        return self.final_answer, is_final_correct

    def retrival_memories(self, query_text: str, agent_id: AgentId):

        filter_conditions = [
            {"round": {"$eq": self.current_round}},
            {"agent_id": {"$eq": f"{agent_id.agent_type}-{int(agent_id.agent_key) % 3}"}},
        ]

        if self.question_type == "GPQA":
            filter_conditions.append(
                {"domain": {"$eq": self.question.category}}
            )

        if self.memory_type == "PN":
            correct_filter_metadata = {
                "$and": [
                    {"response_correct": {"$eq": "True"}},
                    {"round": {"$eq": self.current_round}},
                    {"agent_id": {"$eq": f"{agent_id.agent_type}-{int(agent_id.agent_key) % 3}"}},
                ]
            }
            correct_retrieved_memories = self.agents[agent_id].retrival_memory(query_text,
                                                                               filter_metadata=correct_filter_metadata)

            error_filter_metadata = {
                "$and": [
                    {"response_correct": {"$eq": "False"}},
                    {"round": {"$eq": self.current_round}},
                    {"agent_id": {"$eq": f"{agent_id.agent_type}-{int(agent_id.agent_key) % 3}"}},
                ]
            }
            error_retrieved_memories = self.agents[agent_id].retrival_memory(query_text,
                                                                             filter_metadata=error_filter_metadata)

            retrieved_memories = {
                "positive": correct_retrieved_memories,
                "negative": error_retrieved_memories,
            }
        elif self.memory_type == "RANDOM":

            if self.memory_db_type == "MP":
                filter_conditions.append(
                    {"response_correct": {"$eq": "True"}}
                )
            elif self.memory_db_type == "MN":
                filter_conditions.append(
                    {"response_correct": {"$eq": "False"}}
                )
            else:
                pass

            filter_metadata = {
                "$and": filter_conditions
            }

            random_retrieved_memories = self.agents[agent_id].random_retrieval_memory(filter_metadata=filter_metadata)
            retrieved_memories = {
                "random": random_retrieved_memories,
            }
        elif self.memory_type == "SIMILAR":

            if self.memory_db_type == "MP":
                filter_conditions.append(
                    {"response_correct": {"$eq": "True"}}
                )
            elif self.memory_db_type == "MN":
                filter_conditions.append(
                    {"response_correct": {"$eq": "False"}}
                )
            else:
                pass

            filter_metadata = {
                "$and": filter_conditions
            }
            similar_retrieved_memories = self.agents[agent_id].retrival_memory(query_text,
                                                                             filter_metadata=filter_metadata)
            retrieved_memories = {
                "similar": similar_retrieved_memories,
            }
        elif self.memory_type == "Diversity":
            if self.memory_db_type == "MP":
                categories = random.sample(range(0, 16), self.agents[agent_id].n_retrival)
            elif self.memory_db_type == "MN":
                categories = random.sample(range(16, 22), self.agents[agent_id].n_retrival)
            else:
                categories = random.sample(range(0, 22), self.agents[agent_id].n_retrival)

            similar_retrieved_memories = []

            if self.memory_db_type == "MPN":
                for category in categories:
                    filter_conditions = [
                        {"round": {"$eq": self.current_round}},
                        {"agent_id": {"$eq": f"{agent_id.agent_type}-{int(agent_id.agent_key) % 3}"}},
                        {"all_category": {"$eq": category}}
                    ]
                    filter_metadata = {
                        "$and": filter_conditions
                    }
                    category_retrieved_memories = self.agents[agent_id].retrival_memory(query_text,
                                                                               filter_metadata=filter_metadata)
                    similar_retrieved_memories.append(category_retrieved_memories[0])
            else:
                for category in categories:
                    filter_conditions = [
                        {"round": {"$eq": self.current_round}},
                        {"agent_id": {"$eq": f"{agent_id.agent_type}-{int(agent_id.agent_key) % 3}"}},
                        {"pn_category": {"$eq": category}}
                    ]
                    filter_metadata = {
                        "$and": filter_conditions
                    }
                    category_retrieved_memories = self.agents[agent_id].retrival_memory(query_text,
                                                                               filter_metadata=filter_metadata)
                    similar_retrieved_memories.append(category_retrieved_memories[0])

            retrieved_memories = {
                "diversity": similar_retrieved_memories,
            }

        else:
            raise RuntimeError(f"Unknown memory type: {self.memory_type}")

        return retrieved_memories

    async def process_agent(self, agent_id: AgentId, is_training: bool,
                            pre_round_responses: List[str]) -> None:
        try:
            if self.verbose:
                print("-" * 200)
                print("-" * 50, f"Question: {self.question.question_id}, Round-{self.current_round} Agent-{agent_id} is started!")

            retrieved_memories = None
            if not is_training and self.agents[agent_id].if_use_memory: 
                query_text = ""
                query_text += f"{self.question.content}\n"  # Question

                if self.current_round > 0:
                    self_pre_round_response = self.agent_responses[agent_id][f"round_{self.current_round - 1}"]["response"]
                    self_pre_round_response = remove_think_tags(self_pre_round_response)
                    query_text += f"{self_pre_round_response}"

                if self.verbose:
                    print("-" * 50, f"Round-{self.current_round} Agent-{agent_id}: Retrieved Memory finished!")

                retrieved_memories = self.retrival_memories(query_text, agent_id)

            response = await self.agents[agent_id].generate_response(
                retrieved_memories=retrieved_memories, pre_round_responses=pre_round_responses
            )
            log_text = f"RESPONSE of agent {agent_id} for question {self.question.question_id} on Round-{self.current_round}: {response.content}"
            self.logger.info(log_text)

            pred_answer = await self.parse_answer(response.content)
            log_text = f"PRED_ANSWER of agent {agent_id} for question {self.question.question_id} on Round-{self.current_round}: {pred_answer}"
            self.logger.info(log_text)
            is_correct = await self.check_answer_correct(pred_answer)
            log_text = f"IS_CORRECT of agent {agent_id} for question {self.question.question_id} on Round-{self.current_round}: {is_correct}"
            self.logger.info(log_text)

            self.agent_responses[agent_id][f"round_{self.current_round}"] = {
                "response": response.content,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "reflections": [],
            }

            if self.verbose:
                print(
                    "-"*50,
                    f"Question: {self.question.question_id}, Round-{self.current_round} Agent-{agent_id} pred_answer: {pred_answer}, is_correct: {is_correct}",
                )

            if is_training and self.if_reflection:
                self_reflection = await self.agents[agent_id].reflect_on_response(response.content, is_self=True,
                                                                                  self_is_correct=is_correct)
                self.agent_responses[agent_id][f"round_{self.current_round}"]['reflections'].append(self_reflection.content)
                log_text = f"SELF_REFLECTION of agent {agent_id} for question {self.question.question_id} on Round-{self.current_round}: {self_reflection.content}"
                self.logger.info(log_text)
                if self.verbose:
                    print("-" * 50, f"Round-{self.current_round} Agent-{agent_id} reflection on Self")

            if self.verbose:
                print("-" * 200)
        except Exception as e:
            if self.verbose:
                print(f"Error processing agent {agent_id} in Round-{self.current_round} for question {self.question.question_id}: {str(e)}")
            self.logger.error(f"Error processing agent {agent_id} in Round-{self.current_round} for question {self.question.question_id}: {str(e)}")
            self.agent_responses[agent_id][f"round_{self.current_round}"] = {
                "response": "",
                "pred_answer": "",
                "is_correct": False,
                "reflections": [],
            }

    async def process_reflection(self, correct_agent_id, error_agent_id, error_response):
        try:
            other_reflection = await self.agents[correct_agent_id].reflect_on_response(error_response, is_self=False)
            self.agent_responses[error_agent_id][f"round_{self.current_round}"]["reflections"].append(
                other_reflection.content)
            other_reflection_content = f"The following is the reflection content of the correct agent regarding the previous round's answer, which may help provide better and more accurate responses in the future.\n\n{other_reflection.content}"
            self.agents[error_agent_id]._task_messages.append(
                AssistantMessage(content=other_reflection_content, source=f"Debater{correct_agent_id.agent_key}")
            )
            log_text = f"OTHER_REFLECTION of correct agent {correct_agent_id} for error agent {error_agent_id} for question {self.question.question_id} on Round-{self.current_round}: {other_reflection.content}"
            self.logger.info(log_text)

            if self.verbose:
                print("-"*50, f"reflection from agent {correct_agent_id} to agent {error_agent_id}", "-" * 50)

        except Exception as e:
            if self.verbose:
                print(f"Error processing reflection from agent {correct_agent_id} to agent {error_agent_id}: {str(e)}")
            self.logger.error(
                f"Error processing reflection from agent {correct_agent_id} to agent {error_agent_id}: {str(e)}")

    async def run_debate_round(self, is_training: bool = True) -> None:

        if self.verbose:
            print("=" * 50, f"Question:{self.question.question_id}, Round-{self.current_round} is started!")

        for agent_id in self.agent_ids:

            if self.current_round == 0:
                pre_round_responses = None
            else:

                pre_round_responses = []
                for other_agent_id in self.agent_ids:
                    if agent_id != other_agent_id:
                        other_agent_response = self.agent_responses[other_agent_id][f"round_{self.current_round - 1}"]["response"]
                        if len(other_agent_response) > 0:
                            pre_round_responses.append(other_agent_response)

            await self.process_agent(agent_id, is_training, pre_round_responses)
            await asyncio.sleep(self.wait_time)
        if is_training and self.if_reflection:
            correct_agent_ids = []
            error_agent_ids = []

            for agent_id in self.agent_ids:
                if self.agent_responses[agent_id][f"round_{self.current_round}"]["is_correct"]:
                    correct_agent_ids.append(agent_id)
                else:
                    error_agent_ids.append(agent_id)

            if (len(correct_agent_ids) >= 1) and (len(error_agent_ids) >= 1):
                # feedback_tasks = []
                for correct_agent_id in correct_agent_ids:
                    for error_agent_id in error_agent_ids:
                        error_response = self.agent_responses[error_agent_id][f"round_{self.current_round}"]["response"]

                        if len(error_response) > 0:
                            await self.process_reflection(correct_agent_id, error_agent_id, error_response)
                            await asyncio.sleep(self.wait_time)
                # await asyncio.gather(*feedback_tasks)

        if is_training and self.if_reflection:
            for agent_id in self.agent_ids:
                agent_response = self.agent_responses[agent_id][f'round_{self.current_round}']['response']

                if len(agent_response) > 0:
                    db_id = f"Q{self.question.question_id}_Round{self.current_round}_{agent_id}_{datetime.now().strftime('%m%d%H%M%S')}"
                    memory_infos = dict()

                    memory_infos["round"] = self.current_round

                    memory_infos["agent_id"] = str(agent_id)

                    memory_infos["question_id"] = self.question.question_id
                    memory_infos['question'] = self.question.content
                    memory_infos['answer'] = self.question.answer
                    memory_infos['solution'] = self.question.solution

                    if self.current_round > 0:
                        pre_round_responses = dict()
                        for pre_agent_id in self.agent_ids:
                            pre_round_responses[str(pre_agent_id)] = self.agent_responses[pre_agent_id][f'round_{self.current_round-1}']['response']
                        memory_infos["pre_round_responses"] = pre_round_responses

                    memory_infos["current_response"] = self.agent_responses[agent_id][f'round_{self.current_round}']['response']
                    memory_infos["current_response_correct"] = str(self.agent_responses[agent_id][f'round_{self.current_round}']['is_correct'])

                    assert len(self.agent_responses[agent_id][f'round_{self.current_round}']['response']) > 0, "No reflection when store memory."
                    memory_infos["self_reflection"] = self.agent_responses[agent_id][f'round_{self.current_round}']['reflections'][0]
                    memory_infos["other_reflection"] = self.agent_responses[agent_id][f'round_{self.current_round}']['reflections'][1:]

                    write_json(memory_infos, CONFIG["MEMORY_DATA_DIR"] / self.question_type / f"{db_id}.json")

        pred_answers = []
        for agent_id in self.agent_ids:
            pred_answers.append(self.agent_responses[agent_id][f"round_{self.current_round}"]["pred_answer"])

        is_consensus, most_common_answer, consensus_ratio = self.check_consensus(pred_answers)

        is_correct = await self.check_answer_correct(most_common_answer)
        log_text = f"ROUND_ANSWER for question {self.question.question_id} on Round-{self.current_round}: {most_common_answer}. The round answer is {is_correct}"
        self.logger.info(log_text)

        self.current_round += 1

        if is_consensus:
            self.status = 'finished'
            self.final_answer = most_common_answer
        else:
            if self.current_round == self.max_rounds:
                self.status = 'finished'

    async def parse_answer(self, text: str) -> str:
        answer = self._parse_answer_func(text)

        if (len(answer) == 0) or ("answer" in answer) or (len(answer) >= 20):
            answer = await self._extract_answer_by_llm(text)

        return answer

    async def check_answer_correct(self, pred_answer: str) -> bool:
        true_answer = self.question.answer

        rule_check_result = check_answers_consensus(pred_answer, true_answer, self.question_type)
        if rule_check_result:
            return True

        final_check_result = False

        if not rule_check_result:
            llm_check_result = await self._check_answer_by_llm(pred_answer, true_answer)

            if llm_check_result:
                final_check_result = True

        return final_check_result

    async def _check_answer_by_llm(self, pred_answer: str, true_answer: str) -> bool:
        check_prompt = CHECK_ANSWER_PROMPT
        check_prompt = check_prompt.replace("{{true_answer}}", true_answer)
        check_prompt = check_prompt.replace("{{pred_answer}}", pred_answer)

        response = await self._call_model(messages=[
            UserMessage(content=check_prompt, source="user"),
        ])
        result = response.content
        result = result.strip().lower()

        if result == "equivalent":
            return True
        elif result == "not equivalent":
            return False
        else:
            if self.verbose:
                print(f"Check answer by LLM failed! The response is {response.content}")
            return False

    async def _extract_answer_by_llm(self, solution_text: str) -> str:

        extract_prompt = EXTRACT_ANSWER_PROMPT[self.question_type]
        extract_prompt = extract_prompt.replace("{{response}}", solution_text)

        response = await self._call_model(messages=[
            UserMessage(content=extract_prompt, source="user"),
        ])
        return response.content

    def check_consensus(self, pred_answers: List[str]) -> Tuple[bool, str, float]:

        if not pred_answers:  
            print("pred_answers is empty when check_consensus!")
            return False, "", 0.0

        normalized_answers = [answer.lower().strip() for answer in pred_answers]

        correct_count = 0
        for agent_id in self.agent_ids:
            if self.agent_responses[agent_id][f"round_{self.current_round}"]['is_correct']:
                correct_count += 1
        if correct_count == len(pred_answers):
            return True, normalized_answers[0], 1.0

        if self.verbose:
            print("="*100, "normalized_answers: ", normalized_answers, "="*100)

        answer_counts = Counter(normalized_answers)

        most_common_answer, max_count = answer_counts.most_common(1)[0]

        consensus_ratio = max_count / len(pred_answers)

        is_consensus = consensus_ratio >= self.consensus_threshold

        if self.verbose:
            print("=" * 100, f"current round {self.current_round}, answer: {most_common_answer}, consensus_ratio: {consensus_ratio}", "=" * 100)

        return is_consensus, most_common_answer, consensus_ratio

    def initialize_debate(self, question: Question) -> None:
        self.question = question

        for agent_id in self.agent_ids:
            self.agents[agent_id].initialize_conversation(question)
        self.status = "initialized"

    def clear_debate(self) -> None:

        self.question = None
        self.current_round = 0

        self.debate_history.clear()
        self.agent_responses.clear()
        self.status = 'null'
