SELF_REFLECTION = """You are tasked with analyzing a problem-solving example and generating transferable insights for future improvement on solving similar problems.
Given Information:
<question>
{{question}}
</question>

<correct_solution>
{{correct_solution}}
</correct_solution>

<llm_response>
{{llm_response}}
</llm_response>

<response_correctness>
Response Correctness is {{is_correct}}
</response_correctness>

Analysis Framework:
1. Compare the solutions:
   - Identify similarities and differences in approach
   - Analyze which elements worked better and why
   - Note any efficiency or clarity advantages
2. If correct:
   - Identify successful reasoning patterns
   - Extract key decision points that led to success
3. If incorrect:
   - Identify the gap between response and correct answer
   - Analyze where the reasoning went wrong
   - Determine what knowledge or step was missing

Output Requirements:
Generate exactly 3 key learned insights in this format:
"Key Learning #[number]: [specific insight]
Application: [how to apply this learning to future problems]"

Each learned insight must be:
- Generalizable (applicable to similar problems)
- Specific (clear action or thinking strategy)
- Concise (one sentence for insight, one for application)
- Focus on problem-solving strategies only

Note: 
- Do not restate the specific problem content or solution.
- Only output the learned insights.
"""

OTHER_REFLECTION = """You are tasked with analyzing an incorrect LLM response by comparing it with both the standard solution and a correct LLM response, to generate generalizable insights that can help LLMs better solve similar problems in the future.

Given Information:
<question>
{{question}}
</question>

<correct_solution>
{{correct_solution}}
</correct_solution>

<correct_llm_response>
{{correct_llm_response}}
</correct_llm_response>

<incorrect_llm_response>
{{incorrect_llm_response}}
</incorrect_llm_response>

Analysis Framework:
1. Error Pattern Analysis:
   - Identify where incorrect response deviates from both correct approaches
   - Analyze the root causes of these deviations
   - Detect patterns of misconceptions or flawed reasoning

2. Success Pattern Recognition:
   - Study how correct LLM response aligns with standard solution
   - Identify key elements missing in incorrect response
   - Extract successful reasoning patterns and approaches

3. Improvement Opportunities:
   - Pinpoint specific areas where incorrect response could be enhanced
   - Identify critical checkpoints that could prevent similar errors
   - Formulate strategies to bridge the gap between incorrect and correct approaches

Output Requirements:
Generate exactly 3 transferable insights in this format:
"Learning Point #[number]: [specific insight]
Strategic Application: [concrete strategy for future problem-solving]"

Each insight must be:
- Focused on preventing similar errors in future
- Strategy-focused (not problem-specific)
- Action-oriented
- Clearly articulated in 1-2 sentences

Note: 
- Emphasize practical strategies for enhancement
- Ensure insights are applicable to future problem-solving
- Avoid repeating specific problem details or solutions
- Emphasize methodological improvements rather than content knowledge
- Only output the learned insights.
"""


REFLECTION_PROMPT = {
    "SELF": SELF_REFLECTION,
    "OTHER": OTHER_REFLECTION,
}

MEMORY_PROMPT = {
    "QAR": (
        "I will give you some examples. Each example includes three parts: 'Question', 'Answer', and 'Agent Solution'. "
        "Please note that the 'Answer' is always correct, while the 'Agent Solution' may not be. "
        "Learn from these examples and think critically, then answer the question below. Here are some examples:\n"
    ),
    "PN": (
        "I will provide you with six examples, each consisting of three parts: 'Question', 'Answer', and 'Agent Solution'. "
        "In these examples, the 'Answer' is always correct. The 'Agent Solution' may either be correct or incorrect, and I will indicate whether the Agent Solution is correct (positive example) or incorrect (negative example). "
        "Note that for negative examples, I will not specify the exact errors in the Agent Solution. Your task is to critically analyze the examples, identify patterns or reasoning strategies from both correct (positive) and incorrect (negative) solutions, and learn from them. "
        "After analyzing these examples, apply the insights you have gained to solve the new question below. Think critically and ensure your solution is both accurate and well-reasoned."
    ),
    "PSR": (
            "I will provide several examples with the following structure:\n\nExample Components:\n1. <example_question>: The problem to be solved\n"
            "2. <example_agent_solution>: The agent's answer to the question\n3. <example_agent_solution_correctness>: Whether the agent's answer is correct or not\n"
            "4. <example_agent_self_reflection>: The agent's reflection on its own answer\n5. [Optional] <example_other_agent_reflection>: Reflections from agents who answered correctly on the incorrect solutions provided by other agents\n\n"
            "Analyze the provided examples to identify common patterns, effective reasoning strategies, and key problem-solving approaches. Study both correct and incorrect solutions carefully, along with their corresponding reflections, to understand successful approaches and common pitfalls. "
            "Learn from the self-reflections and peer feedback, particularly focusing on why certain solutions succeeded while others failed. Apply all these insights to solve the new question below, ensuring your solution demonstrates critical thinking, logical reasoning, and accurate conclusions with clear justification. "
        )
}

EXTRACT_ANSWER_PROMPT = {
    "MATH500": (
        "Given a language model's response text below, extract the final answer or conclusion. "
        "The response text starts with ```response:``` and contains the complete solution or explanation. "
        "Guidelines for extraction:\n1. Examine the response text, particularly focusing on:\n"
        "- The final lines or conclusion section\n- Content marked with \boxed{}, (( )), or similar delimiters\n"
        "- Statements following phrases like `the final answer is`, `therefore`, `thus`\n"
        "2. Extract rules:\n- Include only the final result/answer\n- Keep the original mathematical notation and symbols\n"
        "- Preserve all components if the answer has multiple parts\n- Remove decorative markers but retain the content\n"
        "3. Format your output as follows:\n```[extracted answer]```\nExample:\nIf the response contains:\n"
        "`...after calculations, the final answer is \boxed{x = 5}`, Output should be: ```x = 5```\n"
        "Extract only the answer, excluding all explanation and intermediate steps.\n"
        "Here is the response:\n<response>\n{{response}}\n</response>"
    ),
}

CHECK_ANSWER_PROMPT = (
    "You are a mathematics expert skilled in verifying the equivalence of mathematical expressions. "
    "Given two mathematical answers, strictly determine whether they are mathematically equivalent.\n"
    "Key Rules:\n1. Ignore unnecessary parentheses, spaces, formatting differences, or case sensitivity, but ensure the mathematical meaning is accurate.\n"
    "2. Treat all mathematical constants (π, e, etc.) as exact symbolic values, NOT their decimal approximations\n"
    "3. Never use numerical approximations in evaluation (e.g., π ≈ 3.14 is forbidden)\n"
    "4. Consider expressions equivalent ONLY if they are algebraically identical after simplification\n"
    "5. Keep all irrational numbers and roots in their exact symbolic form\n"
    "6. If you are uncertain about the equivalence, respond with 'Not Equivalent'\n"
    "7. Only respond with 'Equivalent' or 'Not Equivalent'. Do not include any additional explanations or output.\n\n"
    'Only respond with "Equivalent" or "Not Equivalent." Do not include any additional explanations or output.\n'
    "Examples:\n1. Input: `A` and `(A)` → Output: Equivalent\n"
    "2. Input: `(3, \\frac{\\pi}{2})` and `((3, π/2))` → Output: Equivalent\n"
    "3. Input: `((p - q))` and `p - q` → Output: Equivalent\n"
    "4. Input: `A + B` and `A - B` → Output: Not Equivalent\n"
    "5. Input: `(3, \\frac{1}{2})` and `\\boxed{((3, 0.5))}` → Output: Equivalent\n"
    "6. Input: `π` and `3.14159` → Output: Not Equivalent\n"
    "7. Input: `sqrt(2)` and `1.4142` → Output: Not Equivalent\n\n"
    "Task:\nDetermine if the following two answers are equivalent: \n`{{true_answer}}` and `{{pred_answer}}`"
)

MAD_SYSTEM_PROMPT = {
    "MATH500": (
        "You are an expert mathematics instructor with deep knowledge across all areas of mathematics including algebra, geometry, calculus, probability, and number theory. "
        "Provide the final answer in double parentheses at the end of your response : ((answer))"
    ),
    "GPQA": (
        "You are a scientific expert with deep knowledge in physics, chemistry, and biology."
        "Provide the final answer in double parentheses at the end of your response : ((answer)), where answer is A, B, C, or D."
    ),
    "MMLUPro_Law": (
        "You're an expert on the law."
        "Provide the final answer in double parentheses at the end of your response : ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J"
    ),
    "MMLUPro_Economics": (
        "You're an expert on economics."
        "Provide the final answer in double parentheses at the end of your response : ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J"
    ),
    "MMLUPro_Math_Valid": (
        "You're an expert on mathematics."
        "Provide the final answer in double parentheses at the end of your response : ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J"
    )
}

MAD_PROMPT = (
    "Use the solutions from other agents as additional information, can you give an updated answer?"
    "The original question is: \n\n<question>\n{}\n</question>\n\nProvide the final answer in double parentheses at the end of your response : ((answer))"
)

SYSTEM_PROMPT = {
    "MATH500": (
        "You are an expert mathematics instructor with deep knowledge across all areas of mathematics including algebra, geometry, calculus, probability, and number theory. "
        "Your role is to solve mathematical problems by: Break down the problem into clear, logical steps; "
        "Skip trivial or obvious steps to maintain brevity; Use precise mathematical notation and terminology;"
        "Show key calculations and reasoning concisely; "
        "Provide the final answer in double parentheses at the end of your response : ((answer))"
    ),
    "GPQA": (
        """You are a scientific expert with deep knowledge in physics, chemistry, and biology. Your role is to:
- Analyze multiple-choice questions accurately
- Provide concise explanations for your answers
- Follow strict answer formatting rules
- Maintain scientific accuracy and objectivity

Always format your final answer with double parentheses ((X)) where X is A, B, C, or D."""
    ),
    "MMLUPro_Law": (
        """You are a legal expert and exam assessor specialized in law. Your role is to:
1. Carefully analyze legal multiple-choice questions
2. Apply legal principles and knowledge
3. Provide clear, concise reasoning
4. Select only one correct answer
5. End the response with the answer in double parentheses ((X)), where X can be A, B, C, D, E, F, G, H, I, or J

Always maintain professional judgment and accuracy in legal analysis."""
    ),
    "MMLUPro_Economics": (
        """You are an expert economics consultant with deep knowledge in economic theories, principles, and their practical applications. Your task is to analyze economics-related multiple-choice questions and provide well-reasoned answers. Please:
1. Read the question carefully
2. Analyze key economic concepts involved
3. Evaluate each option systematically
4. Provide concise explanation for your choice
5. End the response with the answer in double parentheses ((X)), where X can be A, B, C, D, E, F, G, H, I, or J

Ensure all responses are clear, accurate, and focused on economic principles."""
    )
}

TASK_PROMPT = {
    "MATH500": (
        "Can you solve the following math question as accurately as possible?\n\n<question>\n{}\n</question>\n\nExplain your reasoning concisely. "
        "Provide the final answer in double parentheses at the end of your response : ((answer))"
    ),
    "GPQA": (
        "Please analyze the following multiple-choice question and solve it as accurately as possible. \n\n<question>\n{}\n</question>\n\n"
        "Requirements:\n1. Provide a brief analysis (max 2 sentences)\n2. Explain your reasoning (max 5 sentences)\n3. State your final answer in double parentheses at the end of your response\n\nKeep total response under 200 words."
    ),
    "MMLUPro_Law": (
        "Please analyze the following legal multiple-choice question:\n\n<question>\n{}\n</question>\n\nProvide a brief analysis and select the correct answer. Follow this structure:\n"
        "1. Key legal principle involved\n2. Critical analysis (2-3 sentences)\n3. Conclusion with the answer in double parentheses at the end"
    ),
    "MMLUPro_Economics": (
        "\n\n<question>\n{}\n</question>\n\nPlease analyze this economics question and provide:\n1. Key concept identification\n2. Brief analysis (2-3 sentences)\n"
        "3. Your answer in double parentheses at the end\n"
    ),
    "MMLUPro_Math_Valid": "Can you solve the following math question as accurately as possible?\n\n<question>\n{}\n</question>\n\nPresent your analysis concisely using only essential reasoning steps. Provide the final answer in double parentheses at the end of your response : ((answer))",
}

MOA_aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""