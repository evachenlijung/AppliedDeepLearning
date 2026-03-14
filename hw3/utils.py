from typing import List
import re

# v1
def get_inference_system_prompt() -> str:
    prompt = (
        f"You are a meticulous test-taker; you excel at question answering. "
        f"You are good at finding the right answer among many passages. "
        f"You are also good at identifying passages that may not contain the answer you want, even if they seem to be related to the question. "
    )
    return prompt
def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    passages = "\n".join(
        [f"Passage {i}: {p.strip()}" for i, p in enumerate(context_list)]
    )
    prompt = (
        f"Find the answer for the query based on the given passages. "
        f"If there's no answer in the given passages, just answer 'CANNOTANSWER'\n"
        f"query: {query}\n"
        f"{passages}"
    )
    return prompt
def parse_generated_answer(pred_ans: str) -> str:
    parsed_ans = pred_ans
    return parsed_ans



# # base    
# def get_inference_system_prompt() -> str:
#     prompt = ""
#     return prompt

# def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
#     passages = "\n".join(
#         [f"Passage {i}: {p.strip()}" for i, p in enumerate(context_list)]
#     )
#     prompt = (
#         f"query: {query}"
#         f"{passages}"
#     )
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     parsed_ans = pred_ans
#     return parsed_ans



# # v2
# def get_inference_system_prompt() -> str:
#     prompt = (
#         "You are an intelligent and precise question-answering assistant. "
#         "You must answer questions based ONLY on the given passages. "
#         "If the passages do not contain the answer, reply exactly with: CANNOTANSWER. "
#         "Do not hallucinate or guess any information. "
#         "Be concise, factual, and use formal English in your responses."
#     ) 
#     return prompt

# def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
#     joined_context = "\n".join(
#         [f"[Passage {i+1}]\n{ctx.strip()}" for i, ctx in enumerate(context_list)]
#     )
#     prompt = (
#         f"Answer in complete sentences and rely strictly on the information from the given passages ONLY. "
#         f"If the answer cannot be found in the passages, reply exactly with: CANNOTANSWER. "
#         f"For example:\n"
#         f"Question: Where is the capital of Taiwan?\n"
#         f"[Passage 1]\nTaiwan is famous for its delicious food.\n"
#         f"[Passage 2]\nThe capital of France is Paris.\n\n"
#         f"[Passage 3]\nTaiwan's cities are busy.\n"
#         f"[Passage 4]\nTaipei is in the northern Taiwan.\n"
#         f"Answer: CANNOTANSWER\n\n"
#         f"The following are passages relevant to a question. "
#         f"Use ONLY these passages to answer.\n\n"       
#         f"Question: {query}\n"
#         f"{joined_context}\n" 
#     )
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     ans = pred_ans.strip()
#     if re.search(r"(CANNOTANSWER|cannot\s*answer|no\s*answer|unknown|not\s*sure|cannot\s*be\s*determined)", ans, re.IGNORECASE):
#         return "CANNOTANSWER"
#     match = re.search(r"(?i)answer[:：]\s*(.*)", ans)
#     if match:
#         return match.group(1).strip()
#     lines = [line.strip() for line in ans.split("\n") if line.strip()]
#     if len(lines) > 0:
#         return lines[-1]
#     return ans