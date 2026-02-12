"""
SimpleQA Verified Benchmark Evaluation

SimpleQA Verified (https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified) is a novel
benchmark from Google DeepMind in collaboration with Google Research which evaluates Large
Language Model (LLM) short-form factuality.

This code is inspired from the SimpleQA benchmark evaluation methodology described in
OpenAI's SimpleEvals GitHub: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py

Research Paper: https://arxiv.org/abs/2509.07968
"""

# --- Setup ---

import argparse
import os
import sys
import pandas as pd
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import json
import string

import numpy as np
import kagglehub
from openai import OpenAI

# --- Configuration ---

JUDGE_MODEL = "gpt-4.1-2025-04-14"

# --- Clients (initialized in main) ---

judge_client: OpenAI = None
test_client: OpenAI = None
TEST_MODEL: str = None
test_extra_headers: dict = {}


def init_clients(
    judge_api_key: str,
    judge_base_url: str | None,
    test_api_key: str,
    test_base_url: str | None,
    test_model: str,
    extra_headers: dict | None = None,
):
    global judge_client, test_client, TEST_MODEL, test_extra_headers
    TEST_MODEL = test_model
    test_extra_headers = extra_headers or {}

    judge_client = OpenAI(
        api_key=judge_api_key,
        **({"base_url": judge_base_url} if judge_base_url else {}),
    )
    test_client = OpenAI(
        api_key=test_api_key,
        **({"base_url": test_base_url} if test_base_url else {}),
    )


def generate_test(prompt: str) -> str:
    completion = test_client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_headers=test_extra_headers if test_extra_headers else None,
    )
    return completion.choices[0].message.content


def generate_judge(prompt: str) -> str:
    completion = judge_client.chat.completions.create(
        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


# --- Grader Template ---

GRADER_TEMPLATE = """
  Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
  First, I will give examples of each grade, and then you will grade a new example.


  The following are examples of CORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia Obama and Sasha Obama
  Predicted answer 1: sasha and malia obama
  Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
  Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
  Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
  ```
  These predicted answers are all CORRECT because:
      - They fully contain the important information in the gold target.
      - They do not contain any information that contradicts the gold target.
      - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
      - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
      - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


  The following are examples of INCORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: Malia.
  Predicted answer 2: Malia, Sasha, and Susan.
  Predicted answer 3: Barack Obama does not have any children.
  Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
  Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
  Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
  ```
  These predicted answers are all INCORRECT because:
      - A factual statement in the answer contradicts the gold target.
      - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


  The following are examples of NOT_ATTEMPTED predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: I don't know.
  Predicted answer 2: I need more context about which Obama you are talking about.
  Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
  Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
  Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
  Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
  Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
  ```
  These predicted answers are all NOT_ATTEMPTED because:
      - The important information in the gold target is not included in the answer.
      - No statements in the answer contradict the gold target.
      - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


  Also note the following things:
  - For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k (acceptable range: anything between 118k and 122k)".
      - Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall within the range specified in the gold target.
      - Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the range specified in the gold target.
      - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
  - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
      - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
  - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
      - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
      - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
      - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer "1.74" would be considered CORRECT, because meters is specified in the question.
      - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
  - Do not punish for typos in people's name if it's clearly the same name.
      - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


  Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
  ```
  Question: {question}
  Gold target: {target}
  Predicted answer: {predicted_answer}
  ```

  Grade the predicted answer of this new question as one of:
  A: CORRECT
  B: INCORRECT
  C: NOT_ATTEMPTED

  Just return the letters "A", "B", or "C", with no text around it.
""".strip()

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"  # Corresponds to NOT_ATTEMPTED

# --- Helpers ---


def format_grader_prompt(question: str, target: str, predicted_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )


def grade_answer_with_llm(question: str, target: str, predicted_answer: str) -> str:
    grader_llm_prompt_content = format_grader_prompt(question, target, predicted_answer)
    grading_response_text = generate_judge(grader_llm_prompt_content)

    match = re.search(r"(A|B|C)", grading_response_text)
    if match:
        return match.group(0)
    else:
        if "CORRECT" in grading_response_text.upper():
            return "A"
        if "INCORRECT" in grading_response_text.upper():
            return "B"
        if "NOT_ATTEMPTED" in grading_response_text.upper():
            return "C"
        print(
            f"Could not parse grade from: '{grading_response_text}'. "
            f"Defaulting to {DEFAULT_GRADE_IF_UNPARSEABLE}."
        )
        return DEFAULT_GRADE_IF_UNPARSEABLE


# --- Load Examples ---

KAGGLEHUB_HANDLE = "deepmind/simpleqa-verified"
SIMPLEQA_CSV_FILENAME = "simpleqa_verified.csv"
_SIMPLEQA_DATASET_PATH = None


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _load_contaminated_questions(jsonl_path: str) -> set[str]:
    """Load questions from a JSONL training file and return normalized set."""
    contaminated = set()
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    contaminated.add(_normalize_text(msg["content"]))
    return contaminated


def load_from_kagglehub(
    num_examples: int = None,
    n_repeats: int = 1,
    decontaminate_path: str = None,
) -> pd.DataFrame:
    global _SIMPLEQA_DATASET_PATH

    if _SIMPLEQA_DATASET_PATH is None:
        dataset_dir_str = kagglehub.dataset_download(KAGGLEHUB_HANDLE)
        dataset_dir = Path(dataset_dir_str)
        _SIMPLEQA_DATASET_PATH = dataset_dir / SIMPLEQA_CSV_FILENAME
        print(f"Dataset downloaded to: {_SIMPLEQA_DATASET_PATH}")

    df = pd.read_csv(_SIMPLEQA_DATASET_PATH)

    # Filter out contaminated prompts (exact match)
    if decontaminate_path:
        contaminated_qs = _load_contaminated_questions(decontaminate_path)
        before = len(df)
        df = df[~df["problem"].apply(_normalize_text).isin(contaminated_qs)]
        after = len(df)
        print(
            f"Decontamination: removed {before - after} / {before} examples "
            f"matching {decontaminate_path} ({after} remaining)"
        )

    examples = [row.to_dict() for _, row in df.iterrows()]

    if num_examples is not None and num_examples > 0:
        rng = random.Random(0)
        k_sample = min(num_examples, len(examples))
        if k_sample > 0:
            sampled_examples = [
                examples[i]
                for i in sorted(rng.sample(range(len(examples)), k=k_sample))
            ]
        else:
            sampled_examples = []
    else:
        sampled_examples = examples

    final_examples = sampled_examples * n_repeats
    return pd.DataFrame(final_examples)


# --- Response Generation ---


def simpleqa_run_task(data_row: dict) -> dict:
    question = data_row.get("problem", "")
    gold_target = data_row.get("answer", "")

    predicted_answer_text = generate_test(prompt=question)

    grade_letter = grade_answer_with_llm(question, gold_target, predicted_answer_text)

    is_correct_val = grade_letter == "A"
    is_incorrect_val = grade_letter == "B"
    is_not_attempted_val = grade_letter == "C"

    return {
        "question": question,
        "gold_target": gold_target,
        "predicted_answer": predicted_answer_text,
        "grade_letter": grade_letter,
        "grade_str": CHOICE_LETTER_TO_STRING.get(grade_letter, "UNKNOWN"),
        "is_correct": is_correct_val,
        "is_incorrect": is_incorrect_val,
        "is_not_attempted": is_not_attempted_val,
    }


# --- Response Evaluation ---


def get_accuracy_given_attempted(df: pd.DataFrame) -> float:
    attempted_count = df["is_correct"].sum() + df["is_incorrect"].sum()
    if attempted_count == 0:
        return 0.0
    return df["is_correct"].sum() / attempted_count


def get_f1_score(df: pd.DataFrame) -> float:
    if df.empty or not (
        "is_correct" in df.columns and "is_incorrect" in df.columns
    ):
        return 0.0

    num_total_samples = len(df)
    if num_total_samples == 0:
        return 0.0

    mean_correct = df["is_correct"].sum() / num_total_samples

    accuracy_given_attempted_val = get_accuracy_given_attempted(df)

    numerator = 2 * accuracy_given_attempted_val * mean_correct
    denominator = accuracy_given_attempted_val + mean_correct
    if denominator == 0:
        return 0.0
    return numerator / denominator


# --- Main ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleQA Verified Benchmark Evaluation"
    )
    parser.add_argument(
        "--test-model",
        required=True,
        help="Model name for the test model (e.g. 'meta-llama/Llama-3.1-8B-Instruct')",
    )
    parser.add_argument(
        "--test-base-url",
        default=None,
        help="Base URL for the test model's OpenAI-compatible API (e.g. 'http://localhost:8000/v1')",
    )
    parser.add_argument(
        "--test-api-key",
        default=None,
        help="API key for the test model (defaults to env var TEST_API_KEY or 'no-key')",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="OpenAI API key for the judge model (defaults to env var OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for the judge model (defaults to OpenAI's API)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--extra-header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra HTTP headers for the test model API (e.g. 'HTTP-Referer=https://example.com'). Can be repeated.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--decontaminate",
        default=None,
        metavar="JSONL_PATH",
        help="Path to a training JSONL file. Benchmark prompts that exact-match any user message in this file will be excluded.",
    )
    return parser.parse_args()


def _run_task_with_index(args_tuple):
    """Wrapper for parallel execution: (index, total, data_row) -> (index, result)"""
    idx, total, data_row = args_tuple
    try:
        result = simpleqa_run_task(data_row=data_row)
        return idx, result, None
    except Exception as e:
        return idx, None, e


if __name__ == "__main__":
    args = parse_args()

    judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "")
    test_api_key = args.test_api_key or os.environ.get("TEST_API_KEY", "no-key")

    extra_headers = {}
    for h in args.extra_header:
        key, _, value = h.partition("=")
        extra_headers[key] = value

    init_clients(
        judge_api_key=judge_api_key,
        judge_base_url=args.judge_base_url,
        test_api_key=test_api_key,
        test_base_url=args.test_base_url,
        test_model=args.test_model,
        extra_headers=extra_headers,
    )

    # Smoke test
    print(f"Test model:  {TEST_MODEL} @ {args.test_base_url or 'OpenAI default'}")
    print(f"Judge model: {JUDGE_MODEL} @ {args.judge_base_url or 'OpenAI default'}")
    print(f"Parallel:    {args.parallel} workers")
    print()

    for label, fn in [("Test", generate_test), ("Judge", generate_judge)]:
        resp = fn("Count to 3.")
        print(f"[Smoke test] {label}: {resp.strip()[:80]}")
    print()

    data_df = load_from_kagglehub(
        num_examples=args.num_examples,
        decontaminate_path=args.decontaminate,
    )
    total = len(data_df)
    print(f"Evaluating {total} examples...\n")

    # Prepare task args
    task_args = [
        (i, total, row.to_dict()) for i, (_, row) in enumerate(data_df.iterrows())
    ]

    results_by_index = {}
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(_run_task_with_index, ta): ta[0] for ta in task_args
        }
        for future in as_completed(futures):
            idx, result, err = future.result()
            completed += 1
            if err is not None:
                errors += 1
                print(
                    f"[{completed}/{total}] ERROR (idx={idx}): {err}",
                    flush=True,
                )
            else:
                results_by_index[idx] = result
                print(
                    f"[{completed}/{total}] {result['grade_str']:15s} | "
                    f"{result['question'][:60]}...",
                    flush=True,
                )

    # Reassemble in original order
    results_list = [results_by_index[i] for i in sorted(results_by_index)]
    results_df = pd.DataFrame(results_list)

    print("\n--- Results ---")
    print(f"Total:         {len(results_df)}")
    print(f"Correct:       {results_df['is_correct'].sum()}")
    print(f"Incorrect:     {results_df['is_incorrect'].sum()}")
    print(f"Not attempted: {results_df['is_not_attempted'].sum()}")
    if errors:
        print(f"Errors:        {errors}")
    print(f"Accuracy (attempted): {get_accuracy_given_attempted(results_df):.4f}")
    print(f"F1 Score:             {get_f1_score(results_df):.4f}")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
