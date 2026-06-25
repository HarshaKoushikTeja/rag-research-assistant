import json
import ollama
from generate import answer

JUDGE_MODEL = "gemma3:4b"
REFUSAL_MARKER = "don't have enough information"


def load_eval_set(path="eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_refusal(answer_text):
    """Detect whether the system declined to answer."""
    return REFUSAL_MARKER.lower() in answer_text.lower()


def build_context_string(chunks):
    blocks = []
    for i, (paper, idx, content, sim) in enumerate(chunks, 1):
        blocks.append(f"[Source {i}: {paper}, chunk {idx}]\n{content}")
    return "\n\n".join(blocks)


def judge_faithfulness(context, generated_answer):
    """Local LLM judge: is the answer grounded in the context? Returns (score, reason)."""
    judge_prompt = f"""You are evaluating whether an ANSWER is faithful to the provided CONTEXT.
A faithful answer makes only claims supported by the context. An unfaithful answer adds facts not present in the context.

Score faithfulness from 0.0 to 1.0:
- 1.0 = every claim is fully supported by the context
- 0.5 = partially supported; some claims not in context
- 0.0 = mostly unsupported or contradicts the context

CONTEXT:
{context}

ANSWER:
{generated_answer}

Respond with ONLY a JSON object, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}"""

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = response["message"]["content"].strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed.get("score"), parsed.get("reason", "")
    except (ValueError, json.JSONDecodeError):
        return None, f"parse_error: {raw[:80]}"


def evaluate(eval_set):
    rows = []
    for q in eval_set:
        answerable = len(q["expected_papers"]) > 0
        ans_text, chunks = answer(q["question"])
        refused = is_refusal(ans_text)

        # Classify into confusion-matrix cell
        if answerable and not refused:
            cell = "answered"          # true positive
        elif answerable and refused:
            cell = "false_refusal"     # false negative
        elif not answerable and refused:
            cell = "correct_refusal"   # true negative
        else:
            cell = "hallucination"     # false positive

        # Faithfulness only for true positives (answerable + answered)
        faith_score, faith_reason = (None, "")
        if cell == "answered":
            context = build_context_string(chunks)
            faith_score, faith_reason = judge_faithfulness(context, ans_text)

        rows.append({
            "id": q["id"], "question": q["question"], "category": q["category"],
            "answerable": answerable, "refused": refused, "cell": cell,
            "answer": ans_text, "faithfulness": faith_score, "faith_reason": faith_reason,
        })
        print(f"  [Q{q['id']:>2}] {cell:16s} faith={faith_score}")
    return rows


def report(rows):
    cells = {"answered": 0, "false_refusal": 0, "correct_refusal": 0, "hallucination": 0}
    for r in rows:
        cells[r["cell"]] += 1

    print(f"\n{'='*60}")
    print("GENERATION CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"{'':22s}{'answered':>12s}{'refused':>12s}")
    print(f"{'answerable':22s}{cells['answered']:>12d}{cells['false_refusal']:>12d}")
    print(f"{'out-of-scope':22s}{cells['hallucination']:>12d}{cells['correct_refusal']:>12d}")

    print(f"\n  True positives (answered correctly):  {cells['answered']}")
    print(f"  True negatives (correct refusals):    {cells['correct_refusal']}")
    print(f"  False negatives (false refusals):     {cells['false_refusal']}")
    print(f"  False positives (hallucinations):     {cells['hallucination']}")

    # Faithfulness over true positives that scored
    faith_scores = [r["faithfulness"] for r in rows
                    if r["cell"] == "answered" and r["faithfulness"] is not None]
    if faith_scores:
        avg = sum(faith_scores) / len(faith_scores)
        print(f"\n  Mean faithfulness (over {len(faith_scores)} answered): {avg:.3f}")

    # Actionable failures
    false_refusals = [r for r in rows if r["cell"] == "false_refusal"]
    hallucinations = [r for r in rows if r["cell"] == "hallucination"]
    low_faith = [r for r in rows if r["cell"] == "answered"
                 and r["faithfulness"] is not None and r["faithfulness"] < 1.0]

    if false_refusals:
        print(f"\n  FALSE REFUSALS (answer existed but system declined):")
        for r in false_refusals:
            print(f"    [Q{r['id']}] {r['question']}")
    if hallucinations:
        print(f"\n  HALLUCINATIONS (answered an out-of-scope question):")
        for r in hallucinations:
            print(f"    [Q{r['id']}] {r['question']}")
    if low_faith:
        print(f"\n  LOW-FAITHFULNESS ANSWERS (score < 1.0):")
        for r in low_faith:
            print(f"    [Q{r['id']}] faith={r['faithfulness']} - {r['faith_reason']}")


if __name__ == "__main__":
    eval_set = load_eval_set()
    print(f"Evaluating generation on {len(eval_set)} questions (2 LLM calls each, will take a few minutes)...\n")
    rows = evaluate(eval_set)
    report(rows)

    # Save full results for the record
    with open("generation_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nFull results saved to generation_eval_results.json")
