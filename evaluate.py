import json
from collections import defaultdict
from retrieve import retrieve

TOP_K = 3


def load_eval_set(path="eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(eval_set, top_k=TOP_K):
    results = []

    for q in eval_set:
        retrieved = retrieve(q["question"], top_k=top_k)
        retrieved_papers = [paper for (paper, idx, content, sim) in retrieved]
        expected = q["expected_papers"]

        if q["category"] == "out_of_scope":
            hit = None
            rr = None
        else:
            hit = any(p in retrieved_papers for p in expected)

            # Reciprocal rank: 1 / (rank of first correct paper), else 0
            rr = 0.0
            for rank, paper in enumerate(retrieved_papers, 1):
                if paper in expected:
                    rr = 1.0 / rank
                    break

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected": expected,
            "retrieved_papers": retrieved_papers,
            "hit": hit,
            "rr": rr,
        })

    return results


def report(results):
    scored = [r for r in results if r["hit"] is not None]
    overall_hits = sum(1 for r in scored if r["hit"])
    overall_mrr = sum(r["rr"] for r in scored) / len(scored)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL METRICS (top-{TOP_K})")
    print(f"{'='*60}")
    print(f"Hit Rate: {overall_hits}/{len(scored)} = {overall_hits/len(scored):.1%}")
    print(f"MRR:      {overall_mrr:.3f}\n")

    # Per-category: both metrics
    by_cat = defaultdict(lambda: {"hits": 0, "rr_sum": 0.0, "total": 0})
    for r in scored:
        c = by_cat[r["category"]]
        c["total"] += 1
        c["rr_sum"] += r["rr"]
        if r["hit"]:
            c["hits"] += 1

    print("By category:")
    print(f"  {'category':18s} {'hit rate':>10s} {'MRR':>8s}")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        hr = c["hits"] / c["total"]
        mrr = c["rr_sum"] / c["total"]
        print(f"  {cat:18s} {hr:>9.1%} {mrr:>8.3f}")

    # Show questions where the right paper wasn't at rank 1 (rr < 1.0)
    imperfect = [r for r in scored if r["rr"] < 1.0]
    if imperfect:
        print(f"\n{'='*60}")
        print(f"RANKED BELOW #1 ({len(imperfect)}) — where the correct paper wasn't first:")
        print(f"{'='*60}")
        for r in imperfect:
            rank_str = f"rank {int(1/r['rr'])}" if r["rr"] > 0 else "NOT in top-K"
            print(f"  [Q{r['id']}] rr={r['rr']:.3f} ({rank_str}) — {r['question']}")
            print(f"       expected {r['expected']}, got {r['retrieved_papers']}")


if __name__ == "__main__":
    eval_set = load_eval_set()
    print(f"Evaluating retrieval on {len(eval_set)} questions (top-{TOP_K})...")
    results = evaluate_retrieval(eval_set)
    report(results)