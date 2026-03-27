from grader import evaluate
from baseline_agent import baseline_policy


result = evaluate(baseline_policy)

print("\nEvaluation Result")
print(result)