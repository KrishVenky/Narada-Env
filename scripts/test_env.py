"""Quick smoke test for the environment."""
import sys
sys.path.insert(0, "src/envs")
from clindetect.server.environment import ClinDetectEnvironment
from clindetect.models import ClinDetectAction

env = ClinDetectEnvironment()
result = env.reset(task_type="monogenic", seed=1)
obs = result.observation
print(f"Reset: task={obs.task_type}")
print(f"  Node: {obs.current_node.name} ({obs.current_node.id})")
print(f"  Phenotypes: {obs.phenotype_names}")
print(f"  Candidates: {[v.gene for v in obs.candidate_variants]}")

# Hop to first neighbor
if obs.current_node.connected_node_ids:
    hop_target = obs.current_node.connected_node_ids[0]
    action = ClinDetectAction(action_type="hop", node_id=hop_target, reasoning="Exploring")
    result = env.step(action)
    obs = result.observation
    print(f"After hop to {hop_target}: reward={result.reward:.4f}, node={obs.current_node.name}")

# Backtrack
action = ClinDetectAction(action_type="backtrack", reasoning="Testing backtrack")
result = env.step(action)
print(f"After backtrack: reward={result.reward:.4f}")

# Flag a candidate (causal)
v_id = obs.candidate_variants[0].id
action = ClinDetectAction(action_type="flag_causal", variant_id=v_id, reasoning="Test flag")
result = env.step(action)
print(f"After flag ({v_id}): done={result.done} reward={result.reward:.4f}")
assert result.done, "Episode should be done after flag_causal"
assert 0.01 <= result.reward <= 0.99, f"Reward {result.reward} out of (0.01, 0.99)"

# State check
state = env.state()
print(f"State: flagged={state.flagged_variants} ground_truth={state.ground_truth_variants[:2]}")

print("ENV SMOKE TEST PASSED")
