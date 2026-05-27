import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parent


def _module_names(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


class NamingContractTest(unittest.TestCase):
    def test_textgrad_trader_method_names_match_reference_paper(self):
        expected = {
            "agent/router.py": {"create_bayesian_contextual_gating_router"},
            "agent/router1.py": {"create_bayesian_contextual_gating_router"},
            "agent/manager.py": {"create_adversarial_pareto_reasoning_manager"},
            "agent/Reflection.py": {"create_semantic_gradient_descent_reflection"},
            "agent/weekly_reflection.py": {"create_weekly_strategy_distillation_node"},
            "graph/graph_workflow.py": {"TextGradTraderWorkflow"},
            "graph/train.py": {"TextGradTraderRunner"},
        }

        for relative_path, expected_names in expected.items():
            self.assertLessEqual(expected_names, _module_names(relative_path))


if __name__ == "__main__":
    unittest.main()
