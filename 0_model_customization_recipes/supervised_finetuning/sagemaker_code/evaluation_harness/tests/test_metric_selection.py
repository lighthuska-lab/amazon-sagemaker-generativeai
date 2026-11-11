import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

EVALUATE = Path(__file__).resolve().parents[1] / "src/evalharness/evaluate.py"
BASE_METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu", "toxicity"]
JUDGE_METRICS = [
    "answer_similarity",
    "answer_correctness",
    "answer_relevance",
    "relevance",
    "faithfulness",
]


def load_evaluator():
    modules = {}
    for name in [
        "mlflow",
        "mlflow.metrics",
        "mlflow.metrics.genai",
        "pandas",
        "bert_score",
    ]:
        modules[name] = ModuleType(name)
    mlflow = modules["mlflow"]
    for name in [
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "evaluate",
        "log_metrics",
    ]:
        setattr(mlflow, name, Mock())
    mlflow.start_run.return_value.__enter__ = Mock()
    mlflow.start_run.return_value.__exit__ = Mock(return_value=False)
    for name in BASE_METRICS:
        setattr(modules["mlflow.metrics"], name, Mock(return_value=object()))
    for name in JUDGE_METRICS:
        setattr(modules["mlflow.metrics.genai"], name, Mock(return_value=object()))
    modules["bert_score"].score = Mock()
    modules["pandas"].DataFrame = Mock()
    spec = importlib.util.spec_from_file_location("evaluation_under_test", EVALUATE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def config(metrics):
    return SimpleNamespace(
        metrics=metrics,
        model_judge=None,
        model_judge_parameters=None,
        mlflow_tracking_server_arn="offline",
        mlflow_experiment_name="experiment",
        mlflow_run_name="parent",
        source_model_predictions_path="source.jsonl",
    )


class MetricSelectionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_evaluator()

    def run_evaluation(self, configuration):
        evaluator = self.module.Evaluate(configuration, {"source_model_test_data": []})
        with patch.object(
            self.module, "get_run_id_from_name", return_value="parent-id"
        ):
            evaluator.run()
        return evaluator

    def test_adjacent_unsupported_metrics_are_skipped(self):
        requested = ["unknown-one", "unknown-two", "rouge1"]
        with self.assertLogs(level="WARNING") as logs:
            self.run_evaluation(config(requested))
        self.assertTrue(any("unknown-one" in line for line in logs.output))
        self.assertTrue(any("unknown-two" in line for line in logs.output))
        self.assertEqual(
            len(self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"]), 1
        )
        self.assertEqual(requested, ["unknown-one", "unknown-two", "rouge1"])

    def test_bert_does_not_skip_the_next_metric(self):
        score = Mock()
        score.mean.return_value = 0.5
        self.module.bert_score.return_value = (score, score, score)
        requested = ["bert", "unknown", "rouge2"]
        evaluator = self.run_evaluation(config(requested))
        self.assertTrue(evaluator.bert)
        self.module.bert_score.assert_called_once()
        self.assertEqual(
            self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"],
            [evaluator.available_metrics["rouge2"]],
        )
        self.assertEqual(requested, ["bert", "unknown", "rouge2"])

    def test_repeat_initialization_preserves_bert(self):
        configuration = config(["bert", "rouge1"])
        first = self.module.Evaluate(configuration, {})
        second = self.module.Evaluate(configuration, {})
        self.assertTrue(first.bert)
        self.assertTrue(second.bert)
        self.assertEqual(configuration.metrics, ["bert", "rouge1"])

    def test_empty_or_optional_metrics(self):
        for requested in [[], None]:
            with self.subTest(metrics=requested):
                self.module.mlflow.evaluate.reset_mock()
                evaluator = self.run_evaluation(config(requested))
                self.assertFalse(evaluator.bert)
                self.assertEqual(
                    self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"], []
                )

    def test_supported_order_preserved(self):
        evaluator = self.run_evaluation(config(["rouge2", "bleu", "rouge1"]))
        self.assertEqual(
            self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"],
            [evaluator.available_metrics[x] for x in ["rouge2", "bleu", "rouge1"]],
        )
        self.module.bert_score.assert_not_called()

    def test_judge_metric_only_selected_with_judge(self):
        configuration = config(["answer_similarity", "rouge1"])
        self.run_evaluation(configuration)
        self.assertEqual(
            len(self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"]), 1
        )
        configuration.model_judge = "offline-judge"
        self.run_evaluation(configuration)
        self.assertEqual(
            len(self.module.mlflow.evaluate.call_args.kwargs["extra_metrics"]), 2
        )


if __name__ == "__main__":
    unittest.main()
