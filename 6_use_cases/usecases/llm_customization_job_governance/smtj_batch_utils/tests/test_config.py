import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

UTILS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTILS))


class OfflineConfigTests(unittest.TestCase):
    def test_import_and_load_without_creating_aws_clients(self):
        with patch("boto3.client", side_effect=AssertionError("AWS client created")):
            config = importlib.reload(importlib.import_module("config"))
            queues = config.load_config(str(UTILS / "config.yaml"))
        self.assertTrue(queues)
        self.assertTrue(
            all(queue.to_create_request()["jobQueueName"] for queue in queues)
        )
