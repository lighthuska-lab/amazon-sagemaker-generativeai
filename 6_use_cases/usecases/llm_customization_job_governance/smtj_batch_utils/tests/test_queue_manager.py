import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

UTILS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTILS))
manager_module = importlib.import_module("queue_manager")


def api_error(code="ClientException", message="Object already exists"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "TestOperation")


class CreationTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.manager = manager_module.TrainingQueueManager(self.client, verbose=False)

    def test_duplicate_service_environment(self):
        self.client.create_service_environment.side_effect = api_error()
        self.client.describe_service_environments.return_value = {
            "serviceEnvironments": [
                {"serviceEnvironmentName": "env", "serviceEnvironmentArn": "env-arn"}
            ]
        }
        self.assertEqual(
            self.manager.create_service_env({"serviceEnvironmentName": "env"})[
                "serviceEnvironmentArn"
            ],
            "env-arn",
        )

    def test_duplicate_queue(self):
        self.client.create_job_queue.side_effect = api_error()
        self.client.describe_job_queues.return_value = {
            "jobQueues": [{"jobQueueName": "queue", "jobQueueArn": "queue-arn"}]
        }
        self.assertEqual(
            self.manager.create_job_queue({"jobQueueName": "queue"})["jobQueueArn"],
            "queue-arn",
        )

    def test_duplicate_policy(self):
        self.client.create_scheduling_policy.side_effect = api_error()
        self.client.list_scheduling_policies.return_value = {
            "schedulingPolicies": [
                {"arn": "arn:aws:batch:us-east-1:123456789012:scheduling-policy/policy"}
            ]
        }
        self.assertEqual(
            self.manager.create_scheduling_policy({"name": "policy"})["name"], "policy"
        )

    def test_duplicate_quota_share(self):
        self.client.create_quota_share.side_effect = api_error(
            message="Quota share already exists"
        )
        self.client.describe_job_queues.return_value = {
            "jobQueues": [{"jobQueueArn": "queue-arn"}]
        }
        self.assertEqual(
            self.manager.create_quota_share(
                {"quotaShareName": "share", "jobQueue": "queue"}
            )["quotaShareArn"],
            "queue-arn/quota-share/share",
        )

    def test_unexpected_creation_errors_preserved(self):
        for method, client_method in [
            ("create_service_env", "create_service_environment"),
            ("create_job_queue", "create_job_queue"),
            ("create_scheduling_policy", "create_scheduling_policy"),
            ("create_quota_share", "create_quota_share"),
        ]:
            for code in ["AccessDeniedException", "ServerException", "ClientException"]:
                with self.subTest(method=method, code=code):
                    error = api_error(code, "Permission denied or invalid request")
                    getattr(self.client, client_method).side_effect = error
                    with self.assertRaises(ClientError) as raised:
                        getattr(self.manager, method)({})
                    self.assertIs(raised.exception, error)


class QuotaDeletionTests(unittest.TestCase):
    arn = "arn:aws:batch:us-east-1:123456789012:job-queue/queue/quota-share/share"

    def setUp(self):
        self.client = Mock()
        self.manager = manager_module.TrainingQueueManager(self.client, verbose=False)
        self.paginator = self.client.get_paginator.return_value

    def test_deleted_only_after_all_pages_are_empty(self):
        self.paginator.paginate.return_value = [
            {"quotaShares": []},
            {"quotaShares": []},
        ]
        self.manager.wait_for_qs_update(self.arn, "DELETED", "DISABLED")
        self.client.get_paginator.assert_called_once_with("list_quota_shares")
        self.paginator.paginate.assert_called_once_with(
            jobQueue=self.arn.rsplit("/quota-share/", 1)[0]
        )
        self.client.describe_quota_share.assert_not_called()

    @patch.object(manager_module.time, "sleep")
    def test_share_on_later_page_keeps_waiting(self, sleep):
        self.paginator.paginate.side_effect = [
            [{"quotaShares": []}, {"quotaShares": [{"quotaShareArn": self.arn}]}],
            [{"quotaShares": []}],
        ]
        self.manager.wait_for_qs_update(self.arn, "DELETED", "DISABLED")
        sleep.assert_called_once_with(5)

    def test_list_errors_are_not_deletion_success(self):
        for code in ["ClientException", "AccessDeniedException", "ServerException"]:
            with self.subTest(code=code):
                error = api_error(code, "Cannot inspect quota shares")
                self.paginator.paginate.side_effect = error
                self.client.describe_quota_share.side_effect = error
                with self.assertRaises(ClientError) as raised:
                    self.manager.wait_for_qs_update(self.arn, "DELETED", "DISABLED")
                self.assertIs(raised.exception, error)

    def test_creation_state_still_uses_describe(self):
        self.client.describe_quota_share.return_value = {
            "status": "VALID",
            "state": "ENABLED",
        }
        self.manager.wait_for_qs_update(self.arn, "VALID")
        self.client.get_paginator.assert_not_called()

    def test_invalid_state_is_reported(self):
        self.client.describe_quota_share.return_value = {
            "status": "INVALID",
            "state": "ENABLED",
        }
        with self.assertRaises(ValueError):
            self.manager.wait_for_qs_update(self.arn, "VALID")


if __name__ == "__main__":
    unittest.main()
