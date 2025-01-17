# Copyright 2022 The Kubeflow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Vertex AI Infra Validation Job Launcher Client module."""

import json
import os

from google_cloud_pipeline_components.container.v1.infra_validation_job import launcher
from google_cloud_pipeline_components.container.v1.infra_validation_job import remote_runner

import unittest
from unittest import mock


class LauncherInfraValidationJobUtilsTests(unittest.TestCase):

  def setUp(self):
    super(LauncherInfraValidationJobUtilsTests, self).setUp()
    self._project = 'test_project'
    self._location = 'test_region'
    self._gcp_resources = os.path.join(
        os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'test_file_path/test_file.txt'
    )

  @mock.patch.object(
      remote_runner, 'create_infra_validation_job', autospec=True
  )
  def test_launcher_on_infra_validation_job_type(
      self, mock_create_infra_validation_job
  ):
    job_type = 'InfraValidationJob'
    payload = {
        'infra_validation_example_path': 'test_infra_validator_path',
        'machine_type': 'n1-standard-8',
    }
    input_args = [
        '--type',
        job_type,
        '--project',
        self._project,
        '--location',
        self._location,
        '--gcp_resources',
        self._gcp_resources,
        '--executor_input',
        'executor_input',
        '--payload',
        json.dumps(payload),
    ]
    launcher.main(input_args)
    mock_create_infra_validation_job.assert_called_once_with(
        type=job_type,
        project=self._project,
        location=self._location,
        payload=json.dumps(payload),
        executor_input='executor_input',
        gcp_resources=self._gcp_resources,
    )
