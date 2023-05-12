# Copyright 2021 The Kubeflow Authors. All Rights Reserved.
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
"""Test Vertex AI Custom Job Client module."""

from google_cloud_pipeline_components.v1.custom_job import utils
from kfp import components
import copy
import os
import unittest
from google.protobuf import json_format
from google_cloud_pipeline_components.v1.custom_job import component
from google_cloud_pipeline_components.tests.v1 import test_utils


class VertexAICustomJobUtilsTests1(unittest.TestCase):

  def setUp(self):
    super(VertexAICustomJobUtilsTests1, self).setUp()
    utils._DEFAULT_CUSTOM_JOB_CONTAINER_IMAGE = 'test_launcher_image'

  def _create_a_container_based_component(self) -> callable:
    """Creates a test container based component factory."""

    return components.load_component_from_text("""
name: ContainerComponent
inputs:
- {name: input_text, type: String, description: "Represents an input parameter."}
outputs:
- {name: output_value, type: String, description: "Represents an output paramter."}
implementation:
  container:
    image: google/cloud-sdk:latest
    command:
    - sh
    - -c
    - |
      set -e -x
      echo "$0, this is an output parameter"
    - {inputValue: input_text}
    - {outputPath: output_value}
""")

  def test_run_as_vertex_ai_custom_job_on_container_spec_with_defualts_values_converts_correctly(
      self,
  ):
    expected_results = {
        'name': 'ContainerComponent',
        'inputs': [
            {
                'name': 'input_text',
                'type': 'String',
                'description': 'Represents an input parameter.',
            },
            {
                'name': 'base_output_directory',
                'type': 'String',
                'default': '',
                'optional': True,
            },
            {
                'name': 'tensorboard',
                'type': 'String',
                'default': '',
                'optional': True,
            },
            {
                'name': 'network',
                'type': 'String',
                'default': '',
                'optional': True,
            },
            {
                'name': 'service_account',
                'type': 'String',
                'default': '',
                'optional': True,
            },
            {'name': 'project', 'type': 'String'},
            {'name': 'location', 'type': 'String'},
        ],
        'outputs': [
            {
                'name': 'output_value',
                'type': 'String',
                'description': 'Represents an output paramter.',
            },
            {'name': 'gcp_resources', 'type': 'String'},
        ],
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        },
    }
    component_factory_function = self._create_a_container_based_component()
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.to_dict(), expected_results
    )

  def test_run_as_vertex_ai_custom_with_accelerator_type_and_count_converts_correctly(
      self,
  ):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4", "accelerator_type": '
                        '"test_accelerator_type", "accelerator_count": 2}, '
                        '"replica_count": 1, "container_spec": {"image_uri": '
                        '"google/cloud-sdk:latest", "command": ["sh", "-c",'
                        ' "set '
                        '-e -x\\necho \\"$0, this is an output'
                        ' parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function,
        accelerator_type='test_accelerator_type',
        accelerator_count=2,
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_boot_disk_type_and_size_converts_correctly(
      self,
  ):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}, {"machine_spec": '
                        '{"machine_type": "n1-standard-4"}, "replica_count":'
                        ' "1", '
                        '"container_spec": {"image_uri": '
                        '"google/cloud-sdk:latest", "command": ["sh", "-c",'
                        ' "set '
                        '-e -x\\necho \\"$0, this is an output'
                        ' parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, replica_count=2
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_replica_count_greater_than_1_converts_correctly(
      self,
  ):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}, {"machine_spec": '
                        '{"machine_type": "n1-standard-4"}, "replica_count":'
                        ' "1", '
                        '"container_spec": {"image_uri": '
                        '"google/cloud-sdk:latest", "command": ["sh", "-c",'
                        ' "set '
                        '-e -x\\necho \\"$0, this is an output'
                        ' parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, replica_count=2
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_time_out_converts_correctly(self):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "scheduling":'
                        ' {"timeout": '
                        '2}, "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, timeout=2
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_restart_job_on_worker_restart_converts_correctly(
      self,
  ):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "scheduling": '
                        '{"restart_job_on_worker_restart": true}, '
                        '"service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, restart_job_on_worker_restart=True
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_custom_service_account_converts_correctly(
      self,
  ):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, service_account='test_service_account'
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_display_name_converts_correctly(self):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "test_display_name", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, display_name='test_display_name'
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_network_converts_correctly(self):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, network='test_network'
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_labels_converts_correctly(self):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "labels": {"test_key": '
                        '"test_value"}, "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, labels={'test_key': 'test_value'}
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )

  def test_run_as_vertex_ai_custom_with_reserved_ip_ranges(self):
    component_factory_function = self._create_a_container_based_component()

    expected_sub_results = {
        'implementation': {
            'container': {
                'image': 'test_launcher_image',
                'command': [
                    'python3',
                    '-u',
                    '-m',
                    'google_cloud_pipeline_components.container.v1.custom_job.launcher',
                ],
                'args': [
                    '--type',
                    'CustomJob',
                    '--payload',
                    (
                        '{"display_name": "ContainerComponent", "job_spec": '
                        '{"worker_pool_specs": [{"machine_spec":'
                        ' {"machine_type": '
                        '"n1-standard-4"}, "replica_count": 1,'
                        ' "container_spec": '
                        '{"image_uri": "google/cloud-sdk:latest", "command": '
                        '["sh", "-c", "set -e -x\\necho \\"$0, this is an'
                        ' output '
                        'parameter\\"\\n", '
                        '"{{$.inputs.parameters[\'input_text\']}}", '
                        '"{{$.outputs.parameters[\'output_value\'].output_file}}"]},'
                        ' "disk_spec": {"boot_disk_type": "pd-ssd", '
                        '"boot_disk_size_gb": 100}}], "reserved_ip_ranges": '
                        '["1.0.0.0", "2.0.0.0"], "service_account": '
                        '"{{$.inputs.parameters[\'service_account\']}}", '
                        '"network": "{{$.inputs.parameters[\'network\']}}", '
                        '"tensorboard": '
                        '"{{$.inputs.parameters[\'tensorboard\']}}", '
                        '"base_output_directory": {"output_uri_prefix": '
                        '"{{$.inputs.parameters[\'base_output_directory\']}}"}}}'
                    ),
                    '--project',
                    {'inputValue': 'project'},
                    '--location',
                    {'inputValue': 'location'},
                    '--gcp_resources',
                    {'outputPath': 'gcp_resources'},
                ],
            }
        }
    }
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_factory_function, reserved_ip_ranges=['1.0.0.0', '2.0.0.0']
    )

    self.assertDictContainsSubset(
        subset=expected_sub_results,
        dictionary=custom_job_spec.component_spec.to_dict(),
    )


class VertexAICustomJobUtilsTests2(unittest.TestCase):

  def setUp(self):
    super(VertexAICustomJobUtilsTests2, self).setUp()

    self._default_worker_pool_spec = {
        'machine_spec': {
            'machine_type': 'n1-standard-4',
        },
        'replica_count': 1,
        'container_spec': {
            'args': [],
            'command': [
                'sh',
                '-c',
                'set -e -x\necho "$0, this is an output parameter"\n',
                "{{$.inputs.parameters['input_text']}}",
                "{{$.outputs.parameters['output_value'].output_file}}",
            ],
            'image_uri': 'google/cloud-sdk:latest',
        },
        'disk_spec': {'boot_disk_type': 'pd-ssd', 'boot_disk_size_gb': 100},
    }

    self._default_component = components.load_component_from_text("""
name: ContainerComponent
inputs:
- {name: input_text, type: String, description: "Represents an input parameter."}
outputs:
- {name: output_value, type: String, description: "Represents an output parameter."}
implementation:
  container:
    image: google/cloud-sdk:latest
    command:
    - sh
    - -c
    - |
      set -e -x
      echo "$0, this is an output parameter"
    - {inputValue: input_text}
    - {outputPath: output_value}
""")

  def test_run_as_vertex_ai_custom_job_on_container_spec_with_default_values_converts_correctly(
      self,
  ):
    custom_job_component = utils.create_custom_training_job_from_component(
        self._default_component
    )
    test_utils.assert_pipeline_equals_golden(
        self,
        custom_job_component,
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'testdata',
            'custom_training_job_spec.json',
        ),
    )

  def test_run_as_vertex_ai_custom_with_accelerator_type_and_count_converts_correctly(
      self,
  ):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component,
        accelerator_type='test_accelerator_type',
        accelerator_count=2,
    )
    expected_worker_pool_spec = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec['machine_spec'][
        'accelerator_type'
    ] = 'test_accelerator_type'
    expected_worker_pool_spec['machine_spec']['accelerator_count'] = 2

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 1
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_run_as_vertex_ai_custom_with_boot_disk_type_and_size_converts_correctly(
      self,
  ):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component,
        boot_disk_type='test_type',
        boot_disk_size_gb=200,
    )
    expected_worker_pool_spec = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec['disk_spec']['boot_disk_type'] = 'test_type'
    expected_worker_pool_spec['disk_spec']['boot_disk_size_gb'] = 200

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 1
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_run_as_vertex_ai_custom_with_replica_count_greater_than_1_converts_correctly(
      self,
  ):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, replica_count=5
    )
    expected_worker_pool_spec0 = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec1 = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec1['replica_count'] = 4

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 2
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec0,
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[1],
        expected_worker_pool_spec1,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_run_as_vertex_ai_custom_with_time_out_converts_correctly(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, timeout='2s'
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['timeout'],
        components.structures.InputSpec(
            type='String', default='2s', optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_restart_job_on_worker_restart_converts_correctly(
      self,
  ):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, restart_job_on_worker_restart=True
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['restart_job_on_worker_restart'],
        components.structures.InputSpec(
            type='Boolean', default=True, optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_custom_service_account_converts_correctly(
      self,
  ):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, service_account='test_service_account'
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['service_account'],
        components.structures.InputSpec(
            type='String', default='test_service_account', optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_display_name_converts_correctly(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, display_name='test_display_name'
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['display_name'],
        components.structures.InputSpec(
            type='String', default='test_display_name', optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_network_converts_correctly(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, network='test_network'
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['network'],
        components.structures.InputSpec(
            type='String', default='test_network', optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_labels_converts_correctly(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, labels={'test_key': 'test_value'}
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['labels'],
        components.structures.InputSpec(
            type='Dict', default={'test_key': 'test_value'}, optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_reserved_ip_ranges(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, reserved_ip_ranges=['test_ip_range_network']
    )
    self.assertEqual(
        custom_job_spec.component_spec.inputs['reserved_ip_ranges'],
        components.structures.InputSpec(
            type='List', default=['test_ip_range_network'], optional=True
        ),
    )

  def test_run_as_vertex_ai_custom_with_nfs_mount(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component, nfs_mounts=[{'server': 's1', 'path': 'p1'}]
    )
    expected_worker_pool_spec = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec['nfs_mounts'] = [{'server': 's1', 'path': 'p1'}]

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 1
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_run_as_vertex_ai_custom_with_executor_input_in_command(self):
    component_function = components.load_component_from_text("""
name: ContainerComponent
outputs:
- {name: output_artifact, type: system.Artifact, description: "Represents an output artifact."}
implementation:
  container:
    image: gcr.io/repo/image:latest
    command: [python3, -u, -m, launcher, --executor_input, "{{$}}"]
""")
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_function
    )
    expected_worker_pool_spec = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec['container_spec'][
        'image_uri'
    ] = 'gcr.io/repo/image:latest'
    expected_worker_pool_spec['container_spec']['command'] = [
        'python3',
        '-u',
        '-m',
        'launcher',
        '--executor_input',
        '{{$.json_escape[1]}}',
    ]

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 1
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_run_as_vertex_ai_custom_name(self):
    custom_job_spec = utils.create_custom_training_job_from_component(
        self._default_component
    )
    self.assertEqual(custom_job_spec.component_spec.name, 'containercomponent')

  def test_run_as_vertex_ai_custom_description(self):
    custom_component = components.load_component_from_text("""
name: ContainerComponent
description: |
    Custom description for Container Component
inputs:
- {name: input_text, type: String, description: "Represents an input parameter."}
outputs:
- {name: output_value, type: String, description: "Represents an output parameter."}
implementation:
  container:
    image: google/cloud-sdk:latest
    command:
    - sh
    - -c
    - |
      set -e -x
      echo "$0, this is an output parameter"
    - {inputValue: input_text}
    - {outputPath: output_value}
""")
    custom_job_spec = utils.create_custom_training_job_from_component(
        custom_component
    )
    self.assertEqual(
        custom_job_spec.description,
        """A custom job that wraps containercomponent.

Original component description:
Custom description for Container Component


Custom Job wrapper description:
Launch a Custom training job using Vertex CustomJob API.""",
    )

  def test_run_as_vertex_ai_custom_with_executor_input_in_args(self):
    component_function = components.load_component_from_text("""
name: ContainerComponent
inputs:
- {name: input_text, type: String, description: "Represents an input parameter."}
outputs:
- {name: output_artifact, type: system.Artifact, description: "Represents an output artifact."}
implementation:
  container:
    image: gcr.io/repo/image:latest
    command: [python3, -u, -m, launcher]
    args: [
        --input_text, {inputValue: input_text},
        --executor_input, "{{$}}"
    ]
""")
    custom_job_spec = utils.create_custom_training_job_op_from_component(
        component_function
    )
    expected_worker_pool_spec = copy.deepcopy(self._default_worker_pool_spec)
    expected_worker_pool_spec['container_spec'][
        'image_uri'
    ] = 'gcr.io/repo/image:latest'
    expected_worker_pool_spec['container_spec']['command'] = [
        'python3',
        '-u',
        '-m',
        'launcher',
    ]
    expected_worker_pool_spec['container_spec']['args'] = [
        '--input_text',
        "{{$.inputs.parameters['input_text']}}",
        '--executor_input',
        '{{$.json_escape[1]}}',
    ]

    self.assertEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].type, 'List'
    )
    self.assertLen(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default, 1
    )
    self.assertDictEqual(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].default[0],
        expected_worker_pool_spec,
    )
    self.assertTrue(
        custom_job_spec.component_spec.inputs['worker_pool_specs'].optional
    )

  def test_custom_job_has_no_artifacts(self):
    # Note: This test is to assert that CustomTrainingJobOp has no input/output
    # artifacts. If this ever changes, then artifacts from the input component
    # would need to be merged with those from CustomTrainingJobOp.
    custom_training_job_dict = json_format.MessageToDict(
        component.custom_training_job.pipeline_spec
    )
    custom_training_job_dict_components = custom_training_job_dict['components']
    custom_training_job_comp_key = list(
        custom_training_job_dict_components.keys()
    )[0]
    custom_job_input_artifacts = custom_training_job_dict_components[
        custom_training_job_comp_key
    ]['inputDefinitions'].get('artifacts', {})
    custom_job_output_artifacts = custom_training_job_dict_components[
        custom_training_job_comp_key
    ]['outputDefinitions'].get('artifacts', {})

    self.assertEmpty(custom_job_input_artifacts)
    self.assertEmpty(custom_job_output_artifacts)
