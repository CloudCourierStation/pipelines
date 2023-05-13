# Copyright 2023 The Kubeflow Authors. All Rights Reserved.
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

import re

from google_cloud_pipeline_components import utils
from kfp import components
from kfp import dsl
import yaml


# preserve function docstrings for components by setting component decorators to passthrough decorators
# also enables autodoc to document the components as functions without using the autodata directive (https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autodata)
def first_order_passthrough_decorator(func):
  func._is_component = True
  return func


def second_order_passthrough_decorator(*args, **kwargs):
  def decorator(func):
    func._is_component = True
    return func

  return decorator


def load_from_file(path: str):
  with open(path) as f:
    contents = f.read()
    component_dict = yaml.safe_load(contents)
  comp = components.load_component_from_text(contents)
  description = component_dict.get('description', '')
  comp.__doc__ = description
  return comp


utils.gcpc_output_name_converter = second_order_passthrough_decorator
dsl.component = second_order_passthrough_decorator
dsl.container_component = first_order_passthrough_decorator
components.load_component_from_file = load_from_file


class OutputPath(dsl.OutputPath):

  def __repr__(self) -> str:
    type_string = getattr(self.type, '__name__', '')
    return f'dsl.OutputPath({type_string})'


dsl.OutputPath = OutputPath


class InputClass:

  def __getitem__(self, type_) -> str:
    type_string = getattr(type_, 'schema_title', getattr(type_, '__name__', ''))
    return f'dsl.Input[{type_string}]'


Input = InputClass()

dsl.Input = Input


class OutputClass:

  def __getitem__(self, type_) -> str:
    type_string = getattr(type_, 'schema_title', getattr(type_, '__name__', ''))
    return f'dsl.Output[{type_string}]'


Output = OutputClass()

dsl.Output = Output

# order from earliest to latest
# start with 2.0.0b3, which is the first time we're using the new theme
V2_DROPDOWN_VERSIONS = ['2.0.0b3']

# The short X.Y version
# update for each release
LATEST_VERSION = V2_DROPDOWN_VERSIONS[-1]

# The full version, including alpha/beta/rc tags
release = LATEST_VERSION

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'm2r2',
    'sphinx_immaterial',
    'autodocsumm',
]
autodoc_default_options = {
    'members': True,
    'member-order': 'alphabetical',
    'imported-members': True,
    'undoc-members': True,
    'show-inheritance': False,
    'autosummary': False,
}

html_theme = 'sphinx_immaterial'
html_title = 'Google Cloud Pipeline Components Reference Documentation'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'icon': {
        'repo': 'fontawesome/brands/github',
    },
    'repo_url': 'https://github.com/kubeflow/pipelines/tree/master/components/google-cloud',
    'repo_name': 'pipelines',
    'repo_type': 'github',
    'edit_uri': 'https://github.com/kubeflow/pipelines/tree/master/components/google-cloud/docs/source',
    'globaltoc_collapse': True,
    'features': [
        'navigation.expand',
        # "navigation.tabs",
        # "toc.integrate",
        'navigation.sections',
        # "navigation.instant",
        # "header.autohide",
        'navigation.top',
        # "navigation.tracking",
        'search.highlight',
        'search.share',
        'toc.follow',
        'toc.sticky',
    ],
    'font': {'text': 'Open Sans'},
    'version_dropdown': True,
    'version_info': [
        {
            'version': f'https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-{version}',
            'title': version,
            'aliases': [],
        }
        for version in reversed(V2_DROPDOWN_VERSIONS)
    ],
    # "toc_title_is_page_title": True,
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'GoogleCloudPipelineComponentsDocs'


def component_grouper(app, what, name, obj, section, parent):
  if getattr(obj, '_is_component', False):
    return 'Components'


def autodoc_skip_member(app, what, name, obj, skip, options):
  skip = True
  if name == 'create_custom_training_job_op_from_component':
    return skip


def setup(app):
  app.connect('autodocsumm-grouper', component_grouper)
  app.connect('autodoc-skip-member', autodoc_skip_member)
