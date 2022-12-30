# coding: utf-8

"""
    Kubeflow Pipelines API

    This file contains REST API specification for Kubeflow Pipelines. The file is autogenerated from the swagger definition.

    Contact: kubeflow-pipelines@google.com
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from kfp_server_api.configuration import Configuration


class V2beta1RuntimeStatus(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'update_time': 'datetime',
        'state': 'V2beta1RuntimeState',
        'error': 'RpcStatus'
    }

    attribute_map = {
        'update_time': 'update_time',
        'state': 'state',
        'error': 'error'
    }

    def __init__(self, update_time=None, state=None, error=None, local_vars_configuration=None):  # noqa: E501
        """V2beta1RuntimeStatus - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._update_time = None
        self._state = None
        self._error = None
        self.discriminator = None

        if update_time is not None:
            self.update_time = update_time
        if state is not None:
            self.state = state
        if error is not None:
            self.error = error

    @property
    def update_time(self):
        """Gets the update_time of this V2beta1RuntimeStatus.  # noqa: E501

        Update time of this state.  # noqa: E501

        :return: The update_time of this V2beta1RuntimeStatus.  # noqa: E501
        :rtype: datetime
        """
        return self._update_time

    @update_time.setter
    def update_time(self, update_time):
        """Sets the update_time of this V2beta1RuntimeStatus.

        Update time of this state.  # noqa: E501

        :param update_time: The update_time of this V2beta1RuntimeStatus.  # noqa: E501
        :type update_time: datetime
        """

        self._update_time = update_time

    @property
    def state(self):
        """Gets the state of this V2beta1RuntimeStatus.  # noqa: E501


        :return: The state of this V2beta1RuntimeStatus.  # noqa: E501
        :rtype: V2beta1RuntimeState
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this V2beta1RuntimeStatus.


        :param state: The state of this V2beta1RuntimeStatus.  # noqa: E501
        :type state: V2beta1RuntimeState
        """

        self._state = state

    @property
    def error(self):
        """Gets the error of this V2beta1RuntimeStatus.  # noqa: E501


        :return: The error of this V2beta1RuntimeStatus.  # noqa: E501
        :rtype: RpcStatus
        """
        return self._error

    @error.setter
    def error(self, error):
        """Sets the error of this V2beta1RuntimeStatus.


        :param error: The error of this V2beta1RuntimeStatus.  # noqa: E501
        :type error: RpcStatus
        """

        self._error = error

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V2beta1RuntimeStatus):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V2beta1RuntimeStatus):
            return True

        return self.to_dict() != other.to_dict()