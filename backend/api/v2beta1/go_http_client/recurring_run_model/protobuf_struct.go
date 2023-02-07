// Code generated by go-swagger; DO NOT EDIT.

package recurring_run_model

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	strfmt "github.com/go-openapi/strfmt"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/swag"
	"github.com/go-openapi/validate"
)

// ProtobufStruct `Struct` represents a structured data value, consisting of fields
// which map to dynamically typed values. In some languages, `Struct`
// might be supported by a native representation. For example, in
// scripting languages like JS a struct is represented as an
// object. The details of that representation are described together
// with the proto support for the language.
//
// The JSON representation for `Struct` is JSON object.
// swagger:model protobufStruct
type ProtobufStruct struct {

	// Unordered map of dynamically typed values.
	Fields map[string]ProtobufValue `json:"fields,omitempty"`
}

// Validate validates this protobuf struct
func (m *ProtobufStruct) Validate(formats strfmt.Registry) error {
	var res []error

	if err := m.validateFields(formats); err != nil {
		res = append(res, err)
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}

func (m *ProtobufStruct) validateFields(formats strfmt.Registry) error {

	if swag.IsZero(m.Fields) { // not required
		return nil
	}

	for k := range m.Fields {

		if err := validate.Required("fields"+"."+k, "body", m.Fields[k]); err != nil {
			return err
		}
		if val, ok := m.Fields[k]; ok {
			if err := val.Validate(formats); err != nil {
				return err
			}
		}

	}

	return nil
}

// MarshalBinary interface implementation
func (m *ProtobufStruct) MarshalBinary() ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	return swag.WriteJSON(m)
}

// UnmarshalBinary interface implementation
func (m *ProtobufStruct) UnmarshalBinary(b []byte) error {
	var res ProtobufStruct
	if err := swag.ReadJSON(b, &res); err != nil {
		return err
	}
	*m = res
	return nil
}
