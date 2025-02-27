# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/src/resource_handle.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboard/src/resource_handle.proto',
  package='tensorboard',
  syntax='proto3',
  serialized_pb=_b('\n%tensorboard/src/resource_handle.proto\x12\x0btensorboard\"m\n\x0eResourceHandle\x12\x0e\n\x06\x64\x65vice\x18\x01 \x01(\t\x12\x11\n\tcontainer\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x11\n\thash_code\x18\x04 \x01(\x04\x12\x17\n\x0fmaybe_type_name\x18\x05 \x01(\tB4\n\x18org.tensorflow.frameworkB\x13ResourceHandleProtoP\x01\xf8\x01\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_RESOURCEHANDLE = _descriptor.Descriptor(
  name='ResourceHandle',
  full_name='tensorboard.ResourceHandle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device', full_name='tensorboard.ResourceHandle.device', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='container', full_name='tensorboard.ResourceHandle.container', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboard.ResourceHandle.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hash_code', full_name='tensorboard.ResourceHandle.hash_code', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maybe_type_name', full_name='tensorboard.ResourceHandle.maybe_type_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=163,
)

DESCRIPTOR.message_types_by_name['ResourceHandle'] = _RESOURCEHANDLE

ResourceHandle = _reflection.GeneratedProtocolMessageType('ResourceHandle', (_message.Message,), dict(
  DESCRIPTOR = _RESOURCEHANDLE,
  __module__ = 'tensorboard.src.resource_handle_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.ResourceHandle)
  ))
_sym_db.RegisterMessage(ResourceHandle)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\023ResourceHandleProtoP\001\370\001\001'))
# @@protoc_insertion_point(module_scope)
