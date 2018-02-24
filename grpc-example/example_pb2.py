# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: example.proto

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
  name='example.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\rexample.proto\"1\n\x04Move\x12\r\n\x05piece\x18\x01 \x01(\t\x12\x0c\n\x04\x66ile\x18\x02 \x01(\t\x12\x0c\n\x04rank\x18\x03 \x01(\r\"=\n\x12ReplyWithJudgement\x12\x14\n\x05reply\x18\x01 \x01(\x0b\x32\x05.Move\x12\x11\n\tjudgement\x18\x02 \x01(\x01\x32\x42\n\tChessProp\x12\x35\n\x11PointCounterpoint\x12\x05.Move\x1a\x13.ReplyWithJudgement\"\x00(\x01\x30\x01\x62\x06proto3')
)




_MOVE = _descriptor.Descriptor(
  name='Move',
  full_name='Move',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='piece', full_name='Move.piece', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='file', full_name='Move.file', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rank', full_name='Move.rank', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
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
  serialized_start=17,
  serialized_end=66,
)


_REPLYWITHJUDGEMENT = _descriptor.Descriptor(
  name='ReplyWithJudgement',
  full_name='ReplyWithJudgement',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reply', full_name='ReplyWithJudgement.reply', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='judgement', full_name='ReplyWithJudgement.judgement', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
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
  serialized_start=68,
  serialized_end=129,
)

_REPLYWITHJUDGEMENT.fields_by_name['reply'].message_type = _MOVE
DESCRIPTOR.message_types_by_name['Move'] = _MOVE
DESCRIPTOR.message_types_by_name['ReplyWithJudgement'] = _REPLYWITHJUDGEMENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Move = _reflection.GeneratedProtocolMessageType('Move', (_message.Message,), dict(
  DESCRIPTOR = _MOVE,
  __module__ = 'example_pb2'
  # @@protoc_insertion_point(class_scope:Move)
  ))
_sym_db.RegisterMessage(Move)

ReplyWithJudgement = _reflection.GeneratedProtocolMessageType('ReplyWithJudgement', (_message.Message,), dict(
  DESCRIPTOR = _REPLYWITHJUDGEMENT,
  __module__ = 'example_pb2'
  # @@protoc_insertion_point(class_scope:ReplyWithJudgement)
  ))
_sym_db.RegisterMessage(ReplyWithJudgement)



_CHESSPROP = _descriptor.ServiceDescriptor(
  name='ChessProp',
  full_name='ChessProp',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=131,
  serialized_end=197,
  methods=[
  _descriptor.MethodDescriptor(
    name='PointCounterpoint',
    full_name='ChessProp.PointCounterpoint',
    index=0,
    containing_service=None,
    input_type=_MOVE,
    output_type=_REPLYWITHJUDGEMENT,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CHESSPROP)

DESCRIPTOR.services_by_name['ChessProp'] = _CHESSPROP

# @@protoc_insertion_point(module_scope)