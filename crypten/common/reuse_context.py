#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from contextlib import contextmanager


_TLS = threading.local()


def _get_tls_attr(name, default=None):
    return getattr(_TLS, name, default)


def _clear_tls_attr(name):
    if hasattr(_TLS, name):
        delattr(_TLS, name)


def set_current_beaver_tag(tag):
    _TLS.current_beaver_tag = tag


def get_current_beaver_tag():
    return _get_tls_attr("current_beaver_tag")


def clear_current_beaver_tag():
    _clear_tls_attr("current_beaver_tag")


@contextmanager
def use_beaver_tag(tag):
    previous = get_current_beaver_tag()
    set_current_beaver_tag(tag)
    try:
        yield
    finally:
        if previous is None:
            clear_current_beaver_tag()
        else:
            set_current_beaver_tag(previous)


def set_current_layer_tag(tag):
    _TLS.current_layer_tag = tag


def get_current_layer_tag():
    return _get_tls_attr("current_layer_tag")


def clear_current_layer_tag():
    _clear_tls_attr("current_layer_tag")


@contextmanager
def use_layer_tag(tag):
    previous = get_current_layer_tag()
    set_current_layer_tag(tag)
    try:
        yield
    finally:
        if previous is None:
            clear_current_layer_tag()
        else:
            set_current_layer_tag(previous)


def set_current_reuse_step(step_id):
    _TLS.current_reuse_step = step_id


def get_current_reuse_step(default=None):
    return _get_tls_attr("current_reuse_step", default)


def clear_current_reuse_step():
    _clear_tls_attr("current_reuse_step")


@contextmanager
def use_reuse_step(step_id):
    previous = get_current_reuse_step(default=None)
    set_current_reuse_step(step_id)
    try:
        yield
    finally:
        if previous is None:
            clear_current_reuse_step()
        else:
            set_current_reuse_step(previous)


def next_beaver_op_uid():
    op_uid = _get_tls_attr("beaver_op_uid", 0)
    _TLS.beaver_op_uid = op_uid + 1
    return op_uid

