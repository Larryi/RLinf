# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compatibility handling for installed but unloadable FlashAttention wheels."""

import sys


def disable_broken_flash_attn() -> None:
    """Make Transformers ignore FlashAttention when its extension cannot load."""
    try:
        import flash_attn  # noqa: F401
    except (ImportError, OSError):
        for module_name in tuple(sys.modules):
            if module_name == "flash_attn" or module_name.startswith("flash_attn."):
                sys.modules.pop(module_name, None)

        import transformers.utils
        from transformers.utils import import_utils

        import_utils.is_flash_attn_2_available = lambda: False
        import_utils.is_flash_attn_3_available = lambda: False
        transformers.utils.is_flash_attn_2_available = lambda: False
        transformers.utils.is_flash_attn_3_available = lambda: False
