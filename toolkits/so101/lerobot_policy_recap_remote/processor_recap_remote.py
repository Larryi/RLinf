# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Identity processors for physical absolute actions served by RLInf."""

from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


def make_recap_remote_pre_post_processors(config, dataset_stats=None):
    del config, dataset_stats
    preprocessor = PolicyProcessorPipeline(
        steps=[],
        name="recap_remote_preprocessor",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[],
        name="recap_remote_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
