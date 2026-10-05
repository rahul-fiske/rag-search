"""Tests that run anywhere: fakes stand in for docling, the models and Apple Vision.

Tier A of docs/design/test-strategy.md: importing the heavy libraries is blocked and Hugging Face is offline
while these tests run (tests/guard.py), so they give the same result on a machine that has docling and torch
and on one that has not."""

from tests import guard

guard.block()
