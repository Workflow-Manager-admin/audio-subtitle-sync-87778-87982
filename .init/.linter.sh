#!/bin/bash
cd /home/kavia/workspace/code-generation/audio-subtitle-sync-87778-87982/BackendService
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

