#!/bin/bash
cd ..;
python3 -m venv networkStats;
source networkStats/bin/activate && pip install -r deps/requirements.txt
