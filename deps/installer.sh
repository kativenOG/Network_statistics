#!/bin/bash
cd ..;
python3 -m venv networkStats;
source networkStats/bin/activate && pip install -r deps/requirements.txt
POS=$(pwd);
echo $POS
echo "alias networkStats='source $POS/networkStats/bin/activate'" >> $HOME/.zshrc";
