#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./scripts/train.sh <task_name>"
  exit 1
fi

export MINEDOJO_HEADLESS=1
python expr_own.py \
    --configs minedojo \
    --task minedojo_$1 \
    --logdir ./logdir \
    --resume_path $2

# 只有当提供了第二个参数($2)时，才添加 resume_path
if [ -n "$2" ]; then
    CMD="$CMD --resume_path \"$2\""
fi

# 打印并执行命令
echo "Executing command: $CMD"
eval $CMD