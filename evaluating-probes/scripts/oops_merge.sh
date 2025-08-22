#!/bin/bash

src="4-mask-pred-balanced"
dst="4-mask-pred-auc-train-eval-balanced"

# Merge files safely without overwriting existing ones
for dir in gen_eval test_eval trained val_eval; do
    if [ -d "$src/$dir" ]; then
        mkdir -p "$dst/$dir"
        find "$src/$dir" -type f -exec cp --update=none {} "$dst/$dir" \;
    fi
done

echo "Files merged safely without overwriting."
