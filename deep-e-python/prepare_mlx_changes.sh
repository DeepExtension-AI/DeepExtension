#!/bin/bash

set -e

missing_files=()

# Updated file paths
DATASETS_FILE="mlx_lm/tuner/datasets.py"
TRAINER_FILE="mlx_lm/tuner/trainer.py"

# Check file existence
[[ -f "$DATASETS_FILE" ]] || missing_files+=("$DATASETS_FILE")
[[ -f "$TRAINER_FILE" ]] || missing_files+=("$TRAINER_FILE")

if [ ${#missing_files[@]} -ne 0 ]; then
  echo "fatal error: the following file(s) do not exist:"
  for file in "${missing_files[@]}"; do
    echo "  - $file"
  done
  exit 1
fi

modified=0

update_datasets_load_local_dataset_py() {
  local start="def load_local_dataset("
  local replaced=0
  local temp_file="${DATASETS_FILE}.tmp"

  awk -v replaced_flag=0 '
    BEGIN { in_target_func = 0 }
    $0 ~ /^def load_local_dataset\(/ { in_target_func = 1 }
    in_target_func == 1 && /^def / && $0 !~ /^def load_local_dataset\(/ { in_target_func = 0 }

    in_target_func && $0 ~ /^[[:space:]]*return train, valid, test/ {
      sub(/    return train, valid, test/, "    return train, train, None")
      replaced_flag = 1
    }

    { print }
    END {
      if (replaced_flag == 1) {
        print "Modified return inside load_local_dataset" > "/dev/stderr"
        exit 0
      } else {
        print "correct: return train, valid, test not found inside load_local_dataset" > "/dev/stderr"
        exit 1
      }
    }
  ' "$DATASETS_FILE" > "$temp_file"

  if [[ $? -eq 0 ]]; then
    mv "$temp_file" "$DATASETS_FILE"
    modified=1
    return 1
  else
    rm -f "$temp_file"
    echo "no replacement required"
    return 0
  fi
}

# Update datasets.py
update_datasets_py() {

  update_datasets_load_local_dataset_py

  if grep -q '^    names = ("train", "valid", "test")' "$DATASETS_FILE"; then
    sed -i '' 's/^    names = ("train", "valid", "test")/    #names = ("train", "valid", "test")/' "$DATASETS_FILE"
    modified=1
  fi

  if grep -q '^    train, valid, test = \[load_subset(data_path \/ f"{n}.jsonl") for n in names\]' "$DATASETS_FILE"; then
    sed -i '' 's/^    train, valid, test = \[load_subset(data_path \/ f"{n}.jsonl") for n in names\]/    train = load_subset(data_path)/' "$DATASETS_FILE"
    modified=1
  fi

  #if grep -q '^    return train, valid, test' "$DATASETS_FILE"; then
  #  sed -i '' 's/^    return train, valid, test/    return train, train, None/' "$DATASETS_FILE"
  #  modified=1
  #fi

  if [[ $modified -eq 1 ]]; then
    echo "Modified: $DATASETS_FILE"
    return 0
  fi

  # Final check if all desired patterns are present
  if grep -q '^    #names = ("train", "valid", "test")' "$DATASETS_FILE" &&
     grep -q '^    train = load_subset(data_path)' "$DATASETS_FILE" &&
     grep -q '^    return train, train, None' "$DATASETS_FILE"; then
    echo "$DATASETS_FILE already modified and verified"
    return 1
  else
    echo "Error: unexpected format in $DATASETS_FILE"
    exit 1
  fi
}


# Update trainer.py
update_trainer_py() {
  if grep -q '    mx.set_wired_limit(mx.metal.device_info()\["max_recommended_working_set_size"\])' "$TRAINER_FILE"; then
    sed -i '' 's/^    mx.set_wired_limit(mx.metal.device_info()\["max_recommended_working_set_size"\])/    mx.set_wired_limit(8 * 1024 * 1024 * 1024)/' "$TRAINER_FILE"
    echo "Modified: $TRAINER_FILE"
    return 0
  elif ! grep -q '    mx.set_wired_limit(mx.metal.device_info()\["max_recommended_working_set_size"\])' "$TRAINER_FILE"; then
    echo "$TRAINER_FILE already modified and verified"
    return 1
  else
    echo "Error, unexpected format in $TRAINER_FILE"
    exit 1
  fi
}

datasets_status=1
trainer_status=1

update_datasets_py && datasets_status=0 || datasets_status=$?
update_trainer_py && trainer_status=0 || trainer_status=$?

if [[ $datasets_status -eq 0 || $trainer_status -eq 0 ]]; then
  echo "replacement is successful"
elif [[ $datasets_status -eq 1 && $trainer_status -eq 1 ]]; then
  echo "replacement was done and checked with success"
fi
