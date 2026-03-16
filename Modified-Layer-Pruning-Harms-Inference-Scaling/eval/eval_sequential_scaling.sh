export OPENAI_KEY=""
export OPENAI_API_KEY="$OPENAI_KEY"
export PROCESSOR="gpt-4o-mini"

TOKENS=(512 1024 2048 4096 8192)
PRUNE_LAYERS=(3) # Change up according to your needs

declare -A TASKS
TASKS=(
  ["aime24_nofigures"]="aime"
  ["gpqa_diamond_openai"]="gpqa"
  ["openai_math"]="math"
)
# PUT IN THE MODEL PATH LINKS 
for PRUNE in "${PRUNE_LAYERS[@]}"; do  
  if [ "$PRUNE" -eq 1 ]; then
      MODEL_PATH=""
  elif [ "$PRUNE" -eq 2 ]; then
      MODEL_PATH=""
  elif [ "$PRUNE" -eq 3 ]; then
      MODEL_PATH="" 
  else
      echo "Invalid layer number"
      exit 1
  fi
  for TASK in "${!TASKS[@]}"; do
    for TOK in "${TOKENS[@]}"; do
      SUBDIR="${TASKS[$TASK]}"
      OUTPUT_PATH_BASE="" #### PUT IN PATH YOU WANT TO SAVE THE RESULTS
      mkdir -p "$OUTPUT_PATH_BASE"

      for SEED in 7 11 42; do # 7 11 42
        OUTPUT_PATH="${OUTPUT_PATH_BASE}/seed${SEED}"
        mkdir -p "$OUTPUT_PATH"

        PROCESSOR="$PROCESSOR" lm_eval \
          --model vllm \
          --model_args pretrained="${MODEL_PATH}",dtype=bfloat16,tensor_parallel_size=1 \
          --tasks $TASK \
          --batch_size auto \
          --apply_chat_template \
          --output_path "$OUTPUT_PATH" \
          --log_samples \
          --gen_kwargs "temperature=1.0,seed=${SEED},max_gen_toks=32768,max_tokens_thinking=${TOK}"
      done
    done
  done
done



















