MAX=11575
STEP=11

# Loop over chunks of 11 rows
for (( start=0; start<MAX; start+=STEP )); do
  end=$(( start + STEP ))
  if (( end > MAX )); then
    end=$MAX
  fi
  echo "→ Processing rows ${start}:${end} of ${MAX}"
  python3 multimolecule_rinalmo.py --start "${start}" --end "${end}"
done