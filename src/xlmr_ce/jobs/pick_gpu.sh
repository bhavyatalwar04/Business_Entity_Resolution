# Source from a PBS job: GPUs are not isolated per job on this cluster, so pin to the least-used idle GPU.
# A lock-protected claims file (~/.gpu_claims) stops two of our jobs that start together from picking the same GPU.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  CLAIMS=$HOME/.gpu_claims; HOST=$(hostname -s); JOB=${PBS_JOBID%%.*}
  exec 9>$CLAIMS.lock; flock 9
  touch $CLAIMS
  # keep only claims of our jobs that are still running
  tmp=$(mktemp); while read h g j; do qstat $j >/dev/null 2>&1 && echo "$h $g $j"; done < $CLAIMS > $tmp; mv $tmp $CLAIMS
  claimed=$(awk -v h=$HOST '$1==h {print $2}' $CLAIMS)
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  best=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | tr -d ' ' | \
         while IFS=, read i u m; do echo "$claimed" | grep -qx "$i" && continue; echo "$busy" | grep -q "$u" && n=1 || n=0; echo "$n,$m,$i"; done | sort -t, -k1,1n -k2,2n | head -1 | cut -d, -f3)
  echo "$HOST $best $JOB" >> $CLAIMS
  flock -u 9
  export CUDA_VISIBLE_DEVICES=$best
fi
echo "using GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,memory.used --format=csv,noheader)"
