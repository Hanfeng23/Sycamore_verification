# Use TF32
export NVIDIA_TF32_OVERRIDE=1
echo "NVIDIA_TF32_OVERRIDE   = $NVIDIA_TF32_OVERRIDE"

# after cuTensor v1.6.1.4
export CUTENSOR_ENABLE_3XTF32=1
echo "CUTENSOR_ENABLE_3XTF32 = $CUTENSOR_ENABLE_3XTF32"

compute_type="fp32"

if [ "$NVIDIA_TF32_OVERRIDE" = "1" ]; then
  compute_type="tf32"
fi

if [ "$CUTENSOR_ENABLE_3XTF32" = "1" ]; then
  compute_type="3xtf32"
fi

DEMO_FILE=demo.py


install_cutensor_python() {
  echo "Installing cuTensor python API..."
  pip install ${SYCAMORE_DEMO_HOME}/cutensor_python/
}


START_TASK=0
TASK_NUM=10

N_CYCLE=20

BITSTRINGS='-bitstrings 3000000'
PATH_FILE='-path_file scheme_n53_m20'

if [ "$N_CYCLE" = "18" ]; then
  BITSTRINGS='-bitstrings 1048576'
  PATH_FILE='-path_file scheme_n53_m18'
elif [ "$N_CYCLE" = "20" ]; then
  BITSTRINGS='-bitstrings 3000000'
  PATH_FILE='-path_file scheme_n53_m20'
else
  echo "layer number"
fi

if [ -z $1 ]; then
  echo "Running ${DEMO_FILE} ..."
  python ${SYCAMORE_DEMO_HOME}/${DEMO_FILE} -get_time -task_num 5 ${BITSTRINGS} ${PATH_FILE}
elif [ "$1" = "-c" ]; then
  echo "Running ${DEMO_FILE}.py with checking mode..."
  python ${SYCAMORE_DEMO_HOME}/${DEMO_FILE} -get_time -task_num 3 -check_results True ${BITSTRINGS} ${PATH_FILE}
elif [ "$1" = "-t" ]; then
  echo "Timing each kernels..."
  python ${SYCAMORE_DEMO_HOME}/${DEMO_FILE} -get_time ${BITSTRINGS} ${PATH_FILE} -task_num 1 -get_timing_kernels True -check_results True
elif [ "$1" = "-i" ]; then
  install_cutensor_python
elif [ "$1" = "-mg" ]; then
  if [ -z $2 ]; then
    START_TASK=0
  else
    START_TASK=$2
  fi
  echo "Running ${DEMO_FILE} on multigpus ..."
  for i in 0 1 2 3 4 5 6 7; do
    echo "START_TASK = ${START_TASK}, TASK_NUM = ${TASK_NUM} on GPU $i"
    export CUDA_VISIBLE_DEVICES=$i
    nohup python ${SYCAMORE_DEMO_HOME}/${DEMO_FILE} -get_time -start_task ${START_TASK} -task_num ${TASK_NUM} -cuda 0 >${SYCAMORE_DEMO_HOME}/log_taskid_${START_TASK}.txt 2>&1 &
    START_TASK=$(expr ${START_TASK} + ${TASK_NUM})
  done
fi