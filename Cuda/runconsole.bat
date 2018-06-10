nvprof.exe --print-gpu-trace --events shared_load,shared_store --metrics achieved_occupancy,ipc,executed_ipc,duration --aggregate-mode on  --csv --log-file ".\output" ".\x64\Release\Cuda.exe" --event-collection-mode kernel --kernels "matrixMulCUDA"

:: -n=1024 -s=2 -m=10