export MASTER_ADDR=0.0.0.0
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=1,2,3
export WORLD_SIZE=2
export NCCL_DEBUG=INFO
RANK=0 LOCAL_RANK=0 python ddp.py &
RANK=1 LOCAL_RANK=1 python ddp.py &
RANK=2 LOCAL_RANK=2 python ddp.py &
