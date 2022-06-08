from torch import optim

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class DebugLog:

    def __init__(self, printfn=None):
        self.print = print if printfn is None else printfn


    def info(self, msg):
        self.print("info:", msg)

    def error(self, msg):
        self.print("error:", msg)
    
  