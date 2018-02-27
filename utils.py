# from Kelvin Guu's gtd
from torch.autograd import Variable


_GPUS_EXIST = True  # True by default


def try_gpu(x):
    """Try to put a Variable/Tensor/Module on GPU."""
    global _GPUS_EXIST

    if _GPUS_EXIST:
        try:
            return x.cuda()
        except (AssertionError, RuntimeError):
            # actually, GPUs don't exist
            print 'No GPUs detected. Sticking with CPUs.'
            _GPUS_EXIST = False
            return x
    else:
        return x


def GPUVariable(data):
    return try_gpu(Variable(data, requires_grad=False))
