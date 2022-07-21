from ._parsing_for_parallel import parsing_for_parallel

from utils import ParallelProcess

def parsing(X, w2indx, **params):

    print("parsing the tokenized sentences...")
    n_cores = params.get("n_cores")
    results = ParallelProcess(X,
                                parsing_for_parallel,
                                w2indx,
                                n_cores = n_cores)

    return results