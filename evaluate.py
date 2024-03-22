# Splits a dataset into t=0 test/train portions containing half the data
# Requires: dataset is a temporally sorted pytorch tensor
#           split is a decimal in [0, 1]
def get_t0_split(dataset, split=0.75):
    t0 = dataset[:dataset.size/2]
    t0_train = t0[:split * t0.size]
    t0_test = t0[split * t0.size + 1:]
    return [t0_train, t0_test]
    
# Splits a dataset into test/train portions then t=1,...n,
#    each contain the data from one month
# Requires: dataset is a temporally sorted pytorch tensor
#           split is a decimal in [0, 1]
def get_monthly_split(dataset, split=0.75):
    t0 = dataset[:dataset.length()/2]
    t0_train = t0[:0.75 * t0.length()]
    t0_test = t0[0.75 * t0.length() + 1:]
    # For each remaining month
        # Do the same
    
# Evaluates a model in our continual learning setting
# Requires that model has a .train function
def evaluate(model, dataset, metrics):
    pass
    # t=0 
