import time
from turtle import forward
# import numpy as np
import pandas
from functools import partial
# import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def make_unifor_design_grid(factor, level, use_star=False)->list:
    pass


# def iter_param(grid, param_data)->iter:
#     pass

class Uniform_Design_Experimentation:

    def __init__(self, filename=None, log=True):
        self.filename = filename or "./uniform_design_exp_log"
        self.log = log


    def setup(self, train_fn, static_param=None,**kwargs):
        assert static_param is None or isinstance(static_param, dict), \
            f"static_param must be dict type, instead of {type(static_param)}"
        params = {}
        self.factor = len(kwargs)
        self.level = None
        for key, value in kwargs:
            params[key] = None
            self.level = self.level or len(value)
            assert self.level == len(value), "each factor must set same number of levels"
        self.param_data = kwargs 
        self.params = params
        self.grid = make_unifor_design_grid(self.factor, self.level)
        self.trainfn = partial(train_fn, **static_param)

    def __enter__(self):
        return self

    def train(self):
        self.results = []
        self.ans_able = True
        for col in self.grid:
            self.make_params(col)
            start = time.time()
            try:
                ret, weight_dir = self.trainfn(**self.params)
                self.ans_able = False
            except Exception as e:
                print("\n\nError!")
                print("happen while:", self.params)
                print("Error msg:", e)
                ret = None
                weight_dir = None
            
            self.results.append((self.params, ret, time.time()-start, time.ctime(), weight_dir))
            self.clean_params()
    
    def clean_params(self):
        for key in self.params.keys():
            self.params[key] = None
    
    def make_params(self, col):
        for key, index in zip(self.params, col):
            self.params[key] = self.param_data[key][index]
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.make_result_table()
        self.ans(self)
        if self.log:
            self.save()
    
    def ans(self):
        if self.ans_able:
            linreg:LinearRegression = LinearRegression().fit(self.table_x.value.astype('float32'), self.table_x.values)
            # model = linreg
            self.weights = linreg.coef_
            self.bias = linreg.intercept_
            self.rank = linreg.score(self.table_x.value.astype('float32'), self.table_x.values)

    def make_result_table(self):
        table_x = {}
        table_y = []
        for item in self.results:
            for key, value in item[0].items():
                table_x[key] = table_x.get(key, list())
                table_x[key].append(value)
            table_y.append(item[0])
        self.table_x = pandas.DataFrame(table_x)
        self.table_y = pandas.Series(table_y, name="results")
    
    def save(self):
        tosave = self.table_x.copy()
        tosave["metric"] = self.table_y
        tosave.to_csv(self.filename + "_results.csv")
        print(f"each experiment and it's result has saved in f{self.filename + '_results.csv'}")
        print(tosave)
        tosave_weights = {}
        for name, weight in zip(self.table_x.columns, self.weights):
            tosave_weights[name] = weight
        tosave_weights["bias"] = self.bias
        tosave_weights["rank"] = self.rank
        tosave_weights = pandas.DataFrame(tosave_weights)
        tosave_weights.to_csv(self.filename + "_weights.csv")
        print(f"Linear Regression result has saved in f{self.filename + '_weights.csv'}")
        print(tosave_weights)


if __name__ == "__main__":
    import torch
    from torch import nn
    data_x = torch.randn(1024 * 8, 512)
    data_y = data_x.mean(dim=-1) * 2 + 3. + torch.randn(1024 * 8) / 10
    
    class MyModel(nn.Module):
        def __init__(self, depth, inner_dims, dropout):
            super().__init__()
            self.model = nn.Sequential()
            for i in range(depth - 1):
                self.model.append(nn.Linear(inner_dims if i != 0 else 512, inner_dims))
                self.model.append(nn.ReLU())
                self.model.append(nn.Dropout(dropout))
            self.model.append(nn.Linear(inner_dims if depth > 1 else 512, 1))
        
        def forward(self, x):
            return self.model(x)
        
        def calc_loss(self, yhat, y):
            # self.eval()
            # yhat = self.model(x)
            mse = ((y - yhat) **2) .mean()
            return mse

        def calc_metric(self, x, y):
            self.eval()
            yhat = self.model(x)
            return self.calc_loss(yhat, y)
        
    def train(lr, beta1, beta2, eps, depth, inner_dims, dropout, epochs, device="cuda:2"):
        device = torch.device(device)
        model = MyModel(depth, inner_dims, dropout).to(device)
        x = data_x.to(device)
        y = data_y.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr, betas=[beta1, beta2], eps=eps)

        # for epoch in range(epochs):
        with tqdm(range(epochs), desc=f"Training", leave=False) as epoch:
            for _ in epoch:
                model.train()
                yhat = model(x)
                loss = model.calc_loss(yhat, y)
                loss.backward()
                epoch.set_postfix(loss=loss.item())
                optimizer.step()
                model.zero_grad()
        metric = model.calc_metric(x, y)
        print(list(model.parameters()))
        return metric, ""

    ret, save_dir = train(1e-3, 0.9, 0.98, 1e-8, 1, 16, 0.0, 3500)
    print(ret)


        

                
                
            


    
        


    
        

