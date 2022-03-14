import torch

class general_chunk_iterator():
    def __init__(self,X,y,shuffle,batch_size):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            result = (self.it_X[self._index],self.it_y[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return general_chunk_iterator(X =self.dataset.X,
                              y = self.dataset.y,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len

class general_dataset():
    def __init__(self,X_tr,y_tr,X_val,y_val,X_test,y_test):
        self.train_X=torch.from_numpy(X_tr).float()
        self.train_y=torch.from_numpy(y_tr).float()
        self.val_X=torch.from_numpy(X_val).float()
        self.val_y=torch.from_numpy(y_val).float()
        self.test_X=torch.from_numpy(X_test).float()
        self.test_y=torch.from_numpy(y_test).float()

    def set(self, mode='train'):
        self.X = getattr(self, f'{mode}_X')
        self.y = getattr(self, f'{mode}_y')

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self):
        return self.X.shape[0]

class train_object_regression():
    def __init__(self,job_params,hparamspace):
        pass
        #load appropriate data
        #get dataloaders if necessary
        #define hparamspace given a model

    def __call__(self):
        pass
        #call

    def hypertune(self):
        pass

class train_object_classification():
    def __init__(self,job_params,hparamspace):
        pass
        #load appropriate data
        #get dataloaders if necessary
        #define hparamspace given a model

    def __call__(self):
        pass
        #call

    def hypertune(self):
        pass







