
class Trainer():
    """
        Can be use to train various models and different types of data.
    """

    def __init__(self, 
            model, 
            criterion, 
            optimizer, 
            train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
            max_epochs: int =1000, 
            patience: int =1000,
            ):
        super(Trainer).__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.patience = patience

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        





if __name__ == "__main__":
    pass