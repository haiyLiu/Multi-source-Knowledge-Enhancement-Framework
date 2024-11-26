import os
import torch
import json
import random
import multiprocessing
import torch.backends.cudnn as cudnn
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp


from logger import logger
from config import args
from trainer import Trainer
       



def main():
    trainer = Trainer(args)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train()
    
    # trainer.generateQuery()

    

if __name__ == "__main__":
    torch.cuda.is_available()
    main()