import copy
from lib2to3.pgen2 import token
import time
import torch
import torch.nn as nn
from torch.utils.data import dataset
from torch import Tensor
import math

import sys
sys.path.append("../")
from utils.utils import generate_square_subsequent_mask, get_batch

def train_by_epoch(model: nn.Module, train_data: dataset, bptt: int, criterion, optimizer, ntokens, scheduler, device, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt=bptt)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return model

def evaluate(model: nn.Module, eval_data: Tensor, bptt, ntokens, criterion, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt=bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def train(model: nn.Module, train_data: dataset, bptt: int, criterion, optimizer, ntokens, scheduler, device, epochs, eval_data):
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        ## train model for 1 epoch
        model = train_by_epoch(model=model, train_data=train_data, bptt=bptt, \
                criterion=criterion, optimizer=optimizer, ntokens=ntokens, \
                scheduler=scheduler, device=device, epoch=epoch)
        
        ## evaluate model
        val_loss = evaluate(model=model, eval_data=eval_data, bptt=bptt, \
                ntokens=ntokens, criterion=criterion, device=device)
        
        
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()
    return best_model