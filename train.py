import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from data_preprocess import DISCONNECTED_CURVATURE


def train_model(model, optimizer, data_o, train_loader, val_loader, test_loader, args, use_curvature=True, weight_fct='exp', curriculum=False, hard_weight=4.0, temperature=0.2, documentation=False, return_details=False):
    m = torch.nn.Sigmoid() 
    loss_fct = torch.nn.BCELoss()
    loss_no_reduction = torch.nn.BCELoss(reduction='none')
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []
    max_auc = 0

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        
    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    
    #for curriculum learning -- positive edges: threshold narrows from
    #permissive (all edges "hard") to strict (only true bottlenecks)
    start_thresh = 1.0
    end_thresh = 0.0
    #for curriculum learning -- negative pairs: mirrored schedule (same
    #idea, opposite direction), since high proxy curvature (shared
    #neighbors) is what makes a negative pair "hard" here -- the opposite of
    #positives, where low curvature is hard. Proxy curvature for negatives
    #can't actually reach 1.0 in practice (that would require two
    #non-adjacent nodes with identical neighborhoods), so mirroring
    #positives' literal [0,1] bound left the threshold above every real
    #negative's curvature for most of training, making it inert. Calibrate
    #both ends to what negatives in THIS training set actually reach
    #instead -- the 5th/95th percentiles, not the raw min/max, so outlier
    #pairs don't stretch the whole schedule.
    if curriculum:
        neg_mask = torch.as_tensor(np.asarray(train_loader.dataset.label)) == 0
        neg_curv = train_loader.dataset.curvature[neg_mask]
        # exclude the disconnected-pair floor (DISCONNECTED_CURVATURE) and
        # any leftover inf (a loader that hasn't been updated to compute
        # real negative curvature) -- neither belongs in the calibration
        real_neg_curv = neg_curv[torch.isfinite(neg_curv) & (neg_curv > DISCONNECTED_CURVATURE)]
        if len(real_neg_curv) > 0:
            start_thresh_neg = torch.quantile(real_neg_curv, 0.05).item()
            end_thresh_neg = torch.quantile(real_neg_curv, 0.95).item()
        else:
            # no real negative curvature to calibrate against -- keep the
            # ramp inert (start == end) rather than guess a fixed bound
            start_thresh_neg = end_thresh_neg = 0.0
    else:
        # unused when curriculum is off (the weighting block below is
        # itself gated on curriculum), so no fixed value needed here either
        start_thresh_neg = end_thresh_neg = 0.0
    for epoch in range(args.epochs):
        t = time.time()
        if documentation:
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []
        #for curriculum learning -- straight linear ramp across the whole run
        progress = epoch / (args.epochs - 1) if args.epochs > 1 else 1.0
        current_threshold = start_thresh + progress * (end_thresh - start_thresh)
        current_threshold_neg = start_thresh_neg + progress * (end_thresh_neg - start_thresh_neg)


        for i, (label, inp, curv_batch) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()
                curv_batch = curv_batch.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(data_o,inp, use_curvature=use_curvature, weight_fct=weight_fct)
            log = torch.squeeze(m(output))
            loss1 = loss_fct(log, label.float())
            loss_simple = loss_no_reduction(log, label.float())
            
            #calculate loss dependent on weight function
            weights = torch.ones_like(loss_simple)
            if curriculum:
                curv_flat = curv_batch.flatten()
                is_pos = label.flatten() == 1
                # positives: hard = low curvature (bottleneck edge) -- full
                # boost well below the threshold, no boost well above it
                pos_hard_mask = torch.sigmoid((current_threshold - curv_flat) / temperature)
                # negatives: hard = high proxy curvature (looks like it should
                # be an edge) -- mirrored, increasing sigmoid
                neg_hard_mask = torch.sigmoid((curv_flat - current_threshold_neg) / temperature)
                hard_mask = torch.where(is_pos, pos_hard_mask, neg_hard_mask)
                # a non-finite curvature means this loader never computed a
                # real value for this sample (e.g. an inf placeholder from a
                # negative-sampling path that hasn't been updated) -- treat
                # as no signal, not as inf saturating neg_hard_mask to 1
                hard_mask = torch.where(torch.isfinite(curv_flat), hard_mask, torch.zeros_like(hard_mask))
                weights = 1.0 + hard_weight * hard_mask
                loss_train = (loss_simple * weights).mean()

            if not curriculum:
                loss_train = loss_simple.mean()

            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + log.flatten().tolist()

            if documentation:
                if i % 100 == 0:
                    print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                        loss_train.cpu().detach().numpy()) + ' current threshold:' + str(current_threshold))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        if not args.fastmode:
            roc_val, prc_val, f1_val, loss_val, _ = test(model, val_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=False)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val
                
            if documentation:
                print('epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'auroc_train: {:.4f}'.format(roc_train),
                          'loss_val: {:.4f}'.format(loss_val.item()),
                          'auroc_val: {:.4f}'.format(roc_val),
                          'auprc_val: {:.4f}'.format(prc_val),
                          'f1_val: {:.4f}'.format(f1_val),
                          'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
                
    #plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    auroc_test, prc_test, f1_test, loss_test, details = test(model_max, test_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=True)
    acc_test = accuracy_score(details['y_label'], [1 if p >= 0.5 else 0 for p in details['y_pred']])
    details.update(auroc_test=auroc_test, prc_test=prc_test, f1_test=f1_test, loss_test=loss_test.item(), acc_test=acc_test)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test),
          'acc_test: {:.4f}'.format(acc_test))
    #torch.save(model_max.state_dict(), "best_model.pth")
    #with open(args.out_file, 'a') as f:
    #    f.write('{0}\t{1}\t{2}\t{7}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\n'.format(
    #        args.in_file[5:8], args.seed, args.aggregator, loss_test.item(), auroc_test, prc_test, f1_test, args.feature_type))
    if return_details:
        return auroc_test, details
    return auroc_test

def test(model, loader, data_o, args, use_curvature=True, weight_fct='exp', final_train=False):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_no_reduction = torch.nn.BCELoss(reduction='none')
    model.eval()
    y_pred = []
    y_label = []
    y_loss = []
    y_curv = []

    with torch.no_grad():
        for i, (label, inp, curv_batch) in enumerate(loader):
            if args.cuda:
                label = label.cuda()

            output = model(data_o,  inp, use_curvature=use_curvature, weight_fct=weight_fct)
            log = torch.squeeze(m(output))

            loss = loss_fct(log, label.float())
            loss_simple = loss_no_reduction(log, label.float())
            # collected on the final test pass so callers can inspect
            # per-edge curvature/prediction (see return_details in train_model)
            if final_train:
                y_loss = y_loss + loss_simple.flatten().tolist()
                y_curv = y_curv + curv_batch.flatten().tolist()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + log.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        details = {'y_curv': y_curv, 'y_label': y_label, 'y_pred': y_pred}

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss, details