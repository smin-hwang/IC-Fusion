'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

from torch.utils.tensorboard import SummaryWriter

import os


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            # if self.output_dir:
            #     checkpoint_paths = [self.output_dir / 'checkpoint.pth']
            #     # extra checkpoint before LR drop and every 100 epochs
            #     if (epoch + 1) % args.checkpoint_step == 0:
            #         checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
            #     for checkpoint_path in checkpoint_paths:
            #         dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            if self.output_dir:
                checkpoint_path = self.output_dir / 'checkpoint.pth'
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_path_epoch = self.output_dir / f'checkpoint{epoch:04}.pth'
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path_epoch)
                if epoch > 0:
                    checkpoint_path_last_epoch = self.output_dir / f'checkpoint{epoch - 1:04}.pth'
                    os.remove(checkpoint_path_last_epoch)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # Save checkpoint if best_stat or last epoch
            if best_stat['epoch'] == epoch:
                checkpoint_path = self.output_dir / f'best_stat.pth'
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        eval_log_path = self.output_dir / "eval" / 'eval_log.txt'
                        with open(eval_log_path, 'a') as f:
                            f.write(f"Epoch: {epoch}\n")
                            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                                f.write(f"{'AP/IoU/0.50-0.95/all/100'}: {coco_eval.stats[0]}\n")
                                f.write(f"{'AP/IoU/0.50/all/100'}: {coco_eval.stats[1]}\n")
                                f.write(f"{'AP/IoU/0.75/all/100'}: {coco_eval.stats[2]}\n")
                                f.write(f"{'AP/IoU/0.50-0.95/small/100'}: {coco_eval.stats[3]}\n")
                                f.write(f"{'AP/IoU/0.50-0.95/medium/100'}: {coco_eval.stats[4]}\n")
                                f.write(f"{'AP/IoU/0.50-0.95/large/100'}: {coco_eval.stats[5]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/all/1'}: {coco_eval.stats[6]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/all/10'}: {coco_eval.stats[7]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/all/100'}: {coco_eval.stats[8]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/small/100'}: {coco_eval.stats[9]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/medium/100'}: {coco_eval.stats[10]}\n")
                                f.write(f"{'AR/IoU/0.50-0.95/large/100'}: {coco_eval.stats[11]}\n")
                            f.write(f"Best_state\n")
                            for k, v in best_stat.items():
                                f.write(f"{k}: {v}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
