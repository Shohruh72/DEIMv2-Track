import copy
import csv
import os
import time
import warnings
from argparse import ArgumentParser

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/COCO'  # or `path` of --data


def train(args, params):
    # Model
    model = getattr(nn, f'deim_v2_{args.model}')(len(params['names']))
    if args.weights:
        util.load_weight(model, args.weights)  # fine-tune from a COCO checkpoint
    else:
        model.backbone.load(params['backbone'])  # pretrained backbone only
    model.cuda()

    # Optimizer
    accumulate = max(round(params['batch_size'] / (args.batch_size * args.world_size)), 1)
    optimizer = torch.optim.AdamW(util.set_params(model, params), params['lr'], betas=(0.9, 0.999))

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    split = params.get('train', 'train2017')
    with open(f'{data_dir}/{split}.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/{split}/' + filename)

    sampler = None
    dataset = Dataset(filenames, params['input_size'], params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler, num_workers=8,
                             pin_memory=True, drop_last=True, collate_fn=dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.FlatCosineLR(params, num_steps, accumulate)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    best = 0
    start_epoch = 0
    restart = args.resume  # resume from the last stage-1 state
    if args.resume:
        state = torch.load('./weights/pt/stage.pt', map_location='cpu', weights_only=False)
        start_epoch, best = state['epoch'], state['best']
    previous = best
    stop_epoch = params['epochs'] - params['no_aug_epoch']
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open('./weights/pt/step.csv', 'a' if args.resume else 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'cls', 'box', 'fgl', 'ddf',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            if not args.resume:
                logger.writeheader()

        for epoch in range(start_epoch, params['epochs']):
            # Last stage: restart from the best stage-1 state, again whenever an epoch does not improve
            if (epoch == stop_epoch or restart) and os.path.exists('./weights/pt/stage.pt'):
                if args.distributed:
                    torch.distributed.barrier()
                state = torch.load('./weights/pt/stage.pt', map_location='cpu', weights_only=False)
                (model.module if args.distributed else model).load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                if ema:
                    ema.ema.load_state_dict(state['ema'])
                    ema.updates = state['updates']

            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            loader.dataset.epoch = epoch

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 6) % ('epoch', 'memory', 'cls', 'box', 'fgl', 'ddf'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_loss = [util.AverageMeter() for _ in range(4)]
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda()
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples, targets)  # targets feed the denoising queries
                with torch.amp.autocast('cuda', enabled=False):
                    loss = criterion(outputs, targets, epoch)

                for meter, x in zip(avg_loss, loss):
                    meter.update(x.item(), samples.size(0))

                # Backward
                amp_scale.scale(sum(loss) / accumulate).backward()

                # Optimize
                if (i + 1) % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 4) % (f'{epoch + 1}/{params["epochs"]}', memory,
                                                       *(x.avg for x in avg_loss))
                    p_bar.set_description(s)

            restart = [False]
            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'cls': str(f'{avg_loss[0].avg:.3f}'),
                                 'box': str(f'{avg_loss[1].avg:.3f}'),
                                 'fgl': str(f'{avg_loss[2].avg:.3f}'),
                                 'ddf': str(f'{avg_loss[3].avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Stage-1 state to restart from (model, EMA, optimizer)
                if epoch < stop_epoch and last[0] > best:
                    torch.save({'model': (model.module if args.distributed else model).state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'ema': ema.ema.state_dict(),
                                'updates': ema.updates,
                                'epoch': epoch + 1,
                                'best': last[0]}, f='./weights/pt/stage.pt')
                if epoch >= stop_epoch:
                    restart = [last[0] <= previous]
                    previous = 0 if restart[0] else last[0]
                else:
                    previous = max(best, last[0])

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/pt/last.pt')
                if best == last[0]:
                    torch.save(save, f='./weights/pt/best.pt')
                del save

            if args.distributed:
                torch.distributed.broadcast_object_list(restart, src=0)
            restart = restart[0]

    if args.local_rank == 0:
        util.strip_optimizer('./weights/pt/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/pt/last.pt')  # strip optimizers


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    split = params.get('val', 'val2017')
    with open(f'{data_dir}/{split}.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/{split}/' + filename)

    dataset = Dataset(filenames, params['input_size'], params, augment=False)
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=dataset.collate_fn)

    if not model:
        model = torch.load(f='./weights/pt/best.pt', map_location='cuda', weights_only=False)
        model = model['model'].float().fuse()

    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        with torch.amp.autocast('cuda'):
            outputs = model(samples)
        # Top-k (no NMS)
        outputs = util.post_process(outputs.float(), params['max_det'])
        # Metrics
        for output, target in zip(outputs, targets):
            cls = target['cls'].cuda()
            box = target['box'].cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls[:, None].float(), util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    return mean_ap, map50, m_rec, m_pre


def load_model(args, params):
    if args.weights:
        # *.pt of this repo, official *.pth or Hugging Face *.safetensors
        model = getattr(nn, f'deim_v2_{args.model}')(len(params['names']))
        return util.load_weight(model, args.weights)
    return torch.load(f='./weights/pt/best.pt', map_location='cpu', weights_only=False)['model'].float()


def export(args, params):
    if args.export == 'engine' and args.weights.endswith('.onnx'):
        # TensorRT engine straight from a released ONNX (use *_fp16.onnx for an FP16 engine)
        name = os.path.splitext(os.path.basename(args.weights))[0]
        util.export_engine(args.weights, args.batch_size, output=f'./weights/engine/{name}.engine')
        return
    model = load_model(args, params)
    # weights/onnx/{name}.onnx, weights/onnx/{name}_fp16.onnx -> weights/engine/{name}_fp16.engine
    name = os.path.splitext(os.path.basename(args.weights or 'best.pt'))[0]
    if args.export == 'onnx':
        util.export_onnx(model, params, f=f'./weights/onnx/{name}.onnx')
    else:
        util.export_engine(util.export_onnx(model, params, f=f'./weights/onnx/{name}_fp16.onnx', half=True),
                           args.batch_size, output=f'./weights/engine/{name}_fp16.engine')


def draw(frame, box, label, color):
    import cv2

    x1, y1, x2, y2 = (int(i) for i in box)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y1, th + 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y - th - 4), (x1 + tw + 2, y), color, -1)
    cv2.putText(frame, label, (x1 + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def color(i):
    return tuple(int(c) for c in numpy.random.default_rng(i).integers(64, 256, 3))


@torch.no_grad()
def demo(args, params):
    import cv2
    from collections import defaultdict, deque
    from utils.tracker import Tracker

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.weights.endswith('.engine'):
        model = util.TensorRT(args.weights)
    elif args.weights.endswith('.onnx'):
        model = util.ONNX(args.weights)
    else:
        model = util.End2End(load_model(args, params).fuse(), params['max_det']).to(device).eval()

    size = params['input_size']
    names = params['names']

    stream = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not stream.isOpened():
        raise RuntimeError(f'Cannot open {args.source}')
    rate = stream.get(cv2.CAP_PROP_FPS) or 30  # cameras may report 0

    tracker = Tracker(params, rate) if args.track else None
    trails = defaultdict(lambda: deque(maxlen=30))
    counts = defaultdict(set)  # class -> track IDs seen

    writer = log = None
    if args.output:
        shape = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), rate, shape)
        if tracker is not None:
            # MOTChallenge format: frame, id, x, y, w, h, score, class, -1, -1
            log = open(os.path.splitext(args.output)[0] + '.txt', 'w')

    fps, index = 0., 0
    while True:
        success, frame = stream.read()
        if not success:
            break
        index += 1
        t = time.perf_counter()
        h, w = frame.shape[:2]
        # same as the dataset: stretch to (size, size), BGR -> RGB, [0, 1], normalization is inside the model
        x = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)[..., ::-1]
        x = torch.from_numpy(x.copy()).to(device).permute(2, 0, 1)[None].float() / 255.
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            output = model(x)[0]
        output[:, :4] *= torch.tensor([w / size, h / size] * 2, device=device)
        output = output.cpu().numpy()
        if args.classes:
            output = output[numpy.isin(output[:, 5], args.classes)]
        if tracker is not None:
            output = tracker.update(output[:, :4], output[:, 4], output[:, 5], frame)
        else:
            output = output[output[:, 4] > args.conf]
        fps = 0.9 * fps + 0.1 / (time.perf_counter() - t)

        if tracker is not None:
            for x1, y1, x2, y2, i, score, cls in output:
                i, cls = int(i), int(cls)
                counts[cls].add(i)
                trails[i].append((int((x1 + x2) / 2), int(y2)))
                cv2.polylines(frame, [numpy.array(trails[i], dtype=numpy.int32)], False, color(i), 2)
                draw(frame, (x1, y1, x2, y2), f'{i} {names[cls]} {score:.2f}', color(i))
                if log is not None:
                    log.write(f'{index},{i},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{score:.3f},{cls},-1,-1\n')
            for i in set(trails) - {t.id for t in tracker.tracks}:
                del trails[i]
            text = '  '.join(f'{names[c]}: {len(ids)}' for c, ids in sorted(counts.items()))
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            for x1, y1, x2, y2, score, cls in output:
                draw(frame, (x1, y1, x2, y2), f'{names[int(cls)]} {score:.2f}', color(int(cls)))
        cv2.putText(frame, f'{fps:.1f} FPS', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if not args.hide:
            cv2.imshow('DEIMv2', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or Esc
                break

    stream.release()
    if writer is not None:
        writer.release()
    if log is not None:
        log.close()
    cv2.destroyAllWindows()


def profile(args, params):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    shape = (1, 3, params['input_size'], params['input_size'])
    model = getattr(nn, f'deim_v2_{args.model}')(len(params['names'])).fuse()

    model.eval()
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(torch.zeros(shape))  # math attention backend, so attention FLOPs are counted

    if args.local_rank == 0:
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / 1E6:.3f}M')
        print(f'Number of FLOPs: {counter.get_total_flops() / 1E9:.3f}G')


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='s', choices=['atto', 'femto', 'pico', 'n', 's', 'm', 'l', 'x'])
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--data', default='', type=str)  # dataset YAML, default: COCO in `data_dir`
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')  # continue --train from weights/pt/stage.pt
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--export', nargs='?', const='onnx', choices=['onnx', 'engine'])
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--source', default='0', type=str)  # camera index, video file or stream URL
    parser.add_argument('--conf', default=0.5, type=float)  # score threshold for --demo
    parser.add_argument('--track', action='store_true')  # multi-object tracking for --demo
    parser.add_argument('--output', default='', type=str)  # save --demo video (and MOT results with --track)
    parser.add_argument('--hide', action='store_true')  # no window, e.g. on a server
    parser.add_argument('--classes', nargs='+', type=int)  # keep only these class indices, e.g. 0 2

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        for folder in ('pt', 'safetensors', 'onnx', 'engine'):
            os.makedirs(f'./weights/{folder}', exist_ok=True)

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    params.update(params[args.model])
    if args.data:
        # dataset: `path`, `train` / `val` splits, `names`, and overrides of the recipe (e.g. epochs)
        with open(args.data, errors='ignore') as f:
            params.update(yaml.safe_load(f))
        global data_dir
        data_dir = params.get('path', data_dir)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.export:
        export(args, params)
    if args.demo:
        demo(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
