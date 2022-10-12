from tqdm import trange
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModelWithRefHe, DiscriminatorFullModelWithRefHe, GeneratorFullModelWithTF, DiscriminatorFullModelWithTF, ExpTransformerTrainer, GeneratorFullModelWithSeg, DiscriminatorFullModelWithSeg

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train_transformer(config, stage, exp_transformer, generator, discriminator, kp_detector, he_estimator, checkpoint, checkpoint_ref, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer = torch.optim.Adam(exp_transformer.parameters(), lr=train_params['lr_exp_transformer'], betas=(0.5, 0.999))

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    # optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint_ref is not None:
        Logger.load_cpk(checkpoint_ref, generator=generator, he_estimator=he_estimator)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator=generator, discriminator=discriminator, he_estimator=he_estimator, exp_transformer=exp_transformer, optimizer_generator=optimizer_generator, optimizer_exp_transformer=optimizer, optimizer_discriminator=optimizer_discriminator, optimizer_he_estimator=optimizer_he_estimator)
    else:
        start_epoch = 0

    # print(f'start epoch: {start_epoch}')
    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    trainer = ExpTransformerTrainer(stage, exp_transformer, kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'], device_ids=device_ids)
    # generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    discriminator_full = DiscriminatorFullModelWithSeg(generator, discriminator, train_params)

    if torch.cuda.is_available():
        # generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        trainer = DataParallelWithCallback(trainer, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        print(f'running ground test...')
        with torch.no_grad():
            x = next(iter(dataloader))
            losses_generator, generated = trainer(x, cycled_drive=True)
            if train_params['loss_weights']['generator_gan'] != 0:
                losses_discriminator = discriminator_full(x, generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                loss = sum(loss_values)

            losses_generator.update(losses_discriminator)
            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
            logger.log_ground(losses=losses, inp=x, out=generated)
            del losses
            del generated
        print(f'start training...')
        for epoch in trange(start_epoch, train_params['num_epochs']):
            num_item_call = 0
            num_cache_hit = 0 

            print(f'len dataloader: {len(dataloader)}')

            for x in tqdm(dataloader):
                # print(f'input x: {x}')
                num_item_call += len(x['hit'])
                num_cache_hit += x['hit'].sum()


                losses_generator, generated = trainer(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                # optimizer_kp_detector.step()
                # optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()
                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler.step()
            scheduler_generator.step()
            scheduler_discriminator.step()
            # scheduler_kp_detector.step()
            scheduler_he_estimator.step()
            
            cache_log = f'cache hit ratio : {num_cache_hit / num_item_call * 100:.2f} %'
            print(cache_log)
            
            with torch.no_grad():
                x = next(iter(dataloader))
                losses_generator, generated = trainer(x, cycled_drive=True)

            logger.log_epoch(epoch, {'exp_transformer': exp_transformer, 
            'generator': generator,
            'discriminator': discriminator,
            # 'kp_detector': kp_detector,
            'he_estimator': he_estimator,
            'optimizer_exp_transformer': optimizer,
            'optimizer_generator': optimizer_generator,
            # 'optimizer_kp_detector': optimizer_kp_detector,
            'optimizer_he_estimator': optimizer_he_estimator, 
            'optimizer_discriminator': optimizer_discriminator}, inp=x, out=generated)

def train_baseline(config, generator, discriminator, kp_detector, he_estimator, checkpoint, log_dir, dataset, device_ids, checkpoint_ref=None, he_estimator_ref=None):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator=generator, discriminator=discriminator, kp_detector=kp_detector, he_estimator=he_estimator,
                                      optimizer_generator=optimizer_generator, optimizer_discriminator=optimizer_discriminator, optimizer_kp_detector=optimizer_kp_detector, optimizer_he_estimator=optimizer_he_estimator)
    else:
        start_epoch = 0

    if checkpoint_ref is not None:
        Logger.load_cpk(checkpoint_ref, generator=generator, discriminator=discriminator, he_estimator=he_estimator,
                                    optimizer_generator=optimizer_generator, optimizer_discriminator=optimizer_discriminator, optimizer_he_estimator=optimizer_he_estimator)
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModelWithRefHe(kp_detector, he_estimator, generator, discriminator, train_params, he_estimator_ref=he_estimator_ref, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    discriminator_full = DiscriminatorFullModelWithRefHe(generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        print(f'start epochs')
        for epoch in trange(start_epoch, train_params['num_epochs']):
            print(f'staring epoch {epoch}')
            for x in tqdm(dataloader):
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_he_estimator.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'he_estimator': he_estimator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_he_estimator': optimizer_he_estimator}, inp=x, out=generated)



def train_hie(config, generator, discriminator, hie_estimator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_hie_estimator = torch.optim.Adam(hie_estimator.parameters(), lr=train_params['lr_hie_estimator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator=generator, discriminator=discriminator, hie_estimator=hie_estimator,
                                      optimizer_generator=optimizer_generator, optimizer_discriminator=optimizer_discriminator, optimizer_hie_estimator=optimizer_hie_estimator)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_hie_estimator = MultiStepLR(optimizer_hie_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_hie_estimator'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModelWithTF(hie_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    discriminator_full = DiscriminatorFullModelWithTF(generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_hie_estimator.step()
                optimizer_hie_estimator.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_hie_estimator.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'hie_estimator': hie_estimator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_hie_estimator': hie_estimator}, inp=x, out=generated)

def train_geo(config, generator, discriminator, kp_detector, he_estimator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    if discriminator is not None:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    else:
        optimizer_discriminator = None
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, None, None, None, 
                                      optimizer_generator, optimizer_discriminator)
        # start_epoch = 0
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch= start_epoch - 1)
    if discriminator is not None:
        scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                            last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    # # load reference info
    # ref_path = '/home/server25/minyeong_workspace/BFv2v/frame_reference.png'
    # reference_dataset = SingleImageDataset(ref_path)
    # ref_img_data = iter(DataLoader(reference_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)).next()
    # with torch.no_grad():
    #     he_ref = he_estimator(ref_img_data['frame'].cuda())
    # reference_info = {'R': ref_img_data['mesh']['R'][0].cuda(), 't': ref_img_data['mesh']['t'][0].cuda(), 'c': ref_img_data['mesh']['c'][0].cuda(), 'he_R': he_ref['R'][0].cuda(), 'he_t': he_ref['t'][0].cuda(), 'img': ref_img_data['frame'][0].permute(1, 2, 0).cuda()}
    
    generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    
    if discriminator is not None:
        discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        if discriminator is not None:
            discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)


    regularizor_weight_0 = train_params['loss_weights']['regularizor']
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            num_item_call = 0
            num_cache_hit = 0 
            if epoch < 1:
                train_params['loss_weights']['regularizor'] = 0
            else:
                train_params['loss_weights']['regularizor'] = regularizor_weight_0
            # print(f'len dataloader: {len(dataloader)}')
            for x in tqdm(dataloader):
                # print(f'input x: {x}')
                num_item_call += len(x['hit'])
                num_cache_hit += x['hit'].sum()

                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                if discriminator is not None:
                    optimizer_discriminator.zero_grad()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                print(losses)
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            if discriminator is not None:
                scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_he_estimator.step()
            # print(f'generated keys: {generated.keys()}')
            # cache_log = f'cache hit ratio : {num_cache_hit / num_item_call * 100:.2f} %'
            print(cache_log)

            if discriminator is not None:
                logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,}, inp=x, out=generated)
                logger.log(cache_log)

            else:
                logger.log_epoch(epoch, {'generator': generator,
                            'optimizer_generator': optimizer_generator,}, inp=x, out=generated)
                logger.log(cache_log)
            

