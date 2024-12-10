from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Compressed_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

from data_provider.fli.fli_data_loader import FLI_Dataset_ETT_minute

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ETTm1_compression': Dataset_ETT_minute #Dataset_Compressed_ETT_minute
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # # Separate compression dataset from other classes to pass additional parameters
    # if args.enable_compression:
    #     data_set = Dataset_Compressed_ETT_minute(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         features=args.features,
    #         target=args.target,
    #         timeenc=timeenc,
    #         freq=freq,
    #         train_only=train_only,
    #         preserve_ratio=args.preserve_ratio
    #     )
    # else:
    #     data_set = Data(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         features=args.features,
    #         target=args.target,
    #         timeenc=timeenc,
    #         freq=freq,
    #         train_only=train_only
    #     )
    #
    # print(flag, len(data_set))
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    # return data_set, data_loader

    # FLI compression experiments
    use_fli = False

    if args.enable_compression and args.tolerated_error != 0:
        use_fli = True
        data_set = FLI_Dataset_ETT_minute(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_only=train_only,
            tolerated_error=args.tolerated_error
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_only=train_only
        )

    print(flag, len(data_set))

    if flag != "test":
        if use_fli:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            # compression_sampler = SubsetRandomSampler(range(0, int(len(data_set)/args.preserve_ratio)))
            # compression_sampler = RandomSampler(data_set, True, int(len(data_set)/args.preserve_ratio))
            compression_sampler = RandomSampler(data_set, True, args.preserve_ratio)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                sampler=compression_sampler,
                num_workers=args.num_workers,
                drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=drop_last)

    return data_set, data_loader
