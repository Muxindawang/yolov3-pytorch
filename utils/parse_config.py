# 模型配置
def parse_model_config(path):
    """
    解析cfg文件,并将每一个块存储为字典,块的属性以及值在字典中作为键值对存储
    Parses the yolo-v3 layer configuration file and returns module definitions
    [convolutional]
    batch_normalize=1
    filters=32
    [shortcut]
    from=-3
    activation=linear
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]    # 去掉空格
    module_defs = []
    for line in lines:
        if line.startswith('['):       # This marks the start of a new block标志着一个新的block
            module_defs.append({})      # 添加一个空字典
            module_defs[-1]['type'] = line[1:-1].rstrip()    # 新增加一个键值对  键为'type'  值为第一个以[起始的行 即 [net]
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
                # [convolutional]
                # batch_normalize=1
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


# 数据配置
def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

