import os
import yaml

def get_precision(filename: str):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    ret = 0
    for _, v in data.items():
        ret += v['nbits_key'] + v['nbits_value']
    ret /= len(data) * 2
    return ret

calibration_presets = os.listdir('./calibration_presets')

for preset in calibration_presets:
    print(f'Precision for {preset}: {get_precision(os.path.join("./calibration_presets", preset))}')