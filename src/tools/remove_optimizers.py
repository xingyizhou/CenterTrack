import os
import torch
IN_PATH = '../../centertrack_models/'
OUT_PATH = '../../models/'
REMOVE_KEYS = ['base.fc']

if __name__ == '__main__':
  models = sorted(os.listdir(IN_PATH))
  for model in models:
    model_path = IN_PATH + model
    print(model)
    data = torch.load(model_path)
    state_dict = data['state_dict']
    keys = state_dict.keys()
    delete_keys = []
    for k in keys:
      should_delete = False
      for remove_key in REMOVE_KEYS:
        if remove_key in k:
          should_delete = True
      if should_delete:
        delete_keys.append(k)
    for k in delete_keys:
      print('delete ', k)
      del state_dict[k]
    out_data = {'epoch': data['epoch'], 'state_dict': state_dict}
    torch.save(out_data, OUT_PATH + model)
