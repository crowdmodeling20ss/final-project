import numpy as np
import torch
import torch.utils.data as data


class PointDataset(data.Dataset):

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.bmp', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.bmp')).convert('RGB')
        center_crop = transforms.CenterCrop(240)
        img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_GT.bmp'))
        target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = target[..., 0]
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels
