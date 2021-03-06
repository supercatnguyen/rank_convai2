# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.scripts.extract_image_feature import extract_feats
from .build import build
try:
    import torch
except Exception as e:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset
from parlai.core.dict import DictionaryAgent

import os
import json

# There is no real dialog in this task, so for the purposes of display_data, we
# include a generic question that applies to all images.
QUESTION = "Describe the above picture in a sentence."


def _path(opt):
    build(opt)

    data_path = os.path.join(opt['datapath'], 'Flickr30k',
                             'dataset.json')
    image_path = os.path.join(opt['datapath'], 'Flickr30k', 'flickr30k_images')

    return data_path, image_path


class FlickrDataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt, shared=None):
        self.opt = opt
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.num_epochs = self.opt.get('num_epochs', 0)
        self.image_loader = ImageLoader(opt)
        data_path, self.image_path = _path(opt)
        self._setup_data(data_path, opt.get('unittest', False))
        self.dict_agent = DictionaryAgent(opt)

    def __getitem__(self, index):
        index %= self.num_episodes()
        cap = self.data[index]
        image_id = int(cap['filename'].replace('.jpg', ''))
        ep = {
            'text': QUESTION,
            'image': self.get_image(image_id),
            'episode_done': True,
        }
        if self.opt.get('extract_image', False):
            ep['image_id'] = image_id
            return ep

        ep['labels'] = [s['raw'] for s in cap['sentences']]
        ep['valid'] = True
        if 'train' not in self.datatype:
            ep['label_candidates'] = self.cands
        return (index, ep)

    def __len__(self):
        return self.num_episodes()

    def _setup_data(self, data_path, unittest):
        with open(data_path) as data_file:
            raw_data = json.load(data_file)['images']
            if 'train' in self.datatype:
                self.data = [d for d in raw_data if d['split'] == 'train']
            elif 'valid' in self.datatype:
                self.data = [d for d in raw_data if d['split'] == 'val']
                self.cands = [l for d in self.data for l in [s['raw'] for s in d['sentences']]]
            else:
                self.data = [d for d in raw_data if d['split'] == 'test']
                self.cands = [l for d in self.data for l in [s['raw'] for s in d['sentences']]]
        if unittest:
            self.caption = self.caption[:10]

    def get_image(self, image_id):
        im_path = os.path.join(self.image_path, '%d.jpg' % (image_id))
        return self.image_loader.load(im_path)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_episodes()

    def num_images(self):
        return self.num_episodes()


class DefaultDataset(FlickrDataset):
    pass


class DefaultTeacher(FixedDialogTeacher):
    """
    Flickr default teacher that expects open-ended descriptions of images
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.image_mode = opt.get('image_mode', 'none')
        data_path, self.image_path = _path(opt)

        if shared:
            # another instance was set up already, just reference its data
            self.data = shared['data']
            self.image_loader = shared['image_loader']
            if 'cands' in shared:
                self.cands = shared['cands']
        else:
            # need to set up data from scratch
            self._setup_data(data_path)
            self.image_loader = ImageLoader(opt)

        self.reset()

    def reset(self):
        super().reset()  # call parent reset so other fields can be set up
        self.example = None  # set up caching fields
        self.imageEpochDone = False

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = os.path.join(self.image_path, '%d.jpg' % (image_id))
        self.data_loader.request_load(self.receive_data,
                                      self.image_loader.load,
                                      (img_path,))

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        action = {
            'text': "",
            'image_id': int(ep['filename'].replace('.jpg', '')),
            'episode_done': True,
            'labels': [s['raw'] for s in ep['sentences']]
        }
        if 'train' not in self.datatype:
            action['label_candidates'] = self.cands
        return action

    def next_example(self):
        """Returns the next example from this dataset after starting to queue
        up the next example.
        """
        ready = None
        # pull up the currently queued example
        if self.example is not None:
            if self.image_mode != 'none' and 'image_id' in self.example:
                # move the image we loaded in the background into the example
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        # get the next base example: super().next_example() calls self.get()
        self.example, self.imageEpochDone = super().next_example()
        if self.image_mode != 'none' and 'image_id' in self.example:
            # load the next image in the background
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        # Try to return the previously cached example
        if ready is None:
            return self.next_example()
        else:
            return ready

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        if hasattr(self, 'cands'):
            shared['cands'] = self.cands
        return shared

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            raw_data = json.load(data_file)['images']
            if 'train' in self.datatype:
                self.data = [d for d in raw_data if d['split'] == 'train']
            elif 'valid' in self.datatype:
                self.data = [d for d in raw_data if d['split'] == 'val']
                self.cands = [l for d in self.data for l in [s['raw'] for s in d['sentences']]]
            else:
                self.data = [d for d in raw_data if d['split'] == 'test']
                self.cands = [l for d in self.data for l in [s['raw'] for s in d['sentences']]]
