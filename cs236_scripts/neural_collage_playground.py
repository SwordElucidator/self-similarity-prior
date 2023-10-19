import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch_src.collage import NeuralCollageOperator2d
import tensorflow as tf
import os


experiment_setups = {
    'images': {"einstein375.png": 'https://firebasestorage.googleapis.com/v0/b/sde-bnn.appspot.com/o/'
                                  'neural-collage%2Feinstein375.png?'
                                  'alt=media&token=e2160ac0-c88a-4f62-9ccd-86de5476d82e'}
}


class NeuralCollageExperiment:
    def __init__(self):
        self.image_bank = self.__load_images()

    @staticmethod
    def __load_images():
        image_bank = {}
        for image_name, image_url in experiment_setups['images'].items():
            image_path = tf.keras.utils.get_file(image_name, image_url, extract=True)
            image_bank[image_name] = os.path.join(os.path.dirname(image_path), image_name)
        return image_bank

    def train(
            self, image_name, *,
            out_res=375,
            out_channels=3,
            range_patch_height=5,
            range_patch_width=5,
            domain_patch_height=375,
            domain_patch_width=375,
            use_augmentations=False,
            n_iterations=200,
            n_decode_steps=10,

    ):
        im = Image.open(self.image_bank[image_name])
        im = np.asarray(im)
        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)[:, :3]
        # Train the collage operator encoder
        collage_operator = NeuralCollageOperator2d(
            out_res=out_res,
            out_channels=out_channels,
            rh=range_patch_height,
            rw=range_patch_width,
            dh=domain_patch_height,
            dw=domain_patch_width,
            use_augmentations=use_augmentations
        )
        opt = torch.optim.Adam(collage_operator.parameters(), lr=1e-2)
        objective = nn.MSELoss()
        norm_im = im.float() / 255

        for k in range(n_iterations):
            recon = collage_operator(norm_im, decode_steps=n_decode_steps, return_co_code=False)
            loss = objective(recon, norm_im)
            print(f'Reconstruction MSE: {loss}', end='\r')
            loss.backward()
            opt.step()
            opt.zero_grad()

        return collage_operator, norm_im

    def calc_size_ratio(self, collage_operator, norm_im):
        _, fractal_weight, fractal_bias = collage_operator(norm_im, decode_steps=10, return_co_code=True)
        size_fractal_code = (torch.numel(fractal_weight) + torch.numel(fractal_bias))
        raw_img_size = torch.numel(norm_im)
        print(f'Ratio: size fractal code / size of raw image is {size_fractal_code / raw_img_size}')

    def plot(self, collage_operator, norm_im):
        recon = collage_operator(norm_im, decode_steps=10)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(norm_im[0].permute(1, 2, 0), cmap='bone')
        axs[1].imshow(recon[0].permute(1, 2, 0).detach(), cmap='bone')
        plt.show()


if __name__ == '__main__':
    experiment = NeuralCollageExperiment()
    collage_operator, norm_im = experiment.train('einstein375.png')
    experiment.calc_size_ratio(collage_operator, norm_im)
    experiment.plot(collage_operator, norm_im)

    collage_operator, norm_im = experiment.train('einstein375.png', n_decode_steps=20)
    experiment.calc_size_ratio(collage_operator, norm_im)
    experiment.plot(collage_operator, norm_im)
