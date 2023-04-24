from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



if __name__ == "__main__":

    fig, ax = plt.subplots(3, 1, figsize=(3, 9))

    img = io.imread("t1_256x256.jpg")
    img = img / np.max(img)

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Image with Tagging")
    
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    ft_abs = np.abs(ft_shift)
    
    ax[1].imshow(20*np.log10(ft_abs), cmap="gray", vmin=0, vmax=80)
    ax[1].set_title("Fourier Transform (FT) of\nTagged Image")

    # draw a rectangle around the tag
    # ft_shift_mask[:, 0:160-16] = 0
    # ft_shift_mask[:, 160+15:] = 0
    # ft_shift_mask[0:128-16, :] = 0
    # ft_shift_mask[128+15:, :] = 0

    rect = Rectangle((160-16, 128-16), 32, 32, linewidth=1, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)

    ft_shift_mask = ft_shift.copy()

    # # mask out everything excelp the square around [x=167, y=128]
    ft_shift_mask[:, 0:160-16] = 0
    ft_shift_mask[:, 160+15:] = 0
    ft_shift_mask[0:128-16, :] = 0
    ft_shift_mask[128+15:, :] = 0

    ft_shift_mask_abs = np.abs(ft_shift_mask)
    mag_for_plot = 20*np.log10(ft_shift_mask_abs)
    mag_for_plot[mag_for_plot == np.inf] = 0
    mag_for_plot[mag_for_plot == -np.inf] = 0
    mag_for_plot = np.nan_to_num(mag_for_plot)

    ax[2].imshow(mag_for_plot, cmap="gray", vmin=0, vmax=80)
    ax[2].set_title("Masked FT of\nTagged Image")

    rect = Rectangle((160-16, 128-16), 32, 32, linewidth=1, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)

    fig.tight_layout()
    plt.savefig("harp.png", dpi=300)

    # # reconstruct the image
    ft_shift_mask = np.fft.ifftshift(ft_shift_mask)
    ft_mask = np.fft.ifft2(ft_shift_mask)

    harmonic_magnitude_img = np.abs(ft_mask)
    harmonic_magnitude_img = harmonic_magnitude_img

    harmonic_phase_img = np.angle(ft_mask)
    harmonic_phase_img = harmonic_phase_img

    harmonic_magnitude_img_thresholded = harmonic_magnitude_img.copy()
    harmonic_magnitude_img_thresholded[harmonic_magnitude_img_thresholded < 0.1] = 0
    harmonic_magnitude_img_thresholded[harmonic_magnitude_img_thresholded >= 0.1] = 1

    harmonic_phase_img_masked = np.ma.masked_where(harmonic_magnitude_img_thresholded == 0, harmonic_phase_img)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(harmonic_magnitude_img, cmap="gray")
    ax[0].set_title("Harmonic Magnitude\nImage")
    ax[1].imshow(harmonic_magnitude_img_thresholded, cmap="gray")
    ax[1].set_title("Harmonic Magnitude\nImage Thresholded")
    ax[2].imshow(harmonic_phase_img)
    ax[2].set_title("Harmonic Phase\nImage")
    ax[3].imshow(harmonic_phase_img_masked)
    ax[3].set_title("Harmonic Phase Image Masked\nUsing Thresholded\nMagnitude Image")
    plt.tight_layout()
    plt.savefig("harp_2.png", dpi=300)