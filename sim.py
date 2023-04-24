import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.filters import gaussian
from matplotlib.ticker import FuncFormatter


def build_grids(n_x, n_y, dx, dy):
    x_count = np.arange(-n_x / 2, n_x / 2, 1)
    y_count = np.arange(-n_y / 2, n_y / 2, 1)

    x_loc = x_count * dx
    y_loc = y_count * dy

    X_count_grid, Y_count_grid = np.meshgrid(x_count, y_count)
    X_loc_grid, Y_loc_grid = np.meshgrid(x_loc, y_loc)

    return X_count_grid, Y_count_grid, X_loc_grid, Y_loc_grid


T1 = {
    "water": 4000,
    "gray": 900,
    "myocardium": 1048,
}
T2 = {
    "water": 2000,
    "gray": 90,
    "myocardium": 50,
}


# tip angles
TIP_ANGLE_45 = np.pi / 4
TIP_ANGLE_90 = np.pi / 2

# gyromagnetic ratio
# GMR = 42.58  # MHz / T
# GMR = 42.58 * 10 ** 6  # Hz / T
GMR = 42.58 * 10**6 * 2 * np.pi  # rad / s / T


# def sim():
#     M_z_start = np.ones((n_x, n_y))
#     M_xy_start = np.zeros((n_x, n_y))
#     M_angle_start = np.zeros((n_x, n_y))

#     # tip 45 degree in y direction
#     M_z_after_tip = np.cos(tip_angle) * M_z_start
#     M_xy_after_tip = np.sin(tip_angle) * M_z_start
#     M_angle_after_tip = M_angle_start

#     M_z_after_grad = np.zeros((n_x, n_y)) * (1 - np.exp(-x_grad_t / T1["water"]))
#     M_xy_after_grad = np.zeros((n_x, n_y)) * (np.exp(-x_grad_t / T2["water"]))

#     gradient_strength_x = x_grad * X_loc_grid
#     M_angle_after_grad = ((gamma * gradient_strength_x * x_grad_t)) % (2 * np.pi)

#     # tip 45 degree again
#     M_x = M_xy_after_tip * np.cos(M_angle_after_grad)
#     M_y = M_xy_after_tip * np.sin(M_angle_after_grad)
#     M_z = M_z_after_tip
#     # R_y by tip angle
#     M_x_after_grad = M_x * np.cos(tip_angle) + M_z * np.sin(tip_angle)
#     M_y_after_grad = M_y
#     M_z_after_grad = -M_x * np.sin(tip_angle) + M_z * np.cos(tip_angle)
#     # back to xy
#     M_xy_after_grad = np.sqrt(M_x_after_grad**2 + M_y_after_grad**2)
#     M_angle_after_grad = np.arctan2(M_y_after_grad, M_x_after_grad)

#     # cursher
#     M_xy_end = M_xy_after_grad
#     M_angle_end = M_angle_after_grad
#     M_z_end = M_z_after_grad

#     plt.figure()
#     plt.imshow(M_z_end)
#     plt.colorbar()
#     plt.show()


def plot_angle(ax, M_angle):
    angle = np.mod(M_angle, 2 * np.pi)
    imshow_obj = ax.imshow(angle, cmap="twilight", aspect='auto', vmin=0, vmax=2 * np.pi)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return imshow_obj


def plot_magnitude(ax, M_mag):
    imshow_obj = ax.imshow(M_mag, cmap="gray", vmin=0, vmax=1, aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return imshow_obj


def sim_spamm(
    M_z: np.ndarray,
    M_xy: np.ndarray,
    M_angle: np.ndarray,
    T1_image: np.ndarray,
    T2_image: np.ndarray,
    X_count_grid: np.ndarray,
    Y_count_grid: np.ndarray,
    X_loc_grid: np.ndarray,
    Y_loc_grid: np.ndarray,
    x_grad: float,
    y_grad: float,
    x_grad_t: float,
    y_grad_t: float,
    gmr: float = GMR,
    tip_angle: float = TIP_ANGLE_45,
):
    fig, axs = plt.subplots(3, 5, figsize=(16, 9))

    M_z_start = M_z.copy()
    M_xy_start = M_xy.copy()
    M_angle_start = M_angle.copy()

    plot_magnitude(axs[0, 0], M_z_start)
    axs[0, 0].set_title("M_z_start")
    plot_magnitude(axs[1, 0], M_xy_start)
    axs[1, 0].set_title("M_xy_start")
    plot_angle(axs[2, 0], M_angle_start)
    axs[2, 0].set_title("M_angle_start")

    # tip 45 degree in y direction
    M_z_after_tip = np.cos(tip_angle) * M_z_start
    M_xy_after_tip = np.sin(tip_angle) * M_z_start
    M_angle_after_tip = M_angle_start

    plot_magnitude(axs[0, 1], M_z_after_tip)
    axs[0, 1].set_title("M_z_after_tip")
    plot_magnitude(axs[1, 1], M_xy_after_tip)
    axs[1, 1].set_title("M_xy_after_tip")
    plot_angle(axs[2, 1], M_angle_after_tip)
    axs[2, 1].set_title("M_angle_after_tip")

    M_z_after_grad = M_z_after_tip - M_z_after_tip * (1 - np.exp(-x_grad_t / T1_image))
    M_xy_after_grad = M_xy_after_tip * (np.exp(-x_grad_t / T2_image))

    gradient_strength_x = x_grad * X_loc_grid
    gradient_strength_y = y_grad * Y_loc_grid
    M_angle_after_grad = (gmr * gradient_strength_x * x_grad_t) + (
        gmr * gradient_strength_y * y_grad_t
    )

    plot_magnitude(axs[0, 2], M_z_after_grad)
    axs[0, 2].set_title("M_z_after_grad")
    plot_magnitude(axs[1, 2], M_xy_after_grad)
    axs[1, 2].set_title("M_xy_after_grad")
    plot_angle(axs[2, 2], M_angle_after_grad)
    axs[2, 2].set_title("M_angle_after_grad")

    # tip 45 degree again
    M_x = M_xy_after_tip * np.cos(M_angle_after_grad)
    M_y = M_xy_after_tip * np.sin(M_angle_after_grad)
    M_z = M_z_after_tip
    # R_y by tip angle
    M_x_after_tip_2 = M_x * np.cos(tip_angle) + M_z * np.sin(tip_angle)
    M_y_after_tip_2 = M_y
    M_z_after_tip_2 = -M_x * np.sin(tip_angle) + M_z * np.cos(tip_angle)
    # back to xy
    M_xy_after_tip_2 = np.sqrt(M_x_after_tip_2**2 + M_y_after_tip_2**2)
    M_angle_after_tip_2 = np.arctan2(M_y_after_tip_2, M_x_after_tip_2)

    plot_magnitude(axs[0, 3], M_z_after_tip_2)
    axs[0, 3].set_title("M_z_after_tip_2")
    plot_magnitude(axs[1, 3], M_xy_after_tip_2)
    axs[1, 3].set_title("M_xy_after_tip_2")
    plot_angle(axs[2, 3], M_angle_after_tip_2)
    axs[2, 3].set_title("M_angle_after_tip_2")

    # cursher
    M_xy_end = np.zeros_like(M_xy_after_tip_2)
    M_angle_end = np.zeros_like(M_angle_after_tip_2)
    M_z_end = M_z_after_tip_2

    imshow_obj_0 = plot_magnitude(axs[0, 4], M_z_end)
    axs[0, 4].set_title("M_z_end")
    imshow_obj_1 = plot_magnitude(axs[1, 4], M_xy_end)
    axs[1, 4].set_title("M_xy_end")
    imshow_obj_2 = plot_angle(axs[2, 4], M_angle_end)
    axs[2, 4].set_title("M_angle_end")

    fig.colorbar(imshow_obj_0, ax=axs[0, 4])
    fig.colorbar(imshow_obj_1, ax=axs[1, 4])
    fig.colorbar(imshow_obj_2, ax=axs[2, 4], format=lambda x, pos: f"{(x/(2*np.pi))*360:.0f}°", ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

    return (M_z_end, M_xy_end, M_angle_end), (fig, axs)


if __name__ == "__main__":
    ### Grid ###

    n_x = 256  # count
    n_y = 256  # count

    dx = 1.5  # mm
    dy = 1.5  # mm

    X_count_grid, Y_count_grid, X_loc_grid, Y_loc_grid = build_grids(n_x, n_y, dx, dy)

    ### Phantom ###

    disc_0 = disk((n_x // 2, n_y // 2), 60)
    disc_1 = disk((n_x // 2, n_y // 2), 70)

    ring_img = np.zeros((n_x, n_y))
    ring_img[disc_1] = 1
    ring_img[disc_0] = 0

    T1_image = np.ones((n_x, n_y))
    T1_image[ring_img == 1] = T1["myocardium"]
    T1_image[ring_img == 0] = T1["water"]

    T2_image = np.ones((n_x, n_y))
    T2_image[ring_img == 1] = T2["myocardium"]
    T2_image[ring_img == 0] = T2["water"]

    ### SPMM Gradients ###
    x_grad = 0.00001 / 10  # T / mm
    y_grad = 0.00001 / 10  # T / mm

    x_grad_t = 0.0006  # s
    y_grad_t = 0.0006  # s

    ### Simulation ###
    M_z = np.ones((n_x, n_y))
    M_xy = np.zeros((n_x, n_y))
    M_angle = np.zeros((n_x, n_y))

    (M_z, M_xy, M_angle), (fig, axs) = sim_spamm(
        M_z,
        M_xy,
        M_angle,
        T1_image,
        T2_image,
        X_count_grid,
        Y_count_grid,
        X_loc_grid,
        Y_loc_grid,
        x_grad*4,
        0,
        x_grad_t,
        0,
        tip_angle=np.pi/2
    )

    fig.tight_layout()
    plt.savefig("spamm.png", dpi=300)
    plt.close(fig)

    # fig, ax = plt.subplots(1, 3)
    # # T2 phantom
    # M_z = M_z / np.max(M_z)
    # fake_image = M_z * (1/T2_image)
    # fake_image = fake_image / np.max(fake_image)

    # ax[0].imshow(fake_image, cmap="gray")
    # ax[0].set_title("Phantom with Tagging")
    
    # ft = np.fft.fft2(fake_image)
    # ft_shift = np.fft.fftshift(ft)

    # ft_abs = np.abs(ft_shift) / np.max(np.abs(ft_shift))
    
    # ax[1].imshow(ft_abs, cmap="gray")
    # ax[1].set_title("Fourier Transform (FT) of\nTagged Phantom")

    # ft_shift_mask = ft_shift.copy()

    # # # mask out everything excelp the square around [x=167, y=128]
    # ft_shift_mask[:, 0:167-16] = 0
    # ft_shift_mask[:, 167+15:] = 0
    # ft_shift_mask[0:128-16, :] = 0
    # ft_shift_mask[128+15:, :] = 0

    # ft_shift_mask_abs = np.abs(ft_shift_mask) / np.max(np.abs(ft_shift_mask))

    # ax[2].imshow(ft_shift_mask_abs, cmap="gray")
    # ax[2].set_title("Masked FT of\nTagged Phantom")

    # fig.tight_layout()
    # plt.savefig("harp.png", dpi=300)

    # # # reconstruct the image
    # ft_shift_mask = np.fft.ifftshift(ft_shift_mask)
    # ft_mask = np.fft.ifft2(ft_shift_mask)

    # harmonic_magnitude_img = np.abs(ft_mask)
    # harmonic_magnitude_img = harmonic_magnitude_img

    # harmonic_phase_img = np.angle(ft_mask)
    # harmonic_phase_img = harmonic_phase_img

    # harmonic_magnitude_img_thresholded = harmonic_magnitude_img.copy()
    # harmonic_magnitude_img_thresholded[harmonic_magnitude_img_thresholded < 0.2] = 0
    # harmonic_magnitude_img_thresholded[harmonic_magnitude_img_thresholded >= 0.2] = 1

    # harmonic_phase_img_masked = np.ma.masked_where(harmonic_magnitude_img_thresholded == 0, harmonic_phase_img)

    # fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    # ax[0].imshow(harmonic_magnitude_img, cmap="gray")
    # ax[0].set_title("Harmonic Magnitude\nImage")
    # ax[1].imshow(harmonic_magnitude_img_thresholded, cmap="gray")
    # ax[1].set_title("Harmonic Magnitude\nImage Thresholded")
    # ax[2].imshow(harmonic_phase_img)
    # ax[2].set_title("Harmonic Phase\nImage")
    # ax[3].imshow(harmonic_phase_img_masked)
    # ax[3].set_title("Harmonic Phase Image Masked\nUsing Thresholded\nMagnitude Image")
    # plt.tight_layout()
    # plt.savefig("harp_2.png", dpi=300)


    # print(ft_mask)

    # plt.imshow(ft_mask, cmap="gray")
    # plt.show()



    


    # (M_z, M_xy, M_angle), (fig, axs) = sim_spamm(
    #     M_z,
    #     M_xy,
    #     M_angle,
    #     T1_image,
    #     T2_image,
    #     X_count_grid,
    #     Y_count_grid,
    #     X_loc_grid,
    #     Y_loc_grid,
    #     0,
    #     y_grad,
    #     0,
    #     y_grad_t,
    # )

    # fig, ax = plt.subplots(1, 1)
    # plot_magnitude(ax, M_z)
    # ax.set_title("M_z")
    # fig.tight_layout()
    # plt.savefig("spamm_2.png", dpi=300)

