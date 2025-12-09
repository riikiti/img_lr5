import cv2
import numpy as np
import matplotlib.pyplot as plt


def darken(img, k=0.5):
    """Предварительное затемнение"""
    return np.clip(img * k, 0, 255).astype(np.uint8)


def manual_contrast(img):
    """Ручное приведение гистограммы"""
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)


def plot_hist(ax, img, title):
    ax.set_title(title)
    ax.hist(img.ravel(), bins=256, range=(0, 255), color='black')


def process_image(path):
    # Загружаем + делаем серым
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Предварительная обработка (затемнение)
    dark = darken(gray)

    # Эквализация гистограммы
    eq = cv2.equalizeHist(dark)

    # Ручное приведение гистограммы
    manual = manual_contrast(dark)

    # Вывод изображений
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    axs[0][0].imshow(dark, cmap='gray')
    axs[0][0].set_title("Исходное (затемнённое)")
    axs[0][0].axis("off")

    axs[0][1].imshow(eq, cmap='gray')
    axs[0][1].set_title("После эквализации")
    axs[0][1].axis("off")

    axs[0][2].imshow(manual, cmap='gray')
    axs[0][2].set_title("После приведения")
    axs[0][2].axis("off")

    plot_hist(axs[1][0], dark, "Гистограмма исходного")
    plot_hist(axs[1][1], eq, "Гистограмма эквализации")
    plot_hist(axs[1][2], manual, "Гистограмма приведения")

    plt.tight_layout()
    plt.show()


# Пример вызова (замените на свои фото)
process_image("clouds.jpg")
