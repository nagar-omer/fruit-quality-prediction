from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from utils import dct_2d, mask_circle
import cv2
import numpy as np
from tqdm.auto import tqdm
import xgboost as xgb


def fruit_detector(img, safety_margin=0, equalize=True, margin=112):
    # convert to grayscale
    img = img.copy()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # equalize & smoothing & edge detection
    if equalize:
        img = cv2.equalizeHist(img.astype(np.uint8))
    img = cv2.GaussianBlur(img, (11, 11), sigmaX=5)
    img = dog(img, sigma1=4, sigma2=5, kernel_size=11)
    # prepare for matching - normalize to [1, 2]
    img = img / img.max() + 1

    # edge detection - directions
    dy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=11))
    dx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=11))

    # look for horizontal edges (underlying grid) -
    # give negative score so selecting it will be less likely
    img[(dy > dy.mean()) & ((dy / (dx + 1e-5)) > 0.999)] = -4

    # look for fruit using circle match filter with different radius
    selected_center, selected_radius, selected_score = None, None, 0
    for r in np.linspace(70, 100, 5):
        # build circle kernel
        kernel = cv2.circle(
            img=np.zeros((int(r*2), int(r*2)), dtype=np.uint8),
            center=(int(r), int(r)),
            radius=int(r),
            color=255,
            thickness=-1).astype(np.float32)
        kernel /= kernel.sum()

        # apply kernel + find max
        fruit_heat_map = cv2.filter2D(img, -1, kernel) / kernel.sum()
        # remove out of bound values
        fruit_heat_map[:int(max(r, margin)), :] = -np.inf
        fruit_heat_map[-int(max(r, margin)):, :] = -np.inf
        fruit_heat_map[:, :int(max(r, margin))] = -np.inf
        fruit_heat_map[:, -int(max(r, margin)):] = -np.inf

        center = np.unravel_index(fruit_heat_map.argmax(), img.shape)

        if fruit_heat_map.max() > selected_score:
            selected_center, selected_radius, selected_score = center, r, fruit_heat_map.max()

    if selected_center is None:
        return None

    c, r = selected_center, selected_radius + safety_margin
    return (int(c[0]), int(c[1])), int(r)


def information_approximation(img, freq_cut=30):
    """
    Using dct asses the amount of information in the image.
    a smooth fruit will have less information
    :return: float - sum of small section in the DCT
    """

    return dct_2d(img)[:freq_cut, :freq_cut].sum()


def preprocess(dataset, dct=True):
    gt, emb, r_all, colors = [], [], [], []

    for i in tqdm(range(len(dataset))):
        sample, label = dataset[i]
        sample, label = sample.numpy(), label.numpy()
        gt += [label.astype(np.uint)]

        # create features
        sample_embs, sample_radi, sample_color = [], [], []
        for j, im in enumerate(sample):
            circle = fruit_detector(im, safety_margin=10)
            if circle is None:
                continue

            # radius
            sample_radi.append(circle[1])

            # color
            rgb_mean = im.mean(axis=(0, 1))
            rgb_std = im.std(axis=(0, 1))
            sample_color.append(np.hstack([rgb_mean, rgb_std]))

            # texture
            # im = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8))
            im = mask_circle(im, circle[0], circle[1], out_shape=(224, 224))

            if dct:
                sample_embs.append(dct_2d(im)[:224 // 8, :224 // 8].flatten())
            else:
                sample_embs.append(im.flatten())

        emb.append(np.vstack(sample_embs))
        r_all.append(np.vstack(sample_radi))
        colors.append(np.vstack(sample_color))

    # zscore emb
    emb_ = np.vstack(emb)
    mu, sigma = emb_.mean(axis=0), emb_.std(axis=0)
    emb = [(e - mu) / sigma for e in emb]
    emb_ = (emb_ - mu) / sigma

    # zscore colors
    mu, sigma = np.vstack(colors).mean(axis=0), np.vstack(colors).std(axis=0)
    colors = [(c - mu) / sigma for c in colors]

    # zscore r
    mu, sigma = np.vstack(r_all).mean(axis=0), np.vstack(r_all).std(axis=0)
    r_all = [(r - mu) / sigma for r in r_all]

    clf = PCA(n_components=16)
    clf.fit(emb_)

    emb = np.stack([np.hstack([r.mean(axis=0), clf.transform(e).max(axis=0), c.mean(axis=0)])
                    for r, e, c in zip(r_all, emb, colors)])

    return np.vstack(emb), np.vstack(gt)


def dog(img, sigma1=1, sigma2=2, kernel_size=5):
    return (cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma1) -
            cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma2))


if __name__ == '__main__':
    from dataset_loader import FruitsDataset

    # ds = FruitsDataset(path="/Users/omernagar/Documents/Projects/fruit-quality-prediction/data", transform=False)
    # X, y = preprocess(ds)

    # save X, y
    # np.save("X.npy", X)
    # np.save("y.npy", y)

    # load X, y
    X = np.load("X.npy")
    y = np.load("y.npy")

    clf = xgb.XGBClassifier(random_state=26, alpha=0, max_depth=2, n_estimators=100,  reg_lambda=1, gamma=1)
    # Specify the number of folds for cross-validation
    k = 5  # You can change this to your desired number of folds

    # Create a KFold object with 'k' folds
    kf = KFold(n_splits=k, shuffle=True, random_state=26)

    # Perform k-fold cross-validation and compute cross-validation scores
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring=lambda clf, x, y: (clf.predict(x).argmax(axis=1) == y.argmax(axis=1)).sum() / y.shape[0])
    print(f"{cv_scores.mean()} +- {cv_scores.std()}")

