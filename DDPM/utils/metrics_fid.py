import numpy as np
import torch
import torchvision.transforms as TF
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from time import time

# Define image pre-processing
preprocess = TF.Compose(
    [
        TF.Resize(299),  # Inception expects 299x299
        TF.CenterCrop(299),
        # TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_features(images, model):
    """Extract Inception-v3 features for a batch of images."""
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(real_images, fake_images, inception_model):
    """Compute FID between real and generated images."""
    # Preprocess and extract features
    real_features = get_features(real_images, inception_model)
    fake_features = get_features(fake_images, inception_model)

    # Calculate mean and covariance
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(
        fake_features, rowvar=False
    )

    # Compute FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def fid_score(test_set, model, diffuser, device):
    # Load pre-trained Inception-v3 (in evaluation mode)
    inception = inception_v3(pretrained=True, transform_input=False).eval()
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception = inception.to(device)

    real_images = torch.stack([preprocess(img) for img, _ in test_set])

    batch_size = 100
    fid_scores = []
    for i in range(0, len(real_images), batch_size):
        start = time()
        size = (batch_size, 3, 32, 32)
        xhat = diffuser.sample(model, size, device)
        batch_real = real_images[i : i + batch_size].to(device)
        batch_fake = torch.stack([preprocess(img) for img in xhat]).to(device)
        fid_scores.append(calculate_fid(batch_real, batch_fake, inception))
        print(f"FID batch: {i//batch_size} time: {time()-start}")
        # del batch_fake, batch_real
        # torch.cuda.empty_cache()
    fid_score = np.mean(fid_scores)
    return fid_score
    # print(f"FID: {fid_score:.2f}")
