import numpy as np
import pandas as pd
import math
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def shannon_entropy(text):
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in freq.values()), 4)


def generate_legitimate(n):
    data = []
    legit_domains = ["google", "amazon", "facebook", "microsoft", "github",
                     "stackoverflow", "wikipedia", "linkedin", "youtube", "reddit"]
    legit_tlds    = [".com", ".org", ".net", ".edu", ".gov"]

    for _ in range(n):
        domain   = random.choice(legit_domains)
        tld      = random.choice(legit_tlds)
        path_len = np.random.randint(0, 30)
        path     = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=path_len))
        url      = f"https://www.{domain}{tld}/{path}"

        data.append({
            "url_length":      len(url),
            "num_dots":        url.count('.'),
            "num_hyphens":     url.count('-'),
            "num_at":          0,
            "num_question":    np.random.randint(0, 2),
            "num_ampersand":   np.random.randint(0, 2),
            "num_digits":      sum(c.isdigit() for c in url),
            "has_https":       1,
            "has_ip":          0,
            "subdomain_count": 1,
            "path_length":     path_len,
            "entropy":         shannon_entropy(url),
            "tld_suspicious":  0,
            "domain_age_days": np.random.randint(365, 6000),
            "label":           0
        })
    return data


def generate_phishing(n):
    data = []
    suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click"]
    fake_keywords   = ["secure", "login", "verify", "update", "account",
                       "banking", "confirm", "paypal", "ebay", "apple"]

    for _ in range(n):
        keyword  = random.choice(fake_keywords)
        tld      = random.choice(suspicious_tlds)
        rand_str = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789-", k=np.random.randint(5, 15)))
        path_len = np.random.randint(20, 120)
        path     = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789?=&%", k=path_len))
        use_ip   = np.random.choice([0, 1], p=[0.4, 0.6])
        domain   = f"{np.random.randint(100,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}" if use_ip else f"{keyword}-{rand_str}{tld}"
        url      = f"http://{domain}/{path}"

        data.append({
            "url_length":      len(url),
            "num_dots":        url.count('.'),
            "num_hyphens":     url.count('-'),
            "num_at":          np.random.randint(0, 3),
            "num_question":    np.random.randint(1, 5),
            "num_ampersand":   np.random.randint(1, 6),
            "num_digits":      sum(c.isdigit() for c in url),
            "has_https":       np.random.choice([0, 1], p=[0.7, 0.3]),
            "has_ip":          int(use_ip),
            "subdomain_count": np.random.randint(2, 6),
            "path_length":     path_len,
            "entropy":         shannon_entropy(url),
            "tld_suspicious":  0 if use_ip else 1,
            "domain_age_days": np.random.randint(1, 180),
            "label":           1
        })
    return data


def generate_dataset(total=2000, phishing_ratio=0.45, output_path="data/phishing_dataset.csv"):
    n_phishing  = int(total * phishing_ratio)
    n_legit     = total - n_phishing

    print(f"Generating {n_legit} legitimate + {n_phishing} phishing samples...")

    legit    = generate_legitimate(n_legit)
    phishing = generate_phishing(n_phishing)

    df = pd.DataFrame(legit + phishing)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to  : {output_path}")
    print(f"Shape             : {df.shape}")
    print(f"Label counts      :\n{df['label'].value_counts().rename({0:'Legitimate', 1:'Phishing'})}")
    print(f"\nSample rows:")
    print(df.head(5).to_string())
    return df


if __name__ == "__main__":
    generate_dataset(total=2000, phishing_ratio=0.45, output_path="data/phishing_dataset.csv")
