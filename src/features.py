"""Turns a domain (or full URL) into the numbers the model learns from.

The feature definitions follow the columns of dataset/phishing_domains_1000.csv.
"""
import math
from collections import Counter
from urllib.parse import urlparse

FEATURES = [
    "length", "num_dots", "num_hyphens", "num_digits", "subdomain_count",
    "suspicious_tld", "sld_length", "has_brand", "has_suspicious_word",
    "digit_ratio", "vowel_ratio", "entropy",
]

BRANDS = ["paypal", "amazon", "google", "apple", "microsoft", "netflix", "facebook", "instagram",
          "sbi", "hdfcbank", "icicibank", "axisbank", "chase", "twitter", "linkedin", "dropbox",
          "spotify", "steam", "binance", "coinbase", "metamask", "whatsapp", "flipkart", "paytm",
          "fedex", "dhl", "usps", "outlook", "yahoo", "phonepe", "adobe", "bankofamerica",
          "wellsfargo", "office365"]
WORDS = ["login", "signin", "verify", "update", "secure", "account", "confirm", "password", "wallet",
         "billing", "support", "suspend", "free", "bonus", "alert", "recovery", "reward", "unlock",
         "refund", "auth", "validate", "payment", "service", "security", "webscr", "customer",
         "kyc", "session"]
BAD_TLDS = {"buzz", "info", "cf", "club", "cn", "ga", "gq", "icu", "live", "ml", "online", "rest", "ru",
            "shop", "site", "support", "tk", "top", "work", "xyz"}
TWO_PART_TLDS = {"co.uk", "co.in", "ac.in", "gov.in", "org.uk", "ac.uk", "com.au", "co.jp"}


def get_domain(text: str) -> str:
    """'https://Login.Example.com:80/a?b' -> 'login.example.com'"""
    text = text.strip()
    host = urlparse(text if "://" in text else "//" + text).hostname or text
    return host.lower().rstrip(".")


def extract(text: str) -> dict:
    domain = get_domain(text)
    labels = domain.split(".")
    n_tld = 2 if ".".join(labels[-2:]) in TWO_PART_TLDS else 1
    tld = labels[-n_tld:]
    sld = labels[-n_tld - 1] if len(labels) > n_tld else labels[0]
    digits = sum(c.isdigit() for c in domain)
    n = max(len(domain), 1)
    counts = Counter(domain)
    return {
        "length": len(domain),
        "num_dots": domain.count("."),
        "num_hyphens": domain.count("-"),
        "num_digits": digits,
        "subdomain_count": max(len(labels) - n_tld - 1, 0),
        "suspicious_tld": int(labels[-1] in BAD_TLDS and n_tld == 1),
        "sld_length": len(sld),
        "has_brand": int(any(b in domain for b in BRANDS)),
        "has_suspicious_word": int(any(w in domain for w in WORDS)),
        "digit_ratio": round(digits / n, 4),
        "vowel_ratio": round(sum(c in "aeiou" for c in domain) / n, 4),
        "entropy": round(-sum(v / n * math.log2(v / n) for v in counts.values()), 4),
    }
