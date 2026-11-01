"""
Production Traffic Simulation Script for Google Play Store Rating API
Demonstrates normal traffic and intentionally drifted traffic to test drift detection and alerts.
"""

import time
import requests
import argparse

API_URL = "http://localhost:8001/predict"

NORMAL_APPS = [
    {
        "App": "Candy Crush Saga",
        "Category": "GAME",
        "Reviews": 22426677,
        "Size": "74M",
        "Installs": "500,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone",
        "Genres": "Casual",
        "Last Updated": "July 5, 2018",
        "Current Ver": "1.129.0",
        "Android Ver": "4.0.3 and up"
    },
    {
        "App": "Photo Editor & Candy Camera",
        "Category": "ART_AND_DESIGN",
        "Reviews": 159,
        "Size": "19M",
        "Installs": "10,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone",
        "Genres": "Art & Design",
        "Last Updated": "January 7, 2018",
        "Current Ver": "1.0.0",
        "Android Ver": "4.0.3 and up"
    },
    {
        "App": "Duolingo: Learn Languages Free",
        "Category": "EDUCATION",
        "Reviews": 6289924,
        "Size": "Varies with device",
        "Installs": "100,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone",
        "Genres": "Education",
        "Last Updated": "August 1, 2018",
        "Current Ver": "Varies with device",
        "Android Ver": "Varies with device"
    },
    {
        "App": "Subway Surfers",
        "Category": "GAME",
        "Reviews": 27722264,
        "Size": "76M",
        "Installs": "1,000,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone 10+",
        "Genres": "Arcade",
        "Last Updated": "July 12, 2018",
        "Current Ver": "1.90.0",
        "Android Ver": "4.1 and up"
    },
    {
        "App": "Clash of Clans",
        "Category": "GAME",
        "Reviews": 44891723,
        "Size": "98M",
        "Installs": "100,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone 10+",
        "Genres": "Strategy",
        "Last Updated": "July 15, 2018",
        "Current Ver": "10.322.16",
        "Android Ver": "4.1 and up"
    },
    {
        "App": "Google Drive",
        "Category": "PRODUCTIVITY",
        "Reviews": 2731171,
        "Size": "Varies with device",
        "Installs": "1,000,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone",
        "Genres": "Productivity",
        "Last Updated": "August 6, 2018",
        "Current Ver": "Varies with device",
        "Android Ver": "Varies with device"
    }
]

DRIFTED_APPS = [
    {
        "App": "I am Rich Premium Luxury VIP Edition",
        "Category": "FINANCE",
        "Reviews": 1,
        "Size": "999M",  # Extreme size anomaly
        "Installs": "5",  # Extremely low installs
        "Type": "Paid",
        "Price": "$399.99",  # Extreme price anomaly
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "December 31, 2026",
        "Current Ver": "99.0",
        "Android Ver": "15.0 and up"
    },
    {
        "App": "Super Diamond Expensive Crypto Wallet",
        "Category": "FINANCE",
        "Reviews": 2,
        "Size": "850M",
        "Installs": "10",
        "Type": "Paid",
        "Price": "$400.00",
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "November 1, 2026",
        "Current Ver": "50.0",
        "Android Ver": "14.0 and up"
    },
    {
        "App": "Most Expensive App 2026",
        "Category": "FINANCE",
        "Reviews": 3,
        "Size": "1024M",
        "Installs": "1",
        "Type": "Paid",
        "Price": "$399.99",
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "October 10, 2026",
        "Current Ver": "77.0",
        "Android Ver": "14.0 and up"
    },
    {
        "App": "Ultra Luxury Jet Finder",
        "Category": "FINANCE",
        "Reviews": 5,
        "Size": "750M",
        "Installs": "15",
        "Type": "Paid",
        "Price": "$350.00",
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "September 15, 2026",
        "Current Ver": "88.0",
        "Android Ver": "14.0 and up"
    },
    {
        "App": "Gold Bar Digital Vault",
        "Category": "FINANCE",
        "Reviews": 4,
        "Size": "900M",
        "Installs": "8",
        "Type": "Paid",
        "Price": "$380.00",
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "October 20, 2026",
        "Current Ver": "90.0",
        "Android Ver": "14.0 and up"
    },
    {
        "App": "Billionaire Club Access",
        "Category": "FINANCE",
        "Reviews": 2,
        "Size": "800M",
        "Installs": "20",
        "Type": "Paid",
        "Price": "$390.00",
        "Content Rating": "Adults only 18+",
        "Genres": "Finance",
        "Last Updated": "November 2, 2026",
        "Current Ver": "95.0",
        "Android Ver": "14.0 and up"
    }
]


def send_traffic(apps: list, mode_name: str):
    print(f"\n[INFO] Sending {len(apps)} requests in '{mode_name}' mode to {API_URL}...")
    success = 0
    for i, app_payload in enumerate(apps, start=1):
        try:
            resp = requests.post(API_URL, json=app_payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  [{i}/{len(apps)}] {data['app_name']:<35} -> Predicted Rating: {data['predicted_rating']} ({data['prediction_latency_ms']}ms)")
                success += 1
            else:
                print(f"  [{i}/{len(apps)}] Request failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"  [{i}/{len(apps)}] Connection error: {e}")
        time.sleep(0.1)

    print(f"[SUCCESS] Sent {success}/{len(apps)} requests successfully.")


def main():
    parser = argparse.ArgumentParser(description="Simulate production API traffic")
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal", help="Traffic mode: 'normal' or 'drift'")
    args = parser.parse_args()

    if args.mode == "normal":
        send_traffic(NORMAL_APPS, "NORMAL (Baseline Aligned)")
    else:
        send_traffic(DRIFTED_APPS, "DRIFT (High Price & Outlier Features)")


if __name__ == "__main__":
    main()
