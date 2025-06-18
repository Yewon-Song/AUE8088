#!/usr/bin/env python3
"""
EvalAI 제출 스크립트 (runs/val/<target_runs>/best_predictions.json 자동 지정)

사용 예:
    python submit_evalai.py --target-runs exp42
"""

import argparse
import time
from pathlib import Path
from pprint import pprint

import requests


# ======================= CLI ======================= #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target-runs",
        "-t",
        default="exp",
        help="runs/val/<target_runs>/best_predictions.json 에서 <target_runs> 값 (기본: exp)",
    )
    return p.parse_args()


# ======================= 상수 ======================= #
API_URL = (
    "http://15.165.31.246:8000/api/jobs/challenge/21/"
    "challenge_phase/41/submission/"
)
HEADERS = {
    "Authorization": "Token 45f4d88f695556b6d93fd15fbccb7524984a09cf",
}
PAYLOAD = {
    "status": "submitting",
    "method_name": "",
    "method_description": "",
    "project_url": "",
    "publication_url": "",
    "submission_metadata": "null",
    "is_public": "true",
}


# ======================= 로직 ======================= #
def submit_file(json_path: Path) -> int:
    if not json_path.is_file():
        raise FileNotFoundError(f"⚠️ 파일을 찾을 수 없습니다: {json_path}")

    with json_path.open("rb") as fp:
        print(f"📤 업로드 중... ({json_path})")
        resp = requests.post(API_URL, headers=HEADERS, data=PAYLOAD, files={"input_file": fp})

    if resp.status_code != 201:
        raise RuntimeError(f"업로드 실패 ({resp.status_code})\n{resp.text}")

    sub_id = resp.json()["id"]
    print(f"✅ 업로드 성공! 제출 ID: {sub_id}")
    return sub_id


def poll_result(sub_id: int):
    time.sleep(5)  # 대기 (필요 시 조정)
    resp = requests.get(API_URL, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"결과 조회 실패\n{resp.text}")

    for item in resp.json().get("results", []):
        if item.get("id") == sub_id:
            res_url = item.get("submission_result_file")
            break
    else:
        print("⌛ 결과 파일이 아직 준비되지 않았거나 None입니다.")
        return

    res_resp = requests.get(res_url)
    try:
        print("✅ 결과:")
        pprint(res_resp.json())
    except Exception:
        print("❌ 결과 JSON 파싱 실패")
        print(res_resp.text)


def main():
    args = parse_args()

    # runs/val/<target_runs>/best_predictions.json
    json_path = Path.cwd() / "runs" / "val" / args.target_runs / "best_predictions.json"

    sub_id = submit_file(json_path)
    poll_result(sub_id)


if __name__ == "__main__":
    main()
