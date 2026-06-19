"""Score one project's generated takeoff xlsx vs the team's xlsx.

Reuses benchmark_samples.read_takeoff_xlsx + score_project so the numbers are
identical to the regression harness. Usage:
    python score_one.py <our_takeoff.xlsx> <team_takeoff.xlsx>
"""
import sys
from benchmark_samples import read_takeoff_xlsx, score_project, aggregate_by_product, aggregate_by_product_tag


def main():
    our_path, team_path = sys.argv[1], sys.argv[2]
    our_rows, our_err = read_takeoff_xlsx(our_path)
    team_rows, team_err = read_takeoff_xlsx(team_path)
    if our_err:
        print(f"ERROR reading our xlsx: {our_err}"); sys.exit(1)
    if team_err:
        print(f"ERROR reading team xlsx: {team_err}"); sys.exit(1)

    score = score_project(team_rows, our_rows)
    print("=" * 64)
    print(f"OUR : {our_path}")
    print(f"TEAM: {team_path}")
    print("=" * 64)
    print(f"team_total      : {score.get('team_total')}")
    print(f"our_total       : {score.get('our_total')}")
    print(f"product_recall  : {score['product_recall']:.1%}")
    print(f"product_precision: {score['product_precision']:.1%}")
    print(f"tag_recall      : {score['tag_recall']:.1%}")
    print(f"tag_precision   : {score['tag_precision']:.1%}")

    team_p = aggregate_by_product(team_rows)
    our_p = aggregate_by_product(our_rows)
    print("\nPer-product  (PRODUCT: team -> ours):")
    for prod in sorted(set(team_p) | set(our_p)):
        t, o = team_p.get(prod, 0), our_p.get(prod, 0)
        flag = "" if t and o else ("  <-- MISSED" if t and not o else "  <-- phantom" if o and not t else "")
        print(f"  {prod:<40} {t:>4} -> {o:>4}{flag}")


if __name__ == '__main__':
    main()
