import json
import config as C
from rag import authorized_search

def main():
    golden = json.load(open("golden_set.json"))
    leaks, checks = [], 0
    for item in golden:
        needed = item["min_clearance"]
        for username, u in C.USERS.items():
            clearance = u["clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=clearance)
            # Any retrieved chunk above this user's clearance = a leak.
            for h in hits:
                checks += 1
                if h.payload["classification_rank"] > clearance:
                    leaks.append((username, item["question"], h.payload["source"]))

    print(f"Authorization checks run: {checks}")
    print(f"LEAKS DETECTED: {len(leaks)}")
    for user, q, src in leaks:
        print(f"  ❌ {user} retrieved '{src}' via: {q!r}")
    if not leaks:
        print("  ✅ No leaks. Authorization is structurally enforced.")

    # Sanity check the inverse: authorized users CAN reach what they need.
    print("\nAccess sanity check:")
    for item in golden:
        # "Unanswerable" questions have no relevant source by design — there is
        # nothing to retrieve, so the access check doesn't apply to them.
        if not item["relevant_sources"]:
            continue
        for username, u in C.USERS.items():
            should_access = u["clearance"] >= item["min_clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=u["clearance"])
            got = any(h.payload["source"] in item["relevant_sources"] for h in hits)
            status = "ok" if got == should_access else "MISMATCH"
            if status == "MISMATCH":
                print(f"  ⚠️  {username} / {item['question'][:40]!r}: "
                      f"expected_access={should_access}, got={got}")

if __name__ == "__main__":
    main()
