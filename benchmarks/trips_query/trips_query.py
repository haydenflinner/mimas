# trips_query.py — same workload as trips_query.mim, stdlib Python.
# dict-based group-by: the query as a hand-written aggregation loop.
N = 200000
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

def gen():
    s = 42
    trips = []
    for _ in range(N):
        s = (1103515245 * s + 12345) % 2147483648
        zone = 1 + s % 265
        s = (1103515245 * s + 12345) % 2147483648
        pay = 1 + s % 4
        s = (1103515245 * s + 12345) % 2147483648
        dist = (50 + s % 2000) / 100.0
        s = (1103515245 * s + 12345) % 2147483648
        fare_c = 500 + s % 4500
        s = (1103515245 * s + 12345) % 2147483648
        tip_c = s % 3000
        s = (1103515245 * s + 12345) % 2147483648
        hour = s % 24
        trips.append((zone, pay, dist, fare_c / 100.0, tip_c / 100.0, hour))
    return trips

trips = gen()
groups = {}
total_tips = 0.0
for zone, pay, _dist, fare, tip, _hour in trips:
    key = (BOROUGHS[(zone - 1) % 5], pay)
    g = groups.setdefault(key, [0, 0.0, 0.0, 0.0])
    g[0] += 1
    g[1] += fare      # revenue
    g[2] += tip       # tips
    g[3] += fare      # fare_sum
    total_tips += tip

out = [
    (key, g) for key, g in groups.items() if g[3] / g[0] > 15
]
out.sort(key=lambda kv: -kv[1][1])
for (borough, pay), g in out:
    print(f"{borough}|{pay}|{g[0]}|{g[1]:.2f}|{g[3]/g[0]:.2f}|{g[2]:.2f}|{g[2]/total_tips:.4f}")
