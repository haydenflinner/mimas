// trips_query.cpp — same workload as trips_query.mim, hand-rolled in C++.
// The "query" is a hash aggregation + filter + sort written by hand;
// idiomatic C++ (vector, map is unnecessary — 20 slots), still ~3× the
// line count of the mimas version.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

static const std::array<const char *, 5> BOROUGHS = {
    "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"};

struct Trip {
    int zone, pay;
    double dist, fare, tip;
    int hour;
};

struct Group {
    int borough = 0, pay = 0;
    long trips = 0;
    double revenue = 0, tips = 0, fare_sum = 0;
};

int main() {
    const int N = 200000;
    std::vector<Trip> trips(N);
    int64_t s = 42;
    for (auto &t : trips) {
        s = (1103515245 * s + 12345) % 2147483648;
        t.zone = 1 + s % 265;
        s = (1103515245 * s + 12345) % 2147483648;
        t.pay = 1 + s % 4;
        s = (1103515245 * s + 12345) % 2147483648;
        t.dist = (50 + s % 2000) / 100.0;
        s = (1103515245 * s + 12345) % 2147483648;
        int64_t fare_c = 500 + s % 4500;
        s = (1103515245 * s + 12345) % 2147483648;
        int64_t tip_c = s % 3000;
        s = (1103515245 * s + 12345) % 2147483648;
        t.hour = s % 24;
        t.fare = fare_c / 100.0;
        t.tip = tip_c / 100.0;
    }

    std::array<Group, 20> g{};
    std::array<bool, 20> used{};
    double total_tips = 0;
    for (const auto &t : trips) {
        int borough = (t.zone - 1) % 5;
        int slot = borough * 4 + (t.pay - 1);
        auto &grp = g[slot];
        if (!used[slot]) {
            used[slot] = true;
            grp.borough = borough;
            grp.pay = t.pay;
        }
        grp.trips++;
        grp.revenue += t.fare;
        grp.tips += t.tip;
        grp.fare_sum += t.fare;
        total_tips += t.tip;
    }

    std::vector<Group> out;
    for (int i = 0; i < 20; i++)
        if (used[i] && g[i].fare_sum / g[i].trips > 15) out.push_back(g[i]);
    std::sort(out.begin(), out.end(),
              [](const Group &a, const Group &b) { return a.revenue > b.revenue; });
    for (const auto &o : out)
        std::printf("%s|%d|%ld|%.2f|%.2f|%.2f|%.4f\n",
                    BOROUGHS[o.borough], o.pay, o.trips,
                    o.revenue, o.fare_sum / o.trips,
                    o.tips, o.tips / total_tips);
}
