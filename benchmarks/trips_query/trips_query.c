// trips_query.c — same workload as trips_query.mim, hand-rolled.
// Generate 200k rows (identical LCG), join zone→borough, group by
// (borough, pay), aggregate, filter avg_fare > 15, tip share, sort by
// revenue desc, print checksum rows.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N 200000
#define NZ 265

static const char *BOROUGHS[5] = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"};

typedef struct {
    int zone, pay;
    double dist, fare, tip;
    int hour;
} Trip;

typedef struct {
    int borough, pay;
    long trips;
    double revenue, tips, fare_sum;
} Group;

static int cmp_rev(const void *a, const void *b) {
    double d = ((const Group *)b)->revenue - ((const Group *)a)->revenue;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

int main(void) {
    Trip *trips = malloc(sizeof(Trip) * N);
    int64_t s = 42;
    for (int i = 0; i < N; i++) {
        s = (1103515245 * s + 12345) % 2147483648;
        trips[i].zone = 1 + s % 265;
        s = (1103515245 * s + 12345) % 2147483648;
        trips[i].pay = 1 + s % 4;
        s = (1103515245 * s + 12345) % 2147483648;
        trips[i].dist = (50 + s % 2000) / 100.0;
        s = (1103515245 * s + 12345) % 2147483648;
        int64_t fare_c = 500 + s % 4500;
        s = (1103515245 * s + 12345) % 2147483648;
        int64_t tip_c = s % 3000;
        s = (1103515245 * s + 12345) % 2147483648;
        trips[i].hour = s % 24;
        trips[i].fare = fare_c / 100.0;
        trips[i].tip = tip_c / 100.0;
    }

    // join + group by (borough, pay): only 5×4 slots, index directly
    Group g[20] = {0};
    int used[20] = {0};
    double total_tips = 0;
    for (int i = 0; i < N; i++) {
        int borough = (trips[i].zone - 1) % 5;
        int slot = borough * 4 + (trips[i].pay - 1);
        if (!used[slot]) {
            used[slot] = 1;
            g[slot].borough = borough;
            g[slot].pay = trips[i].pay;
        }
        g[slot].trips++;
        g[slot].revenue += trips[i].fare;
        g[slot].tips += trips[i].tip;
        g[slot].fare_sum += trips[i].fare;
        total_tips += trips[i].tip;
    }

    // filter avg_fare > 15, sort by revenue desc, print
    Group out[20];
    int m = 0;
    for (int i = 0; i < 20; i++)
        if (used[i] && g[i].fare_sum / g[i].trips > 15) out[m++] = g[i];
    qsort(out, m, sizeof(Group), cmp_rev);
    for (int i = 0; i < m; i++)
        printf("%s|%d|%ld|%.2f|%.2f|%.2f|%.4f\n",
               BOROUGHS[out[i].borough], out[i].pay, out[i].trips,
               out[i].revenue, out[i].fare_sum / out[i].trips,
               out[i].tips, out[i].tips / total_tips);
    free(trips);
    return 0;
}
