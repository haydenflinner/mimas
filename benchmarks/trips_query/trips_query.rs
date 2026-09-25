// trips_query.rs — same workload as trips_query.mim, hand-rolled in Rust.
use std::collections::HashMap;

const BOROUGHS: [&str; 5] = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"];
const N: usize = 200_000;

#[derive(Default)]
struct Group {
    trips: i64,
    revenue: f64,
    tips: f64,
    fare_sum: f64,
}

fn main() {
    let mut s: i64 = 42;
    let mut next = move || {
        s = (1103515245 * s + 12345) % 2147483648;
        s
    };
    let mut trips = Vec::with_capacity(N);
    for _ in 0..N {
        let zone = 1 + next() % 265;
        let pay = 1 + next() % 4;
        let dist = (50 + next() % 2000) as f64 / 100.0;
        let fare_c = 500 + next() % 4500;
        let tip_c = next() % 3000;
        let hour = next() % 24;
        trips.push((zone, pay, dist, fare_c as f64 / 100.0, tip_c as f64 / 100.0, hour));
    }

    let mut groups: HashMap<(usize, i64), Group> = HashMap::new();
    let mut total_tips = 0.0;
    for &(zone, pay, _dist, fare, tip, _hour) in &trips {
        let g = groups.entry(((zone - 1) as usize % 5, pay)).or_default();
        g.trips += 1;
        g.revenue += fare;
        g.tips += tip;
        g.fare_sum += fare;
        total_tips += tip;
    }

    let mut out: Vec<((usize, i64), Group)> = groups
        .into_iter()
        .filter(|(_, g)| g.fare_sum / g.trips as f64 > 15.0)
        .collect();
    out.sort_by(|a, b| b.1.revenue.partial_cmp(&a.1.revenue).unwrap());
    for ((borough, pay), g) in out {
        println!(
            "{}|{}|{}|{:.2}|{:.2}|{:.2}|{:.4}",
            BOROUGHS[borough],
            pay,
            g.trips,
            g.revenue,
            g.fare_sum / g.trips as f64,
            g.tips,
            g.tips / total_tips
        );
    }
}
