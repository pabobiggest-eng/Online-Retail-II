# Customer Repeat-Purchase Behaviour: Findings and Recommendations

## Group definition
- Non-repeat / target 0: 2,778 customers (55.82% of this dataset).
- Repeat / target 1: 2,199 customers (44.18% of this dataset).
- These labels refer to the saved target repeat_purchase_next_90d.

## What distinguishes the groups?
The table below compares the actual median feature values. Differences are associations, not proof of customer motivation.

| Behaviour feature | Non-repeat median | Repeat median | Difference |
|---|---:|---:|---:|
| total_spend | 320.685 | 633.990 | 313.305 |
| total_quantity | 174.000 | 353.000 | 179.000 |
| maximum_basket_value | 289.785 | 366.580 | 76.795 |
| average_basket_value | 250.020 | 304.400 | 54.380 |
| recency_days_at_day_90 | 90.000 | 46.965 | -43.035 |
| items_per_invoice | 130.000 | 166.667 | 36.667 |
| unique_products | 20.000 | 32.000 | 12.000 |
| unique_invoices | 1.000 | 2.000 | 1.000 |
| active_purchase_days | 1.000 | 2.000 | 1.000 |
| average_days_between_invoices | 28.803 | 29.422 | 0.619 |

## Interpretation for the client

Use the measured group differences above to identify behaviours associated with repeat purchasing:
- If repeat customers have more invoices or active purchase days, test loyalty rewards and convenient reorder reminders.
- If they buy more unique products, test relevant cross-sell recommendations based on prior purchases.
- If recency differs, test carefully timed re-engagement messages for customers who have become inactive.
- If average order gaps differ, test reminders near customers' observed reorder intervals.
- If spend or basket value differs, test relevant bundles or thresholds; do not assume discounts are necessary.

## Why some customers may not repeat
This transaction dataset can identify lower purchase frequency, fewer observed products, longer recency, or other measured differences when the table supports them. It cannot establish that customers were unhappy, found prices too high, or experienced delivery problems because those causes are not directly recorded.

## Recommended next steps
1. Prioritise customers whose behaviour resembles the measured repeat-customer pattern.
2. Run a small, controlled test of reorder reminders, relevant recommendations, or loyalty offers.
3. Compare repeat-purchase rate and revenue between test and comparison groups.
4. Collect optional feedback on price, product fit, and service to investigate the actual reasons for not returning.

## Files
- Detailed group comparison: `outputs/reports/repeat_vs_nonrepeat_comparison.csv`
- Ranked behavioural differences: `outputs/reports/repeat_purchase_pattern_ranking.csv`
- Chart: `outputs/figures/repeat_vs_nonrepeat_patterns.png`