# CareerBuilder Job Recommendation Benchmark Plan

Use the Kaggle CareerBuilder Job Recommendation Challenge as an implicit-feedback
benchmark, not as a direct human-labelled CV-JD relevance benchmark.

## Benchmark Framing

- Query: a user/candidate profile assembled from user metadata and job history.
- Candidate items: job postings.
- Positive pairs: observed applications from `apps.tsv`.
- Negative pairs: sampled jobs from the same window that the user did not apply to.
- Evaluation: rank jobs for each user and report Recall@K, MAP@K, and NDCG@K.

This dataset should be presented as a job recommendation benchmark. It is useful
for testing ranking scalability and implicit-feedback retrieval, but it does not
replace a human relevance benchmark such as Vanetik or PJB.

## Recommended Baselines

- BM25: user profile/history text as query, job posting as document.
- SBERT cosine: encode user profile/history and job posting, rank by cosine.
- O*NET/entity-only: rank by extracted skill/occupation overlap when enough text is available.
- Proposed: semantic score plus normalized skill/role/seniority features.

## Data Access

The competition data requires Kaggle authentication and accepting competition rules:

```powershell
kaggle competitions download -c job-recommendation -p .\external\careerbuilder
Expand-Archive .\external\careerbuilder\job-recommendation.zip -DestinationPath .\external\careerbuilder -Force
```

If using `kagglehub`, authentication is still required for this competition.

## Reporting Note

Describe labels as implicit application feedback:

> Positive labels indicate that a user applied to a job. Missing interactions are
> treated as sampled negatives for ranking evaluation, but they are not guaranteed
> to be true negative relevance judgments.

## Current Local Runs

All-domain sample:

```powershell
python benchmark_careerbuilder.py `
  --data-dir .\job-recommendation `
  --output-dir .\artifacts\careerbuilder_all_100 `
  --num-users 100 `
  --negatives-per-user 30 `
  --positives-per-user 1 `
  --job-filter all `
  --embedding-device cuda `
  --embedding-batch-size 64 `
  --max-onet-terms 12000 `
  --overwrite
```

| Method | MAP | MRR | NDCG@1 | NDCG@5 | NDCG@10 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5345 | 0.5345 | 0.3800 | 0.5561 | 0.5973 | 0.7000 |
| SBERT cosine | 0.4308 | 0.4308 | 0.3000 | 0.4397 | 0.4787 | 0.5700 |
| O*NET entity-only | 0.1503 | 0.1503 | 0.0300 | 0.1263 | 0.1571 | 0.2100 |
| Proposed hybrid | 0.5725 | 0.5725 | 0.4400 | 0.5842 | 0.6253 | 0.7000 |

IT-filtered sample:

```powershell
python benchmark_careerbuilder.py `
  --data-dir .\job-recommendation `
  --output-dir .\artifacts\careerbuilder_it_50 `
  --num-users 50 `
  --negatives-per-user 30 `
  --positives-per-user 1 `
  --job-filter it `
  --embedding-device cuda `
  --embedding-batch-size 64 `
  --max-onet-terms 12000 `
  --overwrite
```

| Method | MAP | MRR | NDCG@1 | NDCG@5 | NDCG@10 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5330 | 0.5330 | 0.3800 | 0.5477 | 0.5934 | 0.6800 |
| SBERT cosine | 0.3784 | 0.3784 | 0.2400 | 0.4068 | 0.4519 | 0.6000 |
| O*NET entity-only | 0.1925 | 0.1925 | 0.0800 | 0.1681 | 0.2193 | 0.2400 |
| Proposed hybrid | 0.5515 | 0.5515 | 0.4200 | 0.5647 | 0.5898 | 0.6800 |

The proposed hybrid method improves MAP, MRR, NDCG@1, and NDCG@5 over the BM25,
SBERT cosine, and O*NET-only baselines in both local samples. BM25 remains a
strong keyword baseline on CareerBuilder because user history is mostly job-title
text, while SBERT is weaker when the query profile is sparse.

## Improved Benchmark Runs

The improved setup uses:

- 3 positive applications per user instead of 1.
- User-profile enrichment from previously applied job descriptions.
- Multiple random seeds where feasible.
- Validation-based tuning for the proposed hybrid weights.
- O*NET lexical entity baseline, plus a semantic O*NET term-bank implementation for smaller diagnostic runs.

All-domain improved run:

```powershell
python benchmark_careerbuilder.py `
  --data-dir .\job-recommendation `
  --output-dir .\artifacts\careerbuilder_improved_all_200_s3 `
  --num-users 200 `
  --negatives-per-user 30 `
  --positives-per-user 3 `
  --job-filter all `
  --embedding-device cuda `
  --embedding-batch-size 64 `
  --max-onet-terms 12000 `
  --onet-scoring lexical `
  --enrich-profile-with-apps `
  --max-profile-apps 5 `
  --tune-weights `
  --valid-fraction 0.25 `
  --seeds 13,42,77 `
  --overwrite
```

| Method | MAP mean+/-std | MRR mean+/-std | NDCG@5 mean+/-std | NDCG@10 mean+/-std | Recall@5 mean+/-std |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7216+/-0.0095 | 0.9118+/-0.0109 | 0.7369+/-0.0075 | 0.7772+/-0.0108 | 0.7059+/-0.0136 |
| SBERT cosine | 0.6462+/-0.0082 | 0.8738+/-0.0087 | 0.6649+/-0.0070 | 0.7247+/-0.0055 | 0.6426+/-0.0128 |
| O*NET entity-only | 0.3828+/-0.0047 | 0.6068+/-0.0005 | 0.3922+/-0.0070 | 0.4448+/-0.0026 | 0.3893+/-0.0091 |
| Proposed static | 0.6829+/-0.0121 | 0.8875+/-0.0138 | 0.7054+/-0.0160 | 0.7617+/-0.0130 | 0.6941+/-0.0242 |
| Proposed tuned | 0.7237+/-0.0075 | 0.9147+/-0.0144 | 0.7390+/-0.0052 | 0.7788+/-0.0091 | 0.7074+/-0.0118 |

IT-filtered improved run:

```powershell
python benchmark_careerbuilder.py `
  --data-dir .\job-recommendation `
  --output-dir .\artifacts\careerbuilder_improved_it_100_s2 `
  --num-users 100 `
  --negatives-per-user 30 `
  --positives-per-user 3 `
  --job-filter it `
  --embedding-device cuda `
  --embedding-batch-size 64 `
  --max-onet-terms 12000 `
  --onet-scoring lexical `
  --enrich-profile-with-apps `
  --max-profile-apps 5 `
  --tune-weights `
  --valid-fraction 0.25 `
  --seeds 13,42 `
  --overwrite
```

| Method | MAP mean+/-std | MRR mean+/-std | NDCG@5 mean+/-std | NDCG@10 mean+/-std | Recall@5 mean+/-std |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5718+/-0.0382 | 0.8323+/-0.0229 | 0.5837+/-0.0366 | 0.6421+/-0.0368 | 0.5422+/-0.0356 |
| SBERT cosine | 0.5465+/-0.0217 | 0.7887+/-0.0163 | 0.5559+/-0.0139 | 0.6362+/-0.0254 | 0.5378+/-0.0267 |
| O*NET entity-only | 0.3128+/-0.0161 | 0.4966+/-0.0009 | 0.3103+/-0.0282 | 0.3597+/-0.0284 | 0.3200+/-0.0489 |
| Proposed static | 0.5641+/-0.0195 | 0.8194+/-0.0257 | 0.5794+/-0.0126 | 0.6578+/-0.0196 | 0.5578+/-0.0244 |
| Proposed tuned | 0.5748+/-0.0351 | 0.8327+/-0.0212 | 0.5882+/-0.0318 | 0.6446+/-0.0342 | 0.5489+/-0.0289 |

Interpretation:

- Enriched profiles make semantic features stronger, because user history is no longer limited to short job titles.
- Proposed tuned is the best overall method on all-domain CareerBuilder.
- On IT-filtered CareerBuilder, tuned hybrid slightly improves over BM25 on MAP, MRR, and NDCG@5, while static hybrid gives the best NDCG@10 and Recall@5.
- BM25 remains a strong baseline because CareerBuilder user profiles are dominated by previous job titles and applications.
