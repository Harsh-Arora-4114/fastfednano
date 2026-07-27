# FAST-FedNano — Frontend on Vercel, ML Model on Cloudflare Workers

## Ye kaise kaam karta hai

Cloudflare Workers Python (sklearn/xgboost) nahi chala sakte — isliye
tere trained models ke **raw numbers** (RF ke trees, XGBoost ke trees,
Neural Net ke weights, StandardScaler ke mean/scale) ko ek JSON file mein
nikaal ke, Cloudflare Worker ke andar **pure JavaScript** mein prediction
math likh diya hai. Koi Python runtime chahiye hi nahi at request time —
bas number-crunching, jo Workers bahut fast karte hain (edge compute).

```
fastfednano/
├── export_model.py           # LOCAL par chalao — .joblib -> model_bundle.json
├── ml-worker/                  # Cloudflare Worker (pure JS ML inference)
│   ├── src/index.js             # /predict, /health — RF + XGB + NN sab JS mein
│   ├── models/model_bundle.json # <-- export_model.py ka output yahan aayega
│   ├── wrangler.toml
│   └── package.json
└── frontend/
    ├── index.html                # Vercel par deploy hoga
    └── vercel.json
```

---

## Step 0 — Models ko JSON mein export karo (apne local machine par)

Apne asli `.joblib` files ke folder mein `export_model.py` copy karo aur:

```bash
pip install scikit-learn xgboost joblib numpy --break-system-packages
python export_model.py
```

Ye `model_bundle.json` banayega. Isko copy karo:
```
ml-worker/models/model_bundle.json   (placeholder file ko replace karo)
```

**Important**: script kuch WARNING print kar sakta hai (agar tera
preprocessor mein categorical columns hain, ya XGBoost ka objective
`reg:squarederror` nahi hai, ya NN activation `relu` nahi hai) — agar
koi warning aaye to mujhe bata dena, main JS inference usi hisaab se
adjust kar dunga. Bina real files dekhe main ye guarantee nahi kar
sakta ki edge cases sab cover hain.

---

## Step 1 — Cloudflare Worker deploy karo

Iske liye Wrangler CLI use karna padega (JSON bundle ko Worker ke saath
package karne ke liye — sirf dashboard se ye possible nahi hai, but baaki
sab kaam CLI khud Cloudflare console/account se hi connect karke karta hai).

```bash
cd ml-worker
npm install
npx wrangler login       # ye browser kholega, Cloudflare account se login karo
npx wrangler deploy
```

Deploy hone ke baad terminal mein URL milega, jaisे:
```
https://fastfednano-ml-worker.<your-subdomain>.workers.dev
```

Test karo:
```bash
curl https://fastfednano-ml-worker.<your-subdomain>.workers.dev/health
```

**Dashboard se verify/manage**: https://dash.cloudflare.com → **Workers &
Pages** → tera `fastfednano-ml-worker` project dikh jaayega — yahan se
logs, custom domain, analytics sab manage kar sakta hai (deploy CLI se
hua, but manage dashboard se bhi hota hai).

---

## Step 2 — Frontend deploy karo (Vercel Console)

1. `frontend/index.html` mein Worker URL update karo:
   ```js
   const API_BASE = ... "https://fastfednano-ml-worker.<your-subdomain>.workers.dev";
   ```
2. Poore project (`fastfednano/`) ko GitHub par push karo:
   ```bash
   cd fastfednano
   git init && git add . && git commit -m "FAST-FedNano: Vercel + Cloudflare Worker"
   git remote add origin https://github.com/<your-username>/fastfednano.git
   git branch -M main && git push -u origin main
   ```
3. https://vercel.com par jaake sign up/login karo (GitHub se login sabse
   aasan hai).
4. Dashboard → **Add New** → **Project** → apna `fastfednano` repo
   **Import** karo.
5. Configure screen par:
   - **Root Directory**: `frontend` (Edit button se select karo)
   - **Framework Preset**: **Other**
   - Build/Output settings — khaali chhod do, static HTML hai
6. **Deploy** click karo. 1 minute mein live ho jaayega:
   `https://fastfednano.vercel.app`

---

## Step 3 — CORS lock karo (production ke liye)

Abhi `ml-worker/src/index.js` mein CORS `"*"` (sab origins allow) pe hai,
taaki testing aasan ho. Jab Vercel URL final ho jaaye, to:

```js
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "https://fastfednano.vercel.app",
  ...
};
```
badal ke phir se `npx wrangler deploy` chala do.

---

## Step 4 (optional) — Apna custom domain

- **Cloudflare Worker** ko custom subdomain dena ho (e.g. `api.fastfednano.com`):
  Cloudflare dashboard → tera Worker → **Settings → Domains & Routes** →
  **Add Custom Domain** → domain daalo (agar domain already Cloudflare
  mein hai to ek click mein ho jaata hai).
- **Vercel frontend** ko custom domain dena ho: Vercel project →
  **Settings → Domains** → apna domain add karo → Vercel DNS instructions
  dikha dega (ya agar domain Cloudflare pe hai, to bas CNAME record daal do).

---

## Cost check (₹0/month)

| Resource | Free tier | Tera usage |
|---|---|---|
| Cloudflare Workers | 100,000 requests/day, permanent free tier | trivial |
| Vercel Hobby plan | 100GB bandwidth/month, unlimited static deploys | trivial |

Ye setup Render/AWS wale se **behtar** hai kyunki: (1) koi cold-start sleep
nahi (Workers hamesha warm hain), (2) Python container maintain nahi
karna, (3) dono free tiers genuinely permanent hain, trial nahi.

---

## Limitation jo yaad rakhni hai

- Agar tera dataset mein categorical columns hain (one-hot encoded), ye
  JS worker unhe handle nahi karta abhi — sirf numeric StandardScaler
  path implement hai (`export_model.py` tujhe warning dega agar aisa hua).
- XGBoost inference `reg:squarederror` objective assume karta hai (identity
  link). Agar tune kuch aur objective use kiya tha, prediction galat aa
  sakti hai — export script isko bhi detect karke warn karega.
- Model bundle size: agar RF/XGBoost mein bahut zyada trees hain (300+),
  JSON bundle bada ho sakta hai. Worker free tier 10MB tak (compressed)
  allow karta hai — agar isse bada ho jaaye, bata dena, Worker ko
  Cloudflare R2 se bundle fetch karne ke liye switch kar dunga.
