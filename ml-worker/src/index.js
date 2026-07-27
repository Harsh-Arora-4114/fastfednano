/**
 * FAST-FedNano | Cloudflare Worker — ML inference in pure JavaScript
 * ------------------------------------------------------------------
 * No Python runtime involved. model_bundle.json (produced by
 * export_model.py, run locally against your real .joblib files) is
 * bundled directly into this Worker at deploy time. Every request runs
 * StandardScaler + RandomForest + XGBoost + a small NN forward pass,
 * all in JS, at the edge.
 *
 * Endpoints:
 *   GET  /health   -> {status, target, n_trees_rf, n_trees_xgb}
 *   POST /predict  -> {target, predictions, ensemble_mean}
 */

import modelBundle from "../models/model_bundle.json";

const {
  target: TARGET_COL,
  feature_names: FEATURE_NAMES,
  scaler: SCALER,
  random_forest: RF,
  xgboost: XGB,
  neural_network: NN,
} = modelBundle;

// ---------------------------------------------------------------
// CORS — restrict this to your Vercel domain in production
// ---------------------------------------------------------------
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

// ---------------------------------------------------------------
// 1. Preprocessing — median-impute (rarely hit here since inputs are
//    required) + StandardScaler, matching data_preprocessing.py
// ---------------------------------------------------------------
function scaleInput(rawRow) {
  return FEATURE_NAMES.map((name, i) => {
    let v = rawRow[name];
    if (v === null || v === undefined || Number.isNaN(v)) {
      v = SCALER.impute_values[i];
    }
    return (v - SCALER.mean[i]) / SCALER.scale[i];
  });
}

// ---------------------------------------------------------------
// 2. Random Forest — average prediction across sklearn's exported trees
//    (children_left/right, feature, threshold, value arrays)
// ---------------------------------------------------------------
function predictOneTree(tree, x) {
  let node = 0;
  while (tree.feature[node] !== -2) {
    // -2 == sklearn's TREE_LEAF / TREE_UNDEFINED sentinel
    const f = tree.feature[node];
    node = x[f] <= tree.threshold[node] ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}

function predictRandomForest(x) {
  let sum = 0;
  for (const tree of RF.trees) sum += predictOneTree(tree, x);
  return sum / RF.trees.length;
}

// ---------------------------------------------------------------
// 3. XGBoost — sum of leaf values across trees (get_dump json format:
//    each node has {split, split_condition, yes, no, missing, children}
//    or {leaf: value} at terminal nodes), plus base_score.
// ---------------------------------------------------------------
function buildNodeMap(tree) {
  const map = {};
  function walk(node) {
    map[node.nodeid] = node;
    if (node.children) node.children.forEach(walk);
  }
  walk(tree);
  return map;
}

function predictOneXgbTree(tree, x) {
  const nodeMap = buildNodeMap(tree);
  let node = nodeMap[0];
  while (node.leaf === undefined) {
    // split field looks like "f3" (index into FEATURE_NAMES) or the raw name
    const featIdx = featureIndexFromSplitName(node.split);
    const val = x[featIdx];
    let nextId;
    if (val === null || val === undefined || Number.isNaN(val)) {
      nextId = node.missing;
    } else {
      nextId = val < node.split_condition ? node.yes : node.no;
    }
    node = nodeMap[nextId];
  }
  return node.leaf;
}

function featureIndexFromSplitName(split) {
  // xgboost dumps splits as "f<index>" by default when trained on a
  // plain numpy array (no column names attached) — which is the case
  // here since PREPROCESSOR.transform() returns a numpy array.
  if (split.startsWith("f")) return parseInt(split.slice(1), 10);
  return FEATURE_NAMES.indexOf(split);
}

function predictXgboost(x) {
  let sum = XGB.base_score;
  for (const tree of XGB.trees) sum += predictOneXgbTree(tree, x);
  return sum;
}

// ---------------------------------------------------------------
// 4. Neural Network — dense layers + ReLU (sklearn MLPRegressor default),
//    identity activation on the final output layer (regression).
// ---------------------------------------------------------------
function matVecAdd(W, x, b) {
  // W: [n_in][n_out], x: [n_in], b: [n_out]
  const nOut = b.length;
  const out = new Array(nOut).fill(0);
  for (let j = 0; j < nOut; j++) {
    let s = b[j];
    for (let i = 0; i < x.length; i++) s += W[i][j] * x[i];
    out[j] = s;
  }
  return out;
}

function relu(vec) {
  return vec.map((v) => Math.max(0, v));
}

function predictNeuralNet(x) {
  let activations = x;
  const nLayers = NN.weights.length;
  for (let l = 0; l < nLayers; l++) {
    activations = matVecAdd(NN.weights[l], activations, NN.biases[l]);
    if (l < nLayers - 1) activations = relu(activations); // last layer: identity (regression)
  }
  return activations[0];
}

// ---------------------------------------------------------------
// Router
// ---------------------------------------------------------------
export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (url.pathname === "/health" && request.method === "GET") {
      return jsonResponse({
        status: "ok",
        target: TARGET_COL,
        n_trees_rf: RF.trees.length,
        n_trees_xgb: XGB.trees.length,
      });
    }

    if (url.pathname === "/predict" && request.method === "POST") {
      let body;
      try {
        body = await request.json();
      } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400);
      }

      const missing = FEATURE_NAMES.filter((f) => body[f] === undefined);
      if (missing.length) {
        return jsonResponse({ error: `Missing fields: ${missing.join(", ")}` }, 422);
      }

      const x = scaleInput(body);

      const predictions = {
        random_forest: predictRandomForest(x),
        xgboost: predictXgboost(x),
        neural_network: predictNeuralNet(x),
      };
      const ensemble_mean =
        (predictions.random_forest + predictions.xgboost + predictions.neural_network) / 3;

      return jsonResponse({ target: TARGET_COL, predictions, ensemble_mean });
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
};
